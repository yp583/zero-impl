from collections import deque
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from engine.communication import (
    ShardedModuleState,
    gather_params_for_module,
    discard_params_after_forward,
    gather_grads_for_backward,
    gather_grads_for_backward
)

from engine.utils import has_direct_params, rank0_print

# ------- Hooks -------
def gather_params_for_forward_hook(module, _):
    params = gather_params_for_module(module=module)
    for name, p in params.items():
        parts = name.split('.')
        target = module
        for part in parts[:-1]:
            target = getattr(target, part)
        target._parameters[parts[-1]] = p

def discard_params_after_forward_hook(module, _, __):
    discard_params_after_forward(module)

@dataclass
class ZeroEngineConfig:
    rank: int
    world_size: int
    generator: torch.Generator
    device: str

class ZeroEngine:
    def __init__(self, config: ZeroEngineConfig):
        self.rank = config.rank
        self.world_size = config.world_size
        self.generator = config.generator
        self.device = config.device

        self.original_register = None
        self.hooks = []
        self.leaf_modules = set()

    def _flat_shard_numel(self, rank, numel):
        base = numel // self.world_size
        remainder = numel % self.world_size
        local_numel = base + (1 if rank < remainder else 0)
        return local_numel
    def _flat_shard_fn(self, numel):
        rank_numels = [self._flat_shard_numel(i, numel) for i in range(self.world_size)]
        local = torch.empty(rank_numels[self.rank], device=self.device).view(-1)

        if local.numel() == 0:
            return local, rank_numels

        std = (5 / local.numel()) ** 0.5
        local.uniform_(-std, std, generator=self.generator)

        return local, rank_numels

    def materialize_sharded_params(self, model):
        with torch.no_grad():
            leaf_modules = deque([model])
            while len(leaf_modules) > 0:
                module = leaf_modules.popleft()
                if not has_direct_params(module):
                    leaf_modules.extend(module.children())
                    continue
                
                shard_state = ShardedModuleState(world_size=self.world_size, meta=module, shard_fn=self._flat_shard_fn)
                setattr(module, "_shard_state", shard_state)

                f_pre_hook = module.register_forward_pre_hook(gather_params_for_forward_hook)
                f_post_hook = module.register_forward_hook(discard_params_after_forward_hook)
                self.hooks.append(f_pre_hook)
                self.hooks.append(f_post_hook)
                
    def __enter__(self):
        self.original_register = nn.Module.register_parameter

        def meta_register(module_self, name, param):
            # Only convert non-meta tensors to meta device
            if isinstance(param, nn.Parameter) and param.data.device.type != 'meta':
                meta = nn.Parameter(torch.empty_like(param, device='meta'),
                                    requires_grad=param.requires_grad)
                meta._shape_dtype = (tuple(param.shape), param.dtype)
                return self.original_register(module_self, name, meta)
            return self.original_register(module_self, name, param)

        nn.Module.register_parameter = meta_register
        return self

    def __exit__(self, *args, **kwargs):
        nn.Module.register_parameter = self.original_register
        for hook in self.hooks:
            hook.remove()