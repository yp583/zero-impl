from dataclasses import dataclass
import torch
import torch.nn as nn
from engine.communication import ShardedModuleState, fetch_params_for_module, discard_params_after_forward

@dataclass
class ZeroEngineConfig:
    rank: int
    world_size: int
    seed: int
    device: str
    bucket_size: int #in bytes

def has_direct_params(module: nn.Module) -> bool:
    return len(list(module.parameters(recurse=False))) > 0

class ZeroEngine:
    def __init__(self, config: ZeroEngineConfig):
        self.rank = config.rank
        self.world_size = config.world_size
        self.seed = config.seed
        self.device = config.device

        self.original_register = None
        self.hooks = []

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
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        local.uniform_(-std, std, generator=g)

        return local, rank_numels

    def materialize_sharded_params(self, model):
        with torch.no_grad():
            for _, module in list(model.named_modules()):
                if not has_direct_params(module):
                    continue
                shard_state = ShardedModuleState(world_size=self.world_size, meta=module, shard_fn=self._flat_shard_fn)
                setattr(module, "_shard_state", shard_state)
                
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
        
        def gather_params_for_forward_hook(module, _):
            for m in module.modules():
                params = fetch_params_for_module(rank=self.rank, world_size=self.world_size, module=m)
                for name, p in params.items():
                    m._parameters[name] = p

        def discard_params_after_forward_hook(module, _, __):
            for m in module.modules():
                discard_params_after_forward(m)



        nn.Module.register_parameter = meta_register
        f_pre_hook = nn.modules.module.register_module_forward_pre_hook(gather_params_for_forward_hook)
        f_post_hook = nn.modules.module.register_module_forward_hook(discard_params_after_forward_hook)
        self.hooks.append(f_pre_hook)
        self.hooks.append(f_post_hook)
        return self

    def __exit__(self, *args, **kwargs):
        nn.Module.register_parameter = self.original_register
        for hook in self.hooks:
            hook.remove()