from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Type
import torch
import torch.nn as nn
from engine.communication import (
    ShardedModuleState,
    gather_params_for_module,
    discard_params_for_module,
)

from engine.utils import has_direct_params, rank0_print
import warnings
warnings.filterwarnings("ignore", message="Full backward hook")

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


    def _flat_shard_fn(self, numel):
        def _flat_shard_numel(rank, numel):
            base = numel // self.world_size
            remainder = numel % self.world_size
            local_numel = base + (1 if rank < remainder else 0)
            return local_numel
        
        rank_numels = [_flat_shard_numel(i, numel) for i in range(self.world_size)]
        param_data = torch.empty(rank_numels[self.rank], device=self.device).view(-1)

        if param_data.numel() == 0:
            return nn.Parameter(param_data), rank_numels

        std = (5 / param_data.numel()) ** 0.5
        param_data.uniform_(-std, std, generator=self.generator)

        param = nn.Parameter(param_data)
        return param, rank_numels

    def register_model(self, model):
        self._materialize_sharded_params(model)
        
    def _materialize_sharded_params(self, model):
        materialized_params = []
        with torch.no_grad():
            leaf_modules = deque([model])
            while len(leaf_modules) > 0:
                module = leaf_modules.popleft()
                if not has_direct_params(module):
                    leaf_modules.extend(module.children())
                    continue
                shard_state = ShardedModuleState(meta=module, shard_fn=self._flat_shard_fn)
                materialized_params.append(shard_state.shard)
                setattr(module, "_shard_state", shard_state)

                f_pre_hook = module.register_forward_pre_hook(gather_params_for_module)
                f_post_hook = module.register_forward_hook(discard_params_for_module)
                b_pre_hook = module.register_full_backward_pre_hook(gather_params_for_module)
                b_post_hook = module.register_full_backward_hook(discard_params_for_module)

                self.hooks.extend([f_pre_hook, f_post_hook, b_pre_hook, b_post_hook])
        return materialized_params
    

    def _override_param_register(self):
        self.original_register = nn.Module.register_parameter

        def meta_register(module_self, name, param):
            if isinstance(param, nn.Parameter) and param.data.device.type != 'meta':
                meta = nn.Parameter(torch.empty_like(param, device='meta'),
                                    requires_grad=param.requires_grad)
                meta._shape_dtype = (tuple(param.shape), param.dtype)
                return self.original_register(module_self, name, meta)
            return self.original_register(module_self, name, param)

        nn.Module.register_parameter = meta_register
    
    # def _override_optimizer_init(self):
    #     @classmethod
    #     def optimizer_subclass_init(cls, **kwargs):
    #         orig_init = cls.__init__
    #         def pass_only_materialized(optim_self, parameters, *args, **kwargs):
    #             # INSERT_YOUR_CODE
    #             # Filter out parameters whose device is meta
    #             if isinstance(parameters, torch.nn.Module):
    #                 # If a module was passed, get its parameters
    #                 parameters = list(parameters.parameters())
    #             # If parameters is an iterator, convert to list so we can filter
    #             parameters = list(parameters)
    #             filtered_parameters = [p for p in parameters if getattr(p, 'device', None) is not None and p.device.type != 'meta']
    #             return orig_init(optim_self, filtered_parameters, *args, **kwargs)
    #             orig_init()


    #         cls.__init__ = 
    #     torch.optim.Optimizer.__init_subclass__ = optimizer_subclass_init

    def __enter__(self):
        self._override_param_register()

        return self

    def __exit__(self, *args, **kwargs):
        nn.Module.register_parameter = self.original_register
        for hook in self.hooks:
            hook.remove()