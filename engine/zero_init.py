from dataclasses import dataclass
import torch
import torch.nn as nn
from engine.communication import ShardedParamState, fetch_params_for_module

@dataclass
class ZeroEngineConfig:
    rank: int
    world_size: int
    seed: int
    device: str
    bucket_size: int #in bytes

class ZeroEngine:
    def __init__(self, config: ZeroEngineConfig):
        self.rank = config.rank
        self.world_size = config.world_size
        self.seed = config.seed
        self.device = config.device
        self.bucket_size = config.bucket_size

        self.original_register = None
        self.hooks = []

    def _flat_shard_numel(self, rank, shape):
        numel = torch.Size(shape).numel()
        base = numel // self.world_size
        remainder = numel % self.world_size
        local_numel = base + (1 if rank < remainder else 0)
        return local_numel
    # Add more functions for other types of sharding like by expert 
    def _flat_shard_fn(self, shape):
        rank_numels = [self._flat_shard_numel(i, shape) for i in range(self.world_size)]
        local = torch.empty(rank_numels[self.rank], device=self.device).view(-1)

        if local.numel() == 0:
            return local, rank_numels

        std = (5 / local.numel()) ** 0.5
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        local.uniform_(-std, std, generator=g)

        return local, rank_numels

    def materialize_sharded_params(self, model):
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        with torch.no_grad():
            for name, p in list(model.named_parameters()):
                *path, param_name = name.split('.')
                parent = model
                for attr in path:
                    parent = getattr(parent, attr)
                
                shard_state = getattr(parent, "_shard_state", None)
                if shard_state is None:
                    shard_state = ShardedParamState(world_size=self.world_size)
                    setattr(parent, "_shard_state", shard_state)
                
                shard_state.add_param(self.rank, param_name, p, self._flat_shard_fn)

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
        
        def gather_params_for_forward(module, input):
            print(f"Rank: {self.rank} Shape: ", module)


        nn.Module.register_parameter = meta_register
        f_pre_hook = nn.modules.module.register_module_forward_pre_hook(gather_params_for_forward)
        self.hooks.append(f_pre_hook)
        return self

    def __exit__(self, *args, **kwargs):
        nn.Module.register_parameter = self.original_register
        for hook in self.hooks:
            hook.remove()