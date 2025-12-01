import torch
import torch.nn as nn

class ZeroEngine:
    def __init__(self, rank, world_size, seed, device):
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.device = device
        self.original_register = None
        self.hooks = []

    # Add more functions for other types of sharding like by expert 
    def _flat_shard_fn(self, shape):
        numel = torch.Size(shape).numel()
        base = numel // self.world_size
        remainder = numel % self.world_size
        local_numel = base + (1 if self.rank < remainder else 0)
        return torch.empty(local_numel, device=self.device).view(-1)

    def materialize_sharded_params(self, model):
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        with torch.no_grad():
            for name, p in list(model.named_parameters()):
                shape, dtype = p._shape_dtype
                local = self._flat_shard_fn(shape).to(dtype=dtype, device=self.device)

                if local.numel() == 0:
                    continue

                std = (5 / local.numel()) ** 0.5
                local.uniform_(-std, std, generator=g)

                param = nn.Parameter(local, requires_grad=p.requires_grad)
                *path, param_name = name.split('.')
                parent = model
                for attr in path:
                    parent = getattr(parent, attr)
                #bypass parameter registration in pytorch module
                object.__setattr__(parent, f"materialized_{param_name}", param)

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