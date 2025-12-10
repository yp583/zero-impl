from collections import deque
from dataclasses import dataclass, field
from typing import Iterator
import torch
import torch.nn as nn
from engine.communication import (
    ShardedParameterState,
    gather_params_for_module,
    discard_params_for_module,
)
import torch.distributed as dist

from engine.utils import has_direct_params, overlap, rank0_print
import warnings
warnings.filterwarnings("ignore", message="Full backward hook")

@dataclass
class ZeroEngineConfig:
    generator: torch.Generator
    device: str

class ZeroEngine:
    def __init__(self, config: ZeroEngineConfig):
        self.generator = config.generator
        self.device = config.device

        self.original_register = None
        self.original_optimizer_subclass_init = None
        self.hooks = []

    def register_model(self, model):
        self._assign_hooks(model)
        self._materialize_sharded_params(model)
        
    
    def _assign_hooks(self, model):
        leaf_modules = deque([model])
        while len(leaf_modules) > 0:
            module = leaf_modules.popleft()
            if not has_direct_params(module):
                leaf_modules.extend(module.children())
                continue

            f_pre_hook = module.register_forward_pre_hook(gather_params_for_module)
            f_post_hook = module.register_forward_hook(discard_params_for_module)
            b_pre_hook = module.register_full_backward_pre_hook(gather_params_for_module)
            b_post_hook = module.register_full_backward_hook(discard_params_for_module)

            self.hooks.extend([f_pre_hook, f_post_hook, b_pre_hook, b_post_hook])

    def _materialize_sharded_params(self, model: nn.Module):
        total_numel = sum([param.numel() for param in model.parameters()])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        curr_offset = 0
        rank_offsets = []
        for r in range(world_size):
            rank_offsets.append(curr_offset)
            curr_offset += total_numel // world_size + (total_numel % world_size > r)
        else:
            rank_offsets.append(curr_offset)

        def get_interval_for_rank(rank: int, curr_offset: int, param_numel: int) -> tuple[int]:
            rank_model_interval = (rank_offsets[rank], rank_offsets[rank + 1])
            curr_param_interval = (curr_offset, curr_offset + param_numel)
            overlap_interval = overlap(rank_model_interval, curr_param_interval)

            if overlap_interval is None:
                return None
            
            return overlap_interval

        curr = 0
        for param_meta in model.parameters():
            materialized = None
            rank_param_intervals = []

            for r in range(world_size):
                interval = get_interval_for_rank(r, curr, param_meta.numel())

                if interval is None:
                    rank_param_intervals.append(None)
                    continue

                # Store both start and end for this rank
                if interval is not None:
                    interval = (interval[0] - curr, interval[1] - curr)
                rank_param_intervals.append(interval)
            
                if rank == r:
                    data = torch.empty(interval[1] - interval[0], device=self.device)
                    data.uniform_(-0.05, 0.05, generator=self.generator)
                    materialized = nn.Parameter(data, requires_grad=param_meta.requires_grad)
                    
            sharded_param_state = ShardedParameterState(param_meta=param_meta, materialized=materialized, rank_intervals=rank_param_intervals)
            setattr(param_meta, "_shard_state", sharded_param_state)
        
            curr += param_meta.numel()

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
    
    def _override_optimizer_init(self):
        self.original_optimizer_subclass_init = torch.optim.Optimizer.__init_subclass__

        def optimizer_subclass_init(cls, **kwargs):
            orig_init = cls.__init__

            def get_shard(param: nn.Parameter) -> nn.Parameter | None:
                shard_state: ShardedParameterState = getattr(param, "_shard_state", None)
                if shard_state is None or shard_state.materialized is None:
                    return None
                return shard_state.materialized

            def pass_only_materialized(optim_self, parameters: Iterator[nn.Parameter], *args, **kwargs):
                materialized_parameters = map(get_shard, parameters)
                materialized_parameters = filter(lambda p: p != None, materialized_parameters)
                return orig_init(optim_self, materialized_parameters, *args, **kwargs)
            cls.__init__ = pass_only_materialized
            setattr(cls, "_orig_init", orig_init)
            self.original_optimizer_subclass_init()
        
        @classmethod
        def class_optimizer_subclass_init(cls, **kwargs):
            optimizer_subclass_init(cls, **kwargs)
        torch.optim.Optimizer.__init_subclass__ = class_optimizer_subclass_init
        for cls in torch.optim.Optimizer.__subclasses__():
            optimizer_subclass_init(cls)

    def __enter__(self):
        self._override_param_register()
        self._override_optimizer_init()

        return self

    def __exit__(self, *args, **kwargs):
        nn.Module.register_parameter = self.original_register
        torch.optim.Optimizer.__init_subclass__ = self.original_optimizer_subclass_init
        for cls in torch.optim.Optimizer.__subclasses__():
            orig_init = getattr(cls, "_orig_init", None)
            if orig_init is not None:
                cls.__init__ = orig_init

        for hook in self.hooks:
            hook.remove()