from collections import deque
from curses import initscr
from dataclasses import dataclass
from functools import partial
from typing import Iterator
import math
import torch
import torch.nn as nn
from engine.communication import (
    gather_params_for_module,
    discard_params_for_module,
    discard_grads_for_module
)
from engine.profilers.mem_profiler import PeakMemoryProfiler
from engine.sharded_param import ShardedParameterState, _set_module_param
import torch.distributed as dist

from engine.utils import has_direct_params, overlap, rank0_print
import warnings

from engine.utils.distributed import rank_print
from engine.utils.math import get_slice_numel, shift_slice
warnings.filterwarnings("ignore", message="Full backward hook")


@dataclass
class ZeroEngineConfig:
    device: str

    prefetch_aggressiveness: int = 1
    debug: bool = False

class ZeroEngine:
    def __init__(self, config: ZeroEngineConfig):
        self.device = config.device
        self.prefetch_aggressiveness = config.prefetch_aggressiveness

        self.debug = config.debug

        self.original_register = None
        self.original_optimizer_subclass_init = None
        self.hooks = []

    def register_model(self, model):
        self._assign_hooks(model)
        self._materialize_sharded_params(model)
        
    def _assign_hooks(self, model):
        forward_gather = partial(gather_params_for_module, device=self.device)
        forward_discard = discard_params_for_module
        backward_discard = discard_grads_for_module

        leaf_modules = deque([model])
        while len(leaf_modules) > 0:
            module = leaf_modules.popleft()
            if not has_direct_params(module):
                leaf_modules.extend(module.children())
                continue

            f_pre_hook = module.register_forward_pre_hook(forward_gather)
            f_post_hook = module.register_forward_hook(forward_discard)
            b_post_hook = module.register_full_backward_hook(backward_discard)


            self.hooks.extend([f_pre_hook, f_post_hook, b_post_hook])

    @torch.no_grad()
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

        def get_interval_for_rank(rank: int, curr_offset: int, param_numel: int) -> slice | None:
            rank_model_interval: tuple[int, int] = (rank_offsets[rank], rank_offsets[rank + 1])
            curr_param_interval: tuple[int, int] = (curr_offset, curr_offset + param_numel)
            overlap_interval = overlap(rank_model_interval, curr_param_interval)

            if overlap_interval is None:
                return None
            
            return slice(overlap_interval[0], overlap_interval[1])

        curr_numel = 0
        rank_numels_materialized = [0 for i in range(world_size)]
        for name, param in list(model.named_parameters()):
            materialized = None
            rank_param_intervals = []
            flat_param = param.flatten()
            param_meta = nn.Parameter(torch.empty_like(param, device='meta'))

            for r in range(world_size):
                interval = get_interval_for_rank(r, curr_numel, param_meta.numel())

                if interval is None:
                    rank_param_intervals.append(None)
                    continue

                adj_interval = shift_slice(interval, -curr_numel)
                rank_param_intervals.append(adj_interval)
            
                numel = get_slice_numel(adj_interval)
                rank_numels_materialized[r] += numel
                if rank == r:
                    owned_numels = flat_param[adj_interval].clone()
                    materialized = nn.Parameter(owned_numels, requires_grad=param.requires_grad)
                                        
            sharded_param_state = ShardedParameterState(
                param_meta=param_meta, 
                materialized=materialized,
                rank_intervals=rank_param_intervals
            )
            setattr(param_meta, "_shard_state", sharded_param_state)
            _set_module_param(model, name, param_meta)
            curr_numel += param_meta.numel()

    # Currently only works for vector based optimizers (ie not Muon)
    def _override_optimizer_init(self):
        self.original_optimizer_subclass_init = torch.optim.Optimizer.__init_subclass__

        def optimizer_subclass_init(cls, **kwargs):
            orig_init = cls.__init__

            def get_shard(param: nn.Parameter) -> torch.Tensor | None:
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
            if self.original_optimizer_subclass_init is not None:  
                self.original_optimizer_subclass_init()
        
        @classmethod
        def class_optimizer_subclass_init(cls, **kwargs):
            optimizer_subclass_init(cls, **kwargs)
        torch.optim.Optimizer.__init_subclass__ = class_optimizer_subclass_init
        for cls in torch.optim.Optimizer.__subclasses__():
            optimizer_subclass_init(cls)

    def __enter__(self):
        # self._override_param_register()
        self._override_optimizer_init()
        return self

    def __exit__(self, *args, **kwargs):
        assert self.original_optimizer_subclass_init is not None
        # assert self.original_register is not None
        # nn.Module.register_parameter = self.original_register
        torch.optim.Optimizer.__init_subclass__ = self.original_optimizer_subclass_init
        for cls in torch.optim.Optimizer.__subclasses__():
            orig_init = getattr(cls, "_orig_init", None)
            if orig_init is not None:
                cls.__init__ = orig_init

        for hook in self.hooks:
            hook.remove()
