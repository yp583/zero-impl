from torch.nn.parameter import Parameter


from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Callable, Optional

from engine.utils import all_gather_uneven, reduce_scatter_uneven

class ShardedParameterState:
    param_meta: nn.Parameter
    materialized: Optional[nn.Parameter]
    rank_intervals: list[tuple[int, int] | None]

    def __init__(self, param_meta: nn.Parameter, rank_intervals: list[tuple[int, int] | None], materialized: torch.Tensor = None):
        self.param_meta = param_meta
        self.materialized = materialized
        self.rank_intervals = rank_intervals

def _get_module_param(module: nn.Module, name: str) -> nn.Parameter:
    parts = name.split('.')
    target = module
    for part in parts[:-1]:
        target = getattr(target, part)
    return target._parameters[parts[-1]]

def _set_module_param(module: nn.Module, name: str, new_param: nn.Parameter):
    parts = name.split('.')
    target = module
    for part in parts[:-1]:
        target = getattr(target, part)
    target._parameters[parts[-1]] = new_param

def set_param_meta(module: nn.Module, name: str, param: nn.Parameter):
    shard_state = getattr(param, "_shard_state", None)
    new_param = nn.Parameter(
        torch.empty_like(param, device="meta"),
        requires_grad=param.requires_grad
    )
    if shard_state is not None:
        new_param._shard_state = shard_state
    _set_module_param(module, name, new_param)

def set_param_materialized(module: nn.Module, name: str, param: nn.Parameter, data: torch.Tensor):
    shard_state = getattr(param, "_shard_state", None)
    new_param = nn.Parameter(data, requires_grad=param.requires_grad)
    if shard_state is not None:
        new_param._shard_state = shard_state
    _set_module_param(module, name, new_param)

def to_flat(shards: list[ShardedParameterState], device="cpu") -> list[torch.Tensor]:
    world_size = dist.get_world_size()
    this_rank = dist.get_rank()
    flat_data = [[] for _ in range(world_size)]

    for shard in shards:
        for rank in range(world_size):
            interval = shard.rank_intervals[rank]
            if interval is None:
                continue
            
            st, end = interval

            numel = end - st
            if rank == this_rank:
                flat_data[rank].append(shard.materialized.data)
            else:
                placeholder = torch.empty(numel, device=device, dtype=shard.param_meta.dtype)
                flat_data[rank].append(placeholder)

    for rank in range(world_size):
        if len(flat_data[rank]) > 0:
            flat_data[rank] = torch.cat(flat_data[rank])
        else:
            flat_data[rank] = torch.empty(0, device=device)
    
    return flat_data



def gather_params_for_module(module: nn.Module, *_, **__):

    all_states = []
    for param in module.parameters():
        shard_state = getattr(param, "_shard_state", None)
        if shard_state is None:
            continue
        all_states.append(shard_state)
    
    tensors_to_gather = to_flat(all_states)

    rank = dist.get_rank()

    all_gather_job = all_gather_uneven(tensors_to_gather, tensors_to_gather[rank], async_op=True)
    all_gather_job.wait()

    flat_params = torch.cat(tensors_to_gather)

    curr_offset = 0
    for name, param in module.named_parameters():
        param_data = flat_params[curr_offset:curr_offset+param.numel()].reshape(param.shape)

        if param.device.type == 'meta':
            set_param_materialized(module, name, param, param_data)
            param = _get_module_param(module, name)
        else:
            param.data = param_data

        shard_state = getattr(param, "_shard_state", None)
        if shard_state is not None:
            interval = shard_state.rank_intervals[rank]
            if interval is not None:
                def grad_hook(grad, st=shard_state, intv=interval):
                    start, end = intv
                    st.materialized.grad = grad.flatten()[start:end]
                param.register_hook(grad_hook)

        curr_offset += param.numel()

def discard_params_for_module(module: nn.Module, *_, **__):
    for name, param in module.named_parameters():
        set_param_meta(module, name, param)

def discard_params_for_module_backwards(module: nn.Module, *_, **__):
    for name, param in module.named_parameters():
        set_param_meta(module, name, param)
