import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional


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


def to_flat(shards: list['ShardedParameterState'], device="cpu") -> list[torch.Tensor]:
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
