import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

from engine.utils.distributed import rank_print
from engine.utils.math import get_slice_numel


class ShardedParameterState:
    param_meta: nn.Parameter
    rank_intervals: list[slice | None]

    materialized: Optional[nn.Parameter]
    full_grad: Optional[torch.Tensor]

    def __init__(self, param_meta: nn.Parameter, rank_intervals: list[slice | None], materialized: torch.Tensor = None, full_grad: torch.Tensor = None):
        self.param_meta = param_meta
        self.rank_intervals = rank_intervals

        self.materialized = materialized.flatten() if materialized is not None else None
        self.full_grad = full_grad.flatten() if full_grad is not None else None

    def get_numel(self, rank: int) -> int:
        if self.rank_intervals[rank] is None:
            return 0
        return get_slice_numel(self.rank_intervals[rank])

    @staticmethod
    def to_flat_params(shards: list['ShardedParameterState'], device="cpu") -> list[torch.Tensor]:
        world_size = dist.get_world_size()
        this_rank = dist.get_rank()
        flat_data = [[] for _ in range(world_size)]

        for shard in shards:
            for rank in range(world_size):
                interval = shard.rank_intervals[rank]
                if interval is None:
                    continue

                numel = get_slice_numel(interval)
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
    
    @staticmethod
    def to_flat_grads(shards: list['ShardedParameterState']) -> list[torch.Tensor]:
        world_size = dist.get_world_size()
        
        tensor_list = [[] for _ in range(world_size)]

        for shard in shards:
            for rank in range(world_size):
                rank_interval = shard.rank_intervals[rank]
                if rank_interval is None:
                    continue
                tensor_list[rank].append(shard.full_grad[rank_interval])
            
        for rank in range(world_size):
            if len(tensor_list[rank]) > 0:
                tensor_list[rank] = torch.cat(tensor_list[rank])
            else:
                tensor_list[rank] = torch.empty(0)
        
        return tensor_list


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
    if param.device.type == 'meta':
        shard_state = getattr(param, "_shard_state", None)
        new_param = nn.Parameter(data, requires_grad=param.requires_grad)
        if shard_state is not None:
            new_param._shard_state = shard_state
        _set_module_param(module, name, new_param)
    else:
        param.data = data
    
    return _get_module_param(module, name)

