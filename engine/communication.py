from torch.nn.parameter import Parameter


from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Callable, Optional

from engine.utils import rank_print

class ShardedParameterState:
    param_meta: nn.Parameter
    materialized: Optional[torch.Tensor]
    rank_intervals: list[tuple[int, int] | None]

    def __init__(self, param_meta: nn.Parameter, rank_intervals: list[tuple[int, int] | None], materialized: torch.Tensor = None):
        self.param_meta = param_meta
        self.materialized = materialized
        self.rank_intervals = rank_intervals

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
                flat_data[rank].append(shard.materialized)
            else:
                placeholder = torch.empty(numel, device=device, dtype=shard.param_meta.dtype)
                flat_data[rank].append(placeholder)

    for rank in range(world_size):
        if len(flat_data[rank]) > 0:
            flat_data[rank] = torch.cat(flat_data[rank])
        else:
            flat_data[rank] = torch.empty(0, device=device, dtype=shard.param_meta.dtype)
    
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
    rank_print([t.shape for t in tensors_to_gather], tensors_to_gather[rank].shape)

    all_gather_job = dist.all_gather(tensors_to_gather, tensors_to_gather[rank], async_op=True)
    all_gather_job.wait()

    flat_params = torch.cat(tensors_to_gather)

    curr_offset = 0
    for name, param in module.named_parameters():
        param_data = flat_params[curr_offset:curr_offset+param.numel()].reshape(param.shape)
        
        if param.device.type == 'meta':
            parts = name.split('.')
            target = module
            for part in parts[:-1]:
                target = getattr(target, part)
            target._parameters[parts[-1]] = nn.Parameter(param_data, requires_grad=param.requires_grad)
        else:
            param.data = param_data
        
        curr_offset += param.numel()

def discard_params_for_module(module: nn.Module, *_, **__):
    for name, param in module.named_parameters():
        shard_state = getattr(param, "_shard_state", None)
        if shard_state is None:
            continue
            
        parts = name.split('.')
        target = module
        for part in parts[:-1]:
            target = getattr(target, part)
        
        target._parameters[parts[-1]] = nn.Parameter(
            torch.empty_like(param, device="meta"),
            requires_grad=param.requires_grad
        )