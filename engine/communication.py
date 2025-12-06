from torch.nn.parameter import Parameter


from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Callable

from engine.utils import rank0_print


class ShardedModuleState:
    rank_numels: list[int]
    module_meta: nn.Module
    shard: torch.Tensor

    def __init__(self, world_size, meta: nn.Module, shard_fn: Callable[int, tuple[torch.Tensor, list[int]]]):
        self.param_shapes = {}
        total_numel = 0
        # recurse since the meta will be the leaf most module
        for _, param in meta.named_parameters(recurse=True):
            total_numel += param.numel()
        self.shard, self.rank_numels = shard_fn(total_numel)
        self.module_meta = meta

    def from_flat(self, flat_tensors: list[torch.Tensor]):
        full_flat = torch.cat(flat_tensors)
        params = {}
        offset = 0
        for name, param_meta in self.module_meta.named_parameters(recurse=True):
            numel = param_meta.numel()
            params[name] = nn.Parameter(full_flat[offset:offset+numel].view(param_meta.shape), requires_grad=False)
            offset += numel
        return params

def gather_params_for_module(module) -> dict[str, nn.Parameter]:

    shard_state: ShardedModuleState = getattr(module, "_shard_state", None)
    if shard_state is None:
        return {}

    device = shard_state.shard.device
    dtype = shard_state.shard.dtype
    module_params = [torch.empty(n, device=device, dtype=dtype) for n in shard_state.rank_numels]

    all_gather_job = dist.all_gather(module_params, shard_state.shard, async_op=True)

    all_gather_job.wait()

    params = shard_state.from_flat(module_params)
    return params

def discard_params_after_forward(module):
    shard_state: ShardedModuleState = getattr(module, "_shard_state", None)
    if shard_state is None:
        return

    for name, param in module.named_parameters(recurse=True):
        parts = name.split('.')
        target = module
        for part in parts[:-1]:
            target = getattr(target, part)
        target._parameters[parts[-1]] = nn.Parameter(
            torch.empty_like(param, device="meta"),
            requires_grad=param.requires_grad
        )

def gather_grads_for_backward(module):
    pass

def discard_grads_after_backward(module):
    
    pass
