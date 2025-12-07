from torch.nn.parameter import Parameter


from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Callable

class ShardedModuleState:
    rank_numels: list[int]
    module_meta: nn.Module
    shard: nn.Parameter

    def __init__(self, meta: nn.Module, shard_fn: Callable[int, tuple[nn.Parameter, list[int]]], model_numel: int):
        self.param_shapes = {}
        self.shard, self.rank_numels = shard_fn(model_numel)
        self.module_meta = meta

    def from_flat(self, flat_tensors: list[torch.Tensor]):
        full_flat = torch.cat(flat_tensors)
        params = {}
        offset = 0
        for name, param_meta in self.module_meta.named_parameters(recurse=True):
            numel = param_meta.numel()
            params[name] = nn.Parameter(full_flat[offset:offset+numel].view(param_meta.shape), requires_grad=param_meta.requires_grad)
            offset += numel
        return params

def gather_params_for_module(module, *_, **__):

    shard_state: ShardedModuleState = getattr(module, "_shard_state", None)
    if shard_state is None:
        return {}

    device = shard_state.shard.device
    dtype = shard_state.shard.dtype
    module_params = [torch.empty(n, device=device, dtype=dtype) for n in shard_state.rank_numels]

    all_gather_job = dist.all_gather(module_params, shard_state.shard.data, async_op=True)

    all_gather_job.wait()

    params = shard_state.from_flat(module_params)

    for name, p in params.items():
        parts = name.split('.')
        target = module
        for part in parts[:-1]:
            target = getattr(target, part)
        target._parameters[parts[-1]] = p

def discard_params_for_module(module, *_, **__):
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