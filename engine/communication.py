from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist

from engine.utils import all_gather_uneven, reduce_scatter_uneven
from engine.sharded_param import (
    ShardedParameterState,
    set_param_meta,
    set_param_materialized,
)


def scatter_grads_for_module(grads: list[torch.Tensor], module: nn.Module, device: str):
    all_states = []
    for grad,  param in zip(grads, module.parameters()):
        shard_state: ShardedParameterState = getattr(param, "_shard_state", None)
        if shard_state is None:
            continue
        shard_state.full_grad = grad.flatten()
        all_states.append(shard_state)

    rank = dist.get_rank()
    tensor_list = ShardedParameterState.to_flat_grads(all_states)
    tensor_to_keep = torch.zeros(tensor_list[rank].numel(), device=device)

    reduce_scatter_job = reduce_scatter_uneven(tensor_to_keep, tensor_list, op=dist.ReduceOp.AVG, async_op=True)
    reduce_scatter_job.wait()

    curr_numel = 0
    for param in module.parameters():
        shard_state: ShardedParameterState = getattr(param, "_shard_state", None)
        if shard_state is None:
            continue

        grad_numel = shard_state.get_numel(rank)
        if grad_numel <= 0:
            continue

        param_grad = tensor_to_keep[curr_numel : curr_numel + grad_numel]
        curr_numel += grad_numel
        shard_state.materialized.grad = param_grad
        shard_state.full_grad = None
    


def gather_params_for_module(module: nn.Module, *_, device: str, **__):
    all_states = []
    for param in module.parameters():
        shard_state = getattr(param, "_shard_state", None)
        if shard_state is None:
            continue
        all_states.append(shard_state)

    tensors_to_gather = ShardedParameterState.to_flat_params(all_states, device=device)

    rank = dist.get_rank()
    all_gather_job = all_gather_uneven(tensors_to_gather, tensors_to_gather[rank], async_op=True)
    all_gather_job.wait()
    flat_params = torch.cat(tensors_to_gather)

    curr_offset = 0
    for name, param in module.named_parameters():
        param_data = flat_params[curr_offset:curr_offset+param.numel()].reshape(param.shape)

        param = set_param_materialized(module, name, param, param_data)

        curr_offset += param.numel()

    torch.autograd.graph.register_multi_grad_hook(list(module.parameters()), partial(scatter_grads_for_module, module=module, device=device))


def discard_params_for_module(module: nn.Module, *_, **__):
    all_states = []
    for name, param in module.named_parameters():
        shard_state: ShardedParameterState = getattr(param, "_shard_state", None)
        if shard_state is None:
            continue
        all_states.append(shard_state)

    for name, param in module.named_parameters():
        set_param_meta(module, name, param)

