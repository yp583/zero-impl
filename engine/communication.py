from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist

from engine.utils import all_gather_uneven
from engine.sharded_param import (
    ShardedParameterState,
    _get_module_param,
    set_param_meta,
    set_param_materialized,
)
from engine.utils.distributed import rank_print, uneven_reduce_scatter


def set_grad(grad: torch.Tensor, st: ShardedParameterState):
    st.full_grad = grad.flatten()


def gather_params_for_module(module: nn.Module, *_, device: str, backwards: bool = False, **__):
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
        shard_state = getattr(param, "_shard_state", None)
        if shard_state is None:
            continue
        grad_hook = partial(set_grad, st=shard_state)
        param.register_hook(grad_hook)

def discard_params_for_module(module: nn.Module, *_, device: str, backwards: bool = False, **__):
    all_states = []
    for name, param in module.named_parameters():
        shard_state: ShardedParameterState = getattr(param, "_shard_state", None)
        if shard_state is None:
            continue
        all_states.append(shard_state)

    if backwards:
        rank = dist.get_rank()
        tensor_list = ShardedParameterState.to_flat_grads(all_states)
        tensor_to_keep = torch.zeros(tensor_list[rank].numel(), device=device)

        reduce_scatter_job = uneven_reduce_scatter(tensor_to_keep, tensor_list, op=dist.ReduceOp.AVG, async_op=True)
        reduce_scatter_job.wait()

    curr_numel = 0
    for name, param in module.named_parameters():
        if backwards:
            shard_state: ShardedParameterState = getattr(param, "_shard_state", None)
            if shard_state is None:
                continue

            grad_numel = shard_state.get_numel(rank)
            if grad_numel <= 0:
                continue

            param_grad = tensor_to_keep[curr_numel : curr_numel + grad_numel]
            curr_numel += grad_numel
            shard_state.materialized.grad = param_grad

        set_param_meta(module, name, param)
