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
    to_flat,
)


def reduce_scatter_grads(grad, st: ShardedParameterState):
    flat_grad = grad.flatten()
    rank = dist.get_rank()
    rank_grads = [
        flat_grad[start:end] if interval is not None else torch.empty(0, dtype=flat_grad.dtype, device=flat_grad.device)
        for interval in st.rank_intervals
        for (start, end) in [interval] if interval is not None or True
    ]
    this_grad = torch.empty(st.rank_intervals[rank][1] - st.rank_intervals[rank][0])
    dist.reduce_scatter(this_grad, rank_grads, op=dist.ReduceOp.SUM)


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
                grad_hook = partial(reduce_scatter_grads, st=shard_state)
                param.register_hook(grad_hook)

        curr_offset += param.numel()


def discard_params_for_module(module: nn.Module, *_, **__):
    for name, param in module.named_parameters():
        set_param_meta(module, name, param)


def discard_params_for_module_backwards(module: nn.Module, *_, **__):
    for name, param in module.named_parameters():
        set_param_meta(module, name, param)
