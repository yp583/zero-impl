from ast import List
from functools import partial
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.profiler import record_function
from torch.utils.hooks import RemovableHandle

from engine.utils import all_gather_uneven, reduce_scatter_uneven
from engine.sharded_param import (
    ShardedParameterState,
    set_param_meta,
    set_param_materialized,
)
from engine.utils.distributed import rank0_print
from engine.profilers import TensorLifecycleProfiler

def remove_after_called(
        func: Callable[[Sequence[torch.Tensor | None]], None],
        handle_ref: list[RemovableHandle]
    ) -> Callable[[Sequence[torch.Tensor | None]], None]:

    def wrapper(data: Sequence[torch.Tensor | None]) -> None:
        nonlocal handle_ref
        try:
            func(data)
        finally:
            for ref in handle_ref:
                ref.remove()
    return wrapper

def scatter_grads_for_module(grads: Sequence[torch.Tensor | None], module: nn.Module, device: str):
    with record_function("scatter_grads"):
        profiler = TensorLifecycleProfiler.current()
        if profiler:
            profiler.step(f"scatter_start:{module._get_name()}")
        all_states = []
        for grad, param in zip(grads, module.parameters()):
            shard_state: ShardedParameterState = getattr(param, "_shard_state", None)
            if shard_state is None:
                continue
            if grad is not None:
                shard_state.full_grad = grad.flatten()
            all_states.append(shard_state)

        rank = dist.get_rank()
        tensor_list = ShardedParameterState.to_flat_grads(all_states)
        tensor_to_keep = torch.zeros(tensor_list[rank].numel(), device=device)

        reduce_scatter_job = reduce_scatter_uneven(tensor_to_keep, tensor_list, op=dist.ReduceOp.AVG, async_op=True)
        reduce_scatter_job.wait()

        del tensor_list

        curr_numel = 0
        for shard_state in all_states:
            grad_numel = shard_state.get_numel(rank)

            if grad_numel > 0:
                param_grad = tensor_to_keep[curr_numel : curr_numel + grad_numel]
                curr_numel += grad_numel
                if shard_state.materialized is not None:
                    shard_state.materialized.grad = param_grad

            shard_state.full_grad = None

        if profiler:
            profiler.step(f"scatter_end:{module._get_name()}")

def discard_grads_for_module(module: nn.Module, *_):
    for _, param in module.named_parameters():
        if param.grad is not None:
            param.grad.set_(torch.empty(0))

def gather_params_for_module(module: nn.Module, *_, device: str, **__):
    with record_function("gather_params"):
        profiler = TensorLifecycleProfiler.current()
        if profiler:
            profiler.step(f"gather_start:{module._get_name()}")
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

        params = ShardedParameterState.from_flat_params(all_states, tensors_to_gather)

        for materialized, (name, param) in zip(params, module.named_parameters()):
            param = set_param_materialized(module, name, param, materialized.reshape(param.shape))

        if profiler:
            profiler.step(f"gather_end:{module._get_name()}")

        scatter_callback = partial(scatter_grads_for_module, module=module, device=device)
        handle_ref = []
        wrapped_callback = remove_after_called(scatter_callback, handle_ref)
        handle = torch.autograd.graph.register_multi_grad_hook(list(module.parameters()), wrapped_callback)
        handle_ref.append(handle)
        

def discard_params_for_module(module: nn.Module, *_, **__):
    for name, param in module.named_parameters():
        set_param_meta(module, name, param)
