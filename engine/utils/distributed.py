from datetime import timedelta
from sympy.core.function import BadArgumentsError
import torch
import torch.distributed as dist
from engine.config import UNEVEN_COMMS

class _UnpadWorkGather:
    def __init__(self, work, tensor_list, padded_tensors, original_sizes):
        self._work = work
        self._tensor_list = tensor_list
        self._padded_tensors = padded_tensors
        self._original_sizes = original_sizes
        self._completed = False

    def wait(self):
        if not self._completed:
            self._work.wait()
            for i, (t, size) in enumerate(zip(self._padded_tensors, self._original_sizes)):
                self._tensor_list[i] = t[:size]
            self._completed = True

    def is_completed(self):
        if self._completed:
            return True
        if self._work.is_completed():
            self.wait()
            return True
        return False
class _UnpadWorkScatter:
    def __init__(self, work, output_tensor, padded_output, original_size):
        self._work = work
        self._output_tensor = output_tensor
        self._padded_output = padded_output
        self._original_size = original_size
        self._completed = False

    def wait(self):
        if not self._completed:
            self._work.wait()
            self._output_tensor.copy_(self._padded_output[:self._original_size])
            self._completed = True

    def is_completed(self):
        if self._completed:
            return True
        if self._work.is_completed():
            self.wait()
            return True
        return False

    def __getattr__(self, name):
        return getattr(self._work, name)

def all_gather_uneven(tensor_list, tensor, group=None, async_op=False):
    rank = dist.get_rank(group)
    backend = dist.get_backend(group)
    supports_uneven = UNEVEN_COMMS.get(backend, False)

    if not supports_uneven:
        original_sizes = [t.numel() for t in tensor_list]
        max_size = max(original_sizes)

        padded_tensors = []
        for t in tensor_list:
            if t.numel() < max_size:
                padding = torch.zeros(max_size - t.numel(), device=t.device, dtype=t.dtype)
                padded_tensors.append(torch.cat([t, padding]))
            else:
                padded_tensors.append(t)

        work = dist.all_gather(padded_tensors, padded_tensors[rank], group=group, async_op=async_op)

        if async_op:
            return _UnpadWorkGather(work, tensor_list, padded_tensors, original_sizes)
        else:
            for i, (t, size) in enumerate(zip(padded_tensors, original_sizes)):
                tensor_list[i] = t[:size]
            return work
    else:
        return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)

def reduce_scatter_uneven(output_tensor: torch.Tensor, input_tensors: list[torch.Tensor], op: dist.ReduceOp = dist.ReduceOp.SUM, group=None, async_op=False):
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    backend = dist.get_backend(group)
    supports_uneven = UNEVEN_COMMS.get(backend, False)

    if output_tensor.numel() != input_tensors[rank].numel():
        raise ValueError(f"Output tensor for rank {rank} has {output_tensor.numel()} elements, but input_tensors[{rank}] has {input_tensors[rank].numel()} elements.")

    if world_size != len(input_tensors):
        raise ValueError(f"Mismatch between world_size ({world_size}) and length of input_tensors ({len(input_tensors)}). input_tensors must have one tensor per rank.")

    if supports_uneven:
        return dist.reduce_scatter(output=output_tensor, input_list=input_tensors, op=op, group=group, async_op=async_op)

    device = input_tensors[0].device
    dtype = input_tensors[0].dtype
    original_size = output_tensor.numel()
    max_numel = max(t.numel() for t in input_tensors)

    padded_tensors = [torch.zeros(max_numel, device=device, dtype=dtype) for _ in range(world_size)]
    padded_output = torch.empty(max_numel, device=device, dtype=dtype)

    for r in range(world_size):
        padded_tensors[r][:input_tensors[r].numel()] = input_tensors[r]

    work = dist.reduce_scatter(output=padded_output, input_list=padded_tensors, op=op, group=group, async_op=async_op)

    if async_op:
        return _UnpadWorkScatter(work, output_tensor, padded_output, original_size)

    output_tensor.copy_(padded_output[:original_size])
    return work

def rank0_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def rank_print(*args, rank_filter=None, **kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank_filter is not None and rank not in rank_filter:
            return
    else:
        rank = 0
    print(f"[Rank {rank}]", *args, **kwargs)

