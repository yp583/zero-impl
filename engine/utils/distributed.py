from engine.config import UNEVEN_COMMS
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
            return _UnpadWork(work, tensor_list, padded_tensors, original_sizes)
        else:
            for i, (t, size) in enumerate(zip(padded_tensors, original_sizes)):
                tensor_list[i] = t[:size]
            return work
    else:
        return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)
import torch
import torch.distributed as dist
from engine.config import UNEVEN_COMMS

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

class _UnpadWork:
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

    def __getattr__(self, name):
        return getattr(self._work, name)
