import torch

def get_tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.shape().prod().item()

def overlap(interval_a: tuple[int], interval_b: tuple[int]) -> tuple[int, int] | None:
    start_a, end_a = interval_a
    start_b, end_b = interval_b

    start = max(start_a, start_b)
    end = min(end_a, end_b)

    if start >= end:
        return None
    return (start, end)

def get_slice_numel(_slice: slice) -> int:
    denom = _slice.step if _slice.step is not None else 1
    return (_slice.stop - _slice.start) // denom

def shift_slice(_slice: slice, shift: int) -> slice:
    return slice(_slice.start + shift, _slice.stop + shift, _slice.step)
