import torch

def get_tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.shape().prod().item()