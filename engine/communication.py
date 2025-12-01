from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Callable


# what do we need
# we need to know the shape of the tensor (bucket) being broadcasted from other ranks. 
# approaches
# have metadata attached to the module during init that will have rank -> numel of the param so that we can broadcast into it
    # store N tensors for each ranks bucket of the params
# need to do N broadcasts where N is the number of ranks (unavoidable)
class ShardedParamState:
    rank_numels: list[int]
    materialized_params: dict[str, nn.Parameter]

    def __init__(self, world_size):
        self.rank_numels = [0] * world_size
        self.materialized_params = {}

    def add_param(self, rank, name: str, param: nn.Parameter, shard_fn: Callable[torch.Tensor, list[int]]):

        shape, dtype = param._shape_dtype
        local, rank_numels = shard_fn(shape)
        local = local.to(dtype=dtype)

        self.rank_numels = [a + b for a, b in zip(self.rank_numels, rank_numels)]

        local_param = nn.Parameter(local, requires_grad=param.requires_grad)
        self.materialized_params[name] = local_param
def fetch_params_for_module(rank, world_size, module) -> dict[str, nn.Parameter]:

    shard_state: ShardedParamState = getattr(module, "_shard_state", None)
    if shard_state is None:
        return

    module_params = [torch.empty(n) for n in shard_state.rank_numels]

    broadcast_params_list = []

    for name, _ in module.named_parameters():
        materialized_param = shard_state.materialized_params.get(name, None)
        if materialized_param is not None:
            broadcast_params_list.append(materialized_param.data)
            
        
    
    broadcast_params = torch.Tensor([])
    if len(broadcast_params_list) > 0:
        broadcast_params = torch.cat(broadcast_params_list)
    module_params[rank] = broadcast_params

    broadcasts = []
    for r in range(world_size):
        broadcasts.append(dist.broadcast(module_params[r], src=r, async_op=True))
    
    for b in broadcasts:
        b.wait()

    total_numel = sum(p.numel() for p in module_params)
    shape_prods = sum(p.numel() for p in module.parameters(recurse=False))
    print(f"[Rank {rank}] Module '{module.__class__.__name__}' total module_params numel: {total_numel}, prod of shapes: {shape_prods}")



def discard_params_after_forward(module):
    # discard all params besides the ones for this rank
    pass

def gather_grads_for_backward(module):
    pass

def discard_grads_after_backward(module):
    
    pass
