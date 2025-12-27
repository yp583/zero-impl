from typing import TypeAlias
import torch
from typing import Protocol

class ParamProto(Protocol)
    param: torch.nn.Parameter
class ShardedFields(torch.nn.Parameter, Protocol):
    shard_id: int


