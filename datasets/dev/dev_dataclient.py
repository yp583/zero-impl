from typing import Iterable
import requests
from abc import ABC, abstractmethod
import os


class DataInterface(ABC):
    @abstractmethod
    def get_shard(rank: int) -> Iterable:
        pass


class DevDatasetClient(DataInterface):
    def __init__(self, rank: int, world_size: int, host_url: str = None):
        self.world_size = world_size
        self.rank = rank
        self.host_url = host_url or os.getenv("DATASET_HOST_URL", "http://localhost:7777")

    def get_shard(self):
        response = requests.post(
            f"{self.host_url}/get_shard",
            json={"shard_id": self.rank, "world_size": self.world_size}
        )
        response.raise_for_status()
        return response.json()["data"]
