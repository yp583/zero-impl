from typing import List, Tuple
import requests
from abc import ABC, abstractmethod
import os


class DataInterface(ABC):
    @abstractmethod
    def get_shard(self) -> List:
        pass


class BertDatasetClient(DataInterface):
    def __init__(self, rank: int, world_size: int, host_url: str = None):
        self.world_size = world_size
        self.rank = rank
        self.host_url = host_url or os.getenv("BERT_DATASET_HOST_URL", "http://localhost:7778")

    def get_shard(self) -> List[Tuple[List[int], List[int], int]]:
        response = requests.post(
            f"{self.host_url}/get_shard",
            json={"shard_id": self.rank}
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [
            (item["input_ids"], item["attention_mask"], item["label"])
            for item in data
        ]


if __name__ == "__main__":
    client = BertDatasetClient(rank=0, world_size=4)
    try:
        shard = client.get_shard()
        print(f"Shard size: {len(shard)}")
        if shard:
            input_ids, attention_mask, label = shard[0]
            print(f"First item - input_ids length: {len(input_ids)}, label: {label}")
    except Exception as e:
        print(f"Error (is the datahost running?): {e}")
