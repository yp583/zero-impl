

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    port = int(os.environ.get("BERT_DATASET_HOST_PORT", 7778))
    max_samples = int(os.environ.get("BERT_MAX_SAMPLES", 1000))

    dataset = BertSentimentDataset(split="train", max_samples=max_samples)
    ds_host = BertDatasetHost(dataset, world_size)

    print(f"Starting BERT dataset host on 0.0.0.0:{port} for world_size={world_size}")
    print(f"Dataset size: {len(dataset)}")
    ds_host.run(host="0.0.0.0", port=port)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

from data_sources.bert.bert_dataset import BertSentimentDataset


class ShardRequest(BaseModel):
    shard_id: int


class BertDatasetHost:
    def __init__(self, dataset, world_size):
        self.dataset = dataset
        self.world_size = world_size
        self.app = FastAPI()

        @self.app.post("/get_shard")
        async def get_shard(request: ShardRequest):
            try:
                shard_id = request.shard_id
                world_size = self.world_size

                assert 0 <= shard_id < world_size, "Invalid shard_id"
                total = len(self.dataset)
                shard_indices = range(shard_id, total, world_size)

                shard_data = [self.dataset[i] for i in shard_indices]

                return {
                    "shard_id": shard_id,
                    "data": [
                        {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "label": label,
                        }
                        for input_ids, attention_mask, label in shard_data
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

    def run(self, host: str = "0.0.0.0", port: int = 7778):
        uvicorn.run(self.app, host=host, port=port)
