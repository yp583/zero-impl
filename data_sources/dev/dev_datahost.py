from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import uvicorn
import os

from data_sources.random import MyRandomDataset


class ShardRequest(BaseModel):
    shard_id: int

class DevDatasetHost:
    def __init__(self, dataset, world_size):
        self.dataset = dataset
        self.world_size = world_size
        self.app = FastAPI()

        @self.app.get("/health", status_code=status.HTTP_200_OK, tags=["healthcheck"])
        async def health_check():
            return {"status": "healthy"}

        @self.app.post("/get_shard")
        async def get_shard(request: ShardRequest):
            try:
                shard_id = request.shard_id
                world_size = self.world_size

                assert 0 <= shard_id < world_size, "Invalid shard_id"
                total = len(self.dataset)
                # Interleave indices: ith shard gets every ith index, starting from shard_id
                shard_indices = range(shard_id, total, world_size)

                shard_data = [self.dataset[i] for i in shard_indices]

                return {
                    "shard_id": shard_id,
                    "data": [(data.tolist(), int(label)) for data, label in shard_data]
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

    def run(self, host: str = "0.0.0.0", port: int = 7777):
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    # Read configuration from environment variables
    world_size = int(os.environ["WORLD_SIZE"])
    port = int(os.environ["DATASET_HOST_PORT"])

    # Create dataset and host
    rnd_ds = MyRandomDataset(in_dim=128, out_dim=128, num_samples=1000)
    ds_host = DevDatasetHost(rnd_ds, world_size)

    print(f"Starting dataset host on 0.0.0.0:{port} for world_size={world_size}")
    ds_host.run(host="0.0.0.0", port=port)
