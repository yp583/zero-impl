import torch
import torch.distributed as dist
from datasets.dev.dev_dataclient import DevDatasetClient
from test.model import TestModel
from engine.zero_init import ZeroEngine, ZeroEngineConfig
from dotenv import load_dotenv
import os

load_dotenv()

def finalize_dist():
    dist.barrier()
    dist.destroy_process_group()

def dist_train():
    dist.init_process_group(backend=os.getenv("TORCH_BACKEND"))  # or "nccl" for GPU
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank + 1}/{world_size}] Process initialized successfully!")

    # Get Data
    # HTTP eventually should be streamed during training loop
    ds_client = DevDatasetClient(rank=rank, world_size=world_size)
    data = ds_client.get_shard()

    #init model with the engine context
    zero_config = ZeroEngineConfig(
        rank=rank,
        world_size=world_size,
        seed=42,
        device="cpu",
        bucket_size=int(10e3)
    )
    with ZeroEngine(config=zero_config) as ze:
        model = TestModel()
        total_params = sum(p.numel() for p in model.parameters())
        ze.materialize_sharded_params(model)
        dummy_input = torch.randn(2, 16, device="cpu")
        out = model.forward(dummy_input)
    
    print(f"[Rank {rank + 1}] Model output: {out.shape}")

    #for each
        #forward
        #backward
        #update
    
    
    # Clean up
    finalize_dist()

if __name__ == "__main__":
    dist_train()