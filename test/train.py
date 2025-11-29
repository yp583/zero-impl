from pyexpat import model
import torch.distributed as dist
from datasets.dev.dev_dataclient import DevDatasetClient


def finalize_dist():
    dist.barrier()
    dist.destroy_process_group()

def dist_train():
    dist.init_process_group(backend="gloo")  # or "nccl" for GPU
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}/{world_size}] Process initialized successfully!")

    # Get Data
    # HTTP eventually should be streamed during training loop
    ds_client = DevDatasetClient(rank=rank, world_size=world_size)
    data = ds_client.get_shard()

    #init model with the engine context

    #for each
        #forward
        #backward
        #update
    
    
    # Clean up
    finalize_dist()

if __name__ == "__main__":
    dist_train()