import torch
import torch.distributed as dist
from datasets.dev.dev_dataclient import DevDatasetClient
from test.model import TestModel
from engine.zero_init import ZeroEngine, ZeroEngineConfig
from engine.profiler import MetaParamCounter
from engine.utils import get_shard_numels, graph_module
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
    # HTTP eventually should be streamed during training loop by the engine
    ds_client = DevDatasetClient(rank=rank, world_size=world_size)
    data = ds_client.get_shard()

    #init model with the engine context
    device = "cpu"
    generator = torch.Generator(device=device).manual_seed(42)

    zero_config = ZeroEngineConfig(
        rank=rank,
        world_size=world_size,
        generator=generator,
        device=device,
    )
    
    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    with MetaParamCounter(graph_folder=graph_dir) as profiler:
        with ZeroEngine(config=zero_config) as ze:
            model = TestModel()
            profiler.register_model(f"rank_{rank}", model)
            ze.register_model(model)
            print(f"[Rank {rank + 1}]: {get_shard_numels(model)}")

            dummy_input = torch.rand(2, 16, device=device, generator=generator)
            target = torch.rand(2, 4, device=device, generator=generator)

            out = model.forward(dummy_input)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(out, target)
            loss.backward()
    
    print(f"[Rank {rank + 1}] Model output: {out.shape}")



    #for each
        #forward
        #backward
        #update
    
    
    # Clean up
    finalize_dist()

if __name__ == "__main__":
    dist_train()