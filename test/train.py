import torch
import torch.distributed as dist
from contextlib import ExitStack
from datasets.dev.dev_dataclient import DevDatasetClient
from test.model import TestModel
from engine.zero_init import ZeroEngine, ZeroEngineConfig
from engine.profilers import MetaParamCounter, LossProfiler
from engine.utils import get_shard_numels, graph_module, rank0_print, rank_print
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
        generator=generator,
        device=device,
    )
    
    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    loss_graph_path = os.path.join(graph_dir, f"loss_rank_{rank}.png")

    with ExitStack() as stack:
        profiler = stack.enter_context(MetaParamCounter(graph_folder=graph_dir))
        ze = stack.enter_context(ZeroEngine(config=zero_config))
        loss_profiler = stack.enter_context(LossProfiler(graph_path=loss_graph_path))

        model = TestModel(input_dim=128, output_dim=128)
        profiler.register_model(f"rank_{rank}", model)
        ze.register_model(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # rank_print(get_shard_numels(model))

        # Convert dataset to tensors
        dataset_tensors = []
        for data_point, label in data:
            input_tensor = torch.tensor(data_point, dtype=torch.float32, device=device)
            label_tensor = torch.tensor(label, dtype=torch.long, device=device)
            dataset_tensors.append((input_tensor, label_tensor))

        loss_fn = torch.nn.CrossEntropyLoss()
        num_epochs = 10

        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for idx, (input_data, target) in enumerate(dataset_tensors):
                # Forward pass
                out = model.forward(input_data)
                loss = loss_fn(out, target.unsqueeze(0))

                # Backward pass
                loss.backward()

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

                loss_profiler.record(loss)

            avg_loss = epoch_loss / len(dataset_tensors)
            rank_print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    #for each
        #forward
        #backward
        #update
    
    
    # Clean up
    finalize_dist()

if __name__ == "__main__":
    dist_train()