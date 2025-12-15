import torch
import torch.distributed as dist
from contextlib import ExitStack
from datasets.dev.dev_dataclient import DevDatasetClient
from test.model import TestModel
from engine.zero_init import ZeroEngine, ZeroEngineConfig
from engine.profilers import MemoryProfiler, LossProfiler
from engine.utils import rank0_print
from dotenv import load_dotenv
import os

load_dotenv()

def finalize_dist():
    dist.barrier()
    dist.destroy_process_group()

def dist_train():
    dist.init_process_group(backend=os.getenv("TORCH_BACKEND"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank + 1}/{world_size}] Process initialized successfully!")

    ds_client = DevDatasetClient(rank=rank, world_size=world_size)
    data = ds_client.get_shard()

    device = "cpu"
    generator = torch.Generator(device=device).manual_seed(42)

    zero_config = ZeroEngineConfig(
        generator=generator,
        device=device,
    )

    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    loss_graph_path = os.path.join(graph_dir, f"loss_dist_rank_{rank}.png")

    with ExitStack() as stack:
        mem_profiler = stack.enter_context(MemoryProfiler(graph_folder=graph_dir, profile_name=f"memory_dist_rank_{rank}"))
        ze = stack.enter_context(ZeroEngine(config=zero_config))
        loss_profiler = stack.enter_context(LossProfiler(graph_path=loss_graph_path))

        model = TestModel(input_dim=128, output_dim=128)
        ze.register_model(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        inputs = torch.stack([torch.tensor(dp[0], dtype=torch.float32, device=device) for dp in data])
        labels = torch.tensor([dp[1] for dp in data], dtype=torch.long, device=device)

        loss_fn = torch.nn.CrossEntropyLoss()
        num_epochs = 100
        batch_size = 32

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                out = model.forward(batch_inputs)
                loss = loss_fn(out, batch_labels)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

                loss_profiler.record(loss)
                mem_profiler.step()

            avg_loss = epoch_loss / num_batches
            rank0_print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    finalize_dist()

if __name__ == "__main__":
    dist_train()
