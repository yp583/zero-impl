
if __name__ == "__main__":
    single_train()

def single_train():
    print("Single process training started!")

    ds_client = DevDatasetClient(rank=0, world_size=1)
    data = ds_client.get_shard()

    device = "cpu"
    generator = torch.Generator(device=device).manual_seed(42)

    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    loss_graph_path = os.path.join(graph_dir, "loss_single.png")

    with ExitStack() as stack:
        mem_profiler = stack.enter_context(MemoryProfiler(graph_folder=graph_dir, profile_name="memory_single"))
        loss_profiler = stack.enter_context(LossProfiler(graph_path=loss_graph_path))

        model = TestModel(input_dim=128, output_dim=128)

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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
import torch
from contextlib import ExitStack
from datasets.dev.dev_dataclient import DevDatasetClient
from test.model import TestModel
from engine.profilers import MemoryProfiler, LossProfiler
from dotenv import load_dotenv
import os

load_dotenv()
