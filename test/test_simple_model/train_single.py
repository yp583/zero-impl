import torch
from contextlib import ExitStack
from data_sources.dev.dev_dataclient import DevDatasetClient
from test.test_simple_model.model import TestModel
from engine.profilers import PeakMemoryProfiler, LossProfiler, IterationProfiler
from dotenv import load_dotenv
import os

load_dotenv()


def single_train():
    print("Single process training started!")

    ds_client = DevDatasetClient(rank=0, world_size=1)
    data = ds_client.get_shard()

    device = "cpu"
    generator = torch.Generator(device=device).manual_seed(42)

    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    loss_graph_path = os.path.join(graph_dir, "loss_single.png")

    with ExitStack() as stack:
        peak_mem_profiler = stack.enter_context(PeakMemoryProfiler(output_folder=graph_dir, profile_name="peak_memory_single", device=device))
        loss_profiler = stack.enter_context(LossProfiler(graph_path=loss_graph_path))
        iter_profiler = stack.enter_context(IterationProfiler(graph_folder=graph_dir, profile_name="iteration_time_single"))

        model = TestModel(input_dim=128, output_dim=128)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        inputs = torch.stack([torch.tensor(dp[0], dtype=torch.float32, device=device) for dp in data])
        labels = torch.tensor([dp[1] for dp in data], dtype=torch.long, device=device)

        loss_fn = torch.nn.CrossEntropyLoss()
        num_epochs = int(os.getenv("NUM_EPOCHS", 100))
        batch_size = int(os.getenv("BATCH_SIZE", 32))

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
                peak_mem_profiler.step()
                iter_profiler.step()

            avg_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    single_train()
