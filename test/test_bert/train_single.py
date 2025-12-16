import torch
from contextlib import ExitStack
from data_sources.bert.bert_dataclient import BertDatasetClient
from test.test_bert.model import create_bert_model
from engine.profilers import PeakMemoryProfiler, LossProfiler, IterationProfiler
from dotenv import load_dotenv
import os

load_dotenv()

def single_train():
    print("BERT single process training started!")

    ds_client = BertDatasetClient(rank=0, world_size=1)
    data = ds_client.get_shard()

    device = "cpu"

    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    loss_graph_path = os.path.join(graph_dir, "loss_single.png")

    with ExitStack() as stack:
        peak_mem_profiler = stack.enter_context(PeakMemoryProfiler(graph_folder=graph_dir, profile_name="peak_memory_single", device=device))
        loss_profiler = stack.enter_context(LossProfiler(graph_path=loss_graph_path))
        iter_profiler = stack.enter_context(IterationProfiler(graph_folder=graph_dir, profile_name="iteration_time_single"))

        model = create_bert_model()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        input_ids = torch.tensor([dp[0] for dp in data], dtype=torch.long, device=device)
        attention_mask = torch.tensor([dp[1] for dp in data], dtype=torch.long, device=device)
        labels = torch.tensor([dp[2] for dp in data], dtype=torch.long, device=device)

        num_epochs = int(os.getenv("NUM_EPOCHS", 100))
        batch_size = int(os.getenv("BATCH_SIZE", 32))

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(input_ids), batch_size):
                batch_input_ids = input_ids[i:i + batch_size]
                batch_attention_mask = attention_mask[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                )
                loss = outputs.loss

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