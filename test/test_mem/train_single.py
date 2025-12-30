import torch
from contextlib import ExitStack
from data_sources.dev.dev_dataclient import DevDatasetClient
from test.test_mem.model import TestModel
from engine.profilers import PeakMemoryProfiler
from dotenv import load_dotenv
import os

load_dotenv()

def single_train():
    print("LOSS single process training started!")

    ds_client = DevDatasetClient(rank=0, world_size=1)
    data = ds_client.get_shard()

    torch.manual_seed(42)
    device = "cpu"

    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

    with ExitStack() as stack:
        peak_mem_profiler = stack.enter_context(PeakMemoryProfiler(
            output_folder=graph_dir,
            profile_name="peak_memory_single",
            device=device,
            export_memory_timeline=True,
            clear_logs=True,
        ))
        

        model = TestModel(input_dim=128, hidden_dim=64, output_dim=128)
        print("[NORM OF INITED PARAMS]: ", sum([torch.norm(param, p=2) for param in model.parameters()]))


        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        inputs = torch.stack([torch.tensor(dp[0], dtype=torch.float32, device=device) for dp in data])
        labels = torch.tensor([dp[1] for dp in data], dtype=torch.long, device=device)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # for portable code
        i = 0
        batch_size = int(os.getenv("BATCH_SIZE", 32))

        batch_inputs = inputs[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        loss_fn = torch.nn.CrossEntropyLoss()

        PeakMemoryProfiler.mark_event("forward_start")
        outputs = model.forward(batch_inputs)
        PeakMemoryProfiler.mark_event("forward_end")


        PeakMemoryProfiler.mark_event("backward_start")
        loss = loss_fn(outputs, batch_labels)
        print("[LOSS]: ", loss) # 40000
        loss.backward()
        PeakMemoryProfiler.mark_event("backward_end")

        PeakMemoryProfiler.mark_event("optimizer_start")
        optimizer.step()
        optimizer.zero_grad()
        PeakMemoryProfiler.mark_event("optimizer_end")

        peak_mem_profiler.step()

        # num_epochs = int(os.getenv("NUM_EPOCHS", 100))
        # batch_size = int(os.getenv("BATCH_SIZE", 32))
        #
        # for epoch in range(num_epochs):
        #     epoch_loss = 0.0
        #     num_batches = 0
        #
        #     for i in range(0, len(input_ids), batch_size):
        #         batch_input_ids = input_ids[i:i + batch_size]
        #         batch_attention_mask = attention_mask[i:i + batch_size]
        #         batch_labels = labels[i:i + batch_size]
        #
        #         outputs = model.forward(
        #             input_ids=batch_input_ids,
        #             attention_mask=batch_attention_mask,
        #             labels=batch_labels,
        #         )
        #         loss = outputs.loss
        #
        #         loss.backward()
        #
        #         optimizer.step()
        #         optimizer.zero_grad()
        #
        #         epoch_loss += loss.item()
        #         num_batches += 1
        #
        #         loss_profiler.record(loss)
        #         peak_mem_profiler.step()
        #         iter_profiler.step()
        #
        #     avg_loss = epoch_loss / num_batches
        #     print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    single_train()
