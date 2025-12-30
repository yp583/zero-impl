from data_sources.dev.dev_dataclient import DevDatasetClient
from engine.utils.distributed import rank_print, DISTRIBUTED_STACK_ORDER
import torch
import torch.distributed as dist
from contextlib import ExitStack

from transformers.modeling_outputs import SequenceClassifierOutput
from test.test_mem.model import TestModel
from engine.zero_init import ZeroEngine, ZeroEngineConfig
from engine.profilers import PeakMemoryProfiler
from engine.utils import rank0_print
from dotenv import load_dotenv
import os
import logging

load_dotenv()


def finalize_dist():
    dist.barrier()
    dist.destroy_process_group()


def dist_train():
    dist.init_process_group(backend=os.getenv("TORCH_BACKEND"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    rank_print(f"LOSS process initialized successfully!")

    ds_client = DevDatasetClient(rank=rank, world_size=world_size)
    data = ds_client.get_shard()

    device = "cpu"
    torch.manual_seed(42 + rank)

    zero_config = ZeroEngineConfig(
        device=device,
        debug=True,
    )

    rank_print(torch.accelerator.current_accelerator())
    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

    with ExitStack() as stack:
        peak_mem_profiler = stack.enter_context(PeakMemoryProfiler(
            output_folder=graph_dir,
            profile_name="peak_memory_dist",
            device=device,
            export_memory_timeline=True,
            clear_logs=True,
            category_stack_order=DISTRIBUTED_STACK_ORDER,
        ))
        ze = stack.enter_context(ZeroEngine(config=zero_config))


        model = TestModel(input_dim=128, hidden_dim=64, output_dim=128)
        ze.register_model(model)
        norm = 0
        for name, param in model.named_parameters():
            if param._shard_state.materialized is not None:
                norm += torch.norm(param._shard_state.materialized)
        print("[NORM OF INITED PARAMS]: ", norm)


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
        rank0_print("[LOSS]: ", loss)
        loss.backward()
        PeakMemoryProfiler.mark_event("backward_end")

        PeakMemoryProfiler.mark_event("optimizer_start")
        optimizer.step()
        optimizer.zero_grad()
        PeakMemoryProfiler.mark_event("optimizer_end")

        peak_mem_profiler.step()

    finalize_dist()

if __name__ == "__main__":
    dist_train()
