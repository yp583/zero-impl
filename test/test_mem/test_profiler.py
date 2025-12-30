"""Standalone test to verify memory profiler tracks processes separately."""
import os
import torch
import torch.distributed as dist
from engine.profilers import PeakMemoryProfiler
from engine.utils.distributed import DISTRIBUTED_STACK_ORDER, record_tensors_as


def main():
    dist.init_process_group(backend=os.getenv("TORCH_BACKEND", "gloo"))
    rank = dist.get_rank()
    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

    # Sync before profiling
    dist.barrier()

    with PeakMemoryProfiler(
        output_folder=graph_dir,
        profile_name="profiler_test",
        device="cpu",
        export_memory_timeline=True,
        clear_logs=True,
        category_stack_order=DISTRIBUTED_STACK_ORDER,
    ) as pf:

        with record_tensors_as("Temporary"):
            # Rank 0 gets large tensor, others get small
            size = 1000 if rank == 0 else 100
            x = torch.randn(size, size)
            PeakMemoryProfiler.mark_event("after_randn")

            y = x @ x.T
            PeakMemoryProfiler.mark_event("after_matmul")

        pf.step()
        del x, y

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
