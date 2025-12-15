

class MemoryProfiler:
    def __init__(
        self,
        graph_folder: Optional[str] = None,
        record_shapes: bool = True,
        row_limit: int = 10,
        profile_name: str = "memory_profile",
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 3,
    ):
        self.graph_folder = graph_folder
        self.record_shapes = record_shapes
        self.row_limit = row_limit
        self.profile_name = profile_name
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.profiler = None
        self.result: Optional[MemoryProfilerResult] = None

    def __enter__(self):
        self.profiler = profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=self.record_shapes,
            schedule=schedule(
                wait=self.wait_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=1
            ),
            on_trace_ready=self._on_trace_ready
        )
        self.profiler.__enter__()
        return self

    def step(self):
        if self.profiler:
            self.profiler.step()

    def _on_trace_ready(self, _prof):
        self._process_results()
        self._print_table()
        self._graph()

    def __exit__(self, *args, **kwargs):
        self.profiler.__exit__(*args, **kwargs)

    def _process_results(self):
        key_averages = self.profiler.key_averages()
        sorted_events = sorted(
            key_averages,
            key=lambda e: e.self_cpu_memory_usage,
            reverse=True
        )[:self.row_limit]

        self.result = MemoryProfilerResult(
            snapshots=[
                MemorySnapshot(
                    name=event.key,
                    cpu_memory=event.cpu_memory_usage,
                    self_cpu_memory=event.self_cpu_memory_usage,
                    call_count=event.count
                )
                for event in sorted_events
            ],
            total_cpu_time=str(key_averages.self_cpu_time_total)
        )

    def _format_memory(self, bytes_val: int) -> str:
        if abs(bytes_val) >= 1024 * 1024:
            return f"{bytes_val / (1024 * 1024):.2f} Mb"
        elif abs(bytes_val) >= 1024:
            return f"{bytes_val / 1024:.2f} Kb"
        return f"{bytes_val} b"

    def _print_table(self):
        rank_print(
            self.profiler.key_averages().table(
                sort_by="self_cpu_memory_usage",
                row_limit=self.row_limit
            )
        )

    def _graph(self):
        if not self.result or not self.result.snapshots:
            rank_print("No memory snapshots to graph")
            return

        if self.graph_folder:
            os.makedirs(self.graph_folder, exist_ok=True)

        plt.figure(figsize=(14, 6))

        names = [s.name for s in self.result.snapshots]
        self_memory = [s.self_cpu_memory for s in self.result.snapshots]
        total_memory = [s.cpu_memory for s in self.result.snapshots]

        x_positions = range(len(names))
        bar_width = 0.35

        plt.bar(
            [x - bar_width/2 for x in x_positions],
            [m / (1024 * 1024) for m in self_memory],
            bar_width,
            label='Self CPU Memory (Mb)',
            color='steelblue'
        )
        plt.bar(
            [x + bar_width/2 for x in x_positions],
            [m / (1024 * 1024) for m in total_memory],
            bar_width,
            label='Total CPU Memory (Mb)',
            color='darkorange'
        )

        plt.xticks(x_positions, names, rotation=45, ha='right', fontsize=8)
        plt.xlabel('Operator')
        plt.ylabel('Memory (Mb)')
        plt.title(f'{self.profile_name}: Memory Usage by Operator')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if self.graph_folder:
            path = os.path.join(self.graph_folder, f"{self.profile_name}.png")
            plt.savefig(path, dpi=150)
            rank_print(f"Memory graph saved to {path}")
        else:
            plt.show()

        plt.close()import torch
from torch.profiler import profile, schedule, ProfilerActivity
from typing import Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import os

from engine.utils.distributed import rank_print


@dataclass
class MemorySnapshot:
    name: str
    cpu_memory: int
    self_cpu_memory: int
    call_count: int


@dataclass
class MemoryProfilerResult:
    snapshots: list = field(default_factory=list)
    total_cpu_time: str = ""
