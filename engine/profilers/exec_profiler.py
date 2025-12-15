import time
from typing import Optional
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from engine.utils.distributed import rank_print


@dataclass
class IterationSnapshot:
    iteration: int
    time_ms: float


class IterationProfiler:
    def __init__(
        self,
        graph_folder: Optional[str] = None,
        profile_name: str = "iteration_time",
    ):
        self.graph_folder = graph_folder
        self.profile_name = profile_name
        self.snapshots: list[IterationSnapshot] = []
        self.iteration = 0
        self.last_step_time: Optional[float] = None

    def __enter__(self):
        self.last_step_time = time.perf_counter()
        return self

    def step(self):
        now = time.perf_counter()
        if self.last_step_time is not None:
            elapsed_ms = (now - self.last_step_time) * 1000
            self.snapshots.append(IterationSnapshot(
                iteration=self.iteration,
                time_ms=elapsed_ms
            ))
            self.iteration += 1
        self.last_step_time = now

    def __exit__(self, *args, **kwargs):
        self._print_summary()
        self._graph()

    def _print_summary(self):
        if not self.snapshots:
            rank_print("No iteration times recorded")
            return

        times = [s.time_ms for s in self.snapshots]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        rank_print(f"\n{'='*60}")
        rank_print(f"Iteration Time Summary")
        rank_print(f"{'='*60}")
        rank_print(f"Total iterations: {len(times)}")
        rank_print(f"Avg time: {avg_time:.2f}ms")
        rank_print(f"Min time: {min_time:.2f}ms")
        rank_print(f"Max time: {max_time:.2f}ms")
        rank_print(f"{'='*60}")

    def _graph(self):
        if not self.snapshots:
            return

        if self.graph_folder:
            os.makedirs(self.graph_folder, exist_ok=True)

        plt.figure(figsize=(12, 5))

        iterations = [s.iteration for s in self.snapshots]
        times = [s.time_ms for s in self.snapshots]

        plt.plot(iterations, times, linewidth=1, alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Time (ms)')
        plt.title(f'{self.profile_name}: Wall Clock Time per Iteration')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.graph_folder:
            path = os.path.join(self.graph_folder, f"{self.profile_name}.png")
            plt.savefig(path, dpi=150)
            rank_print(f"Iteration time graph saved to {path}")

        plt.close()
