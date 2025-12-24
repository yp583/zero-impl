import time
from typing import Optional
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from engine.profilers.base import ZeroProfiler


@dataclass
class IterationSnapshot:
    iteration: int
    time_ms: float


class IterationProfiler(ZeroProfiler):
    def __init__(
        self,
        graph_folder: Optional[str] = None,
        profile_name: str = "iteration_time",
        log_ranks: Optional[list[int]] = None,
    ):
        super().__init__(graph_folder, profile_name, log_ranks)
        self.graph_folder = graph_folder
        self.profile_name = profile_name
        self.snapshots: list[IterationSnapshot] = []
        self.iteration = 0
        self.last_step_time: Optional[float] = None

    def __enter__(self):
        self._register_instance()
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
        self._unregister_instance()

    def _print_summary(self):
        if not self.snapshots:
            self._log("No iteration times recorded")
            return

        times = [s.time_ms for s in self.snapshots]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        self._log(f"\n{'='*60}")
        self._log(f"Iteration Time Summary")
        self._log(f"{'='*60}")
        self._log(f"Total iterations: {len(times)}")
        self._log(f"Avg time: {avg_time:.2f}ms")
        self._log(f"Min time: {min_time:.2f}ms")
        self._log(f"Max time: {max_time:.2f}ms")
        self._log(f"{'='*60}")

    def _graph(self):
        if not self.snapshots or not self._should_log():
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
            self._log(f"Iteration time graph saved to {path}")

        plt.close()
