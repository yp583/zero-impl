import weakref
import os
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
import matplotlib.pyplot as plt

from engine.profilers.base import ZeroProfiler


@dataclass
class TensorMemorySnapshot:
    step: int
    cumulative_bytes: int
    label: Optional[str] = None


class TensorLifecycleProfiler(ZeroProfiler):
    """Tracks tensor allocations and deallocations via nn.Parameter overrides.

    Uses weak references to detect when tensors are garbage collected.
    Maintains cumulative byte tracking and outputs a graph showing memory over time.
    """

    CLEANUP_INTERVAL = 100

    def __init__(
        self,
        graph_folder: Optional[str] = None,
        profile_name: str = "tensor_lifecycle",
        log_ranks: Optional[list[int]] = None,
    ):
        super().__init__(graph_folder, profile_name, log_ranks)
        self.graph_folder = graph_folder
        self.profile_name = profile_name
        self._weak_refs: list = []
        self._original_init = None
        self._cumulative_bytes = 0
        self._step_count = 0
        self._steps_since_cleanup = 0
        self.snapshots: list[TensorMemorySnapshot] = []

    def __enter__(self):
        self._register_instance()
        self._weak_refs = []
        self._cumulative_bytes = 0
        self._step_count = 0
        self.snapshots = []
        profiler = self

        self._original_init = nn.Parameter.__init__
        orig_init = self._original_init

        def new_init(t_self, *args, **kwargs):
            orig_init(t_self)
            storage_size = t_self.storage().size()
            is_meta = t_self.is_meta

            if not is_meta and storage_size > 0:
                profiler._cumulative_bytes += storage_size

                def on_freed(ref, size=storage_size):
                    profiler._cumulative_bytes -= size

                ref = weakref.ref(t_self, on_freed)
                profiler._weak_refs.append(ref)

        nn.Parameter.__init__ = new_init
        return self

    def _cleanup_dead_refs(self):
        self._weak_refs = [ref for ref in self._weak_refs if ref() is not None]

    def step(self, label: Optional[str] = None):
        self._steps_since_cleanup += 1
        if self._steps_since_cleanup >= self.CLEANUP_INTERVAL:
            self._cleanup_dead_refs()
            self._steps_since_cleanup = 0

        self.snapshots.append(TensorMemorySnapshot(
            step=self._step_count,
            cumulative_bytes=self._cumulative_bytes,
            label=label,
        ))
        self._step_count += 1

    def _graph(self):
        if not self.snapshots or not self._should_log():
            return

        if self.graph_folder:
            os.makedirs(self.graph_folder, exist_ok=True)

        plt.figure(figsize=(12, 5))

        steps = [s.step for s in self.snapshots]
        cumulative_mb = [s.cumulative_bytes / (1024 * 1024) for s in self.snapshots]

        plt.plot(steps, cumulative_mb, label='Cumulative Memory (MB)', color='steelblue', linewidth=1.5)
        plt.fill_between(steps, cumulative_mb, alpha=0.3, color='steelblue')

        colors = ['red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_idx = 0
        for snapshot in self.snapshots:
            if snapshot.label:
                plt.axvline(x=snapshot.step, color=colors[color_idx % len(colors)], linestyle=':', linewidth=1.5, alpha=0.8)
                plt.text(snapshot.step, plt.ylim()[1] * 0.95, snapshot.label, rotation=90,
                        verticalalignment='top', fontsize=8, color=colors[color_idx % len(colors)])
                color_idx += 1

        plt.xlabel('Step')
        plt.ylabel('Cumulative Memory (MB)')
        plt.title(f'{self.profile_name}: Tensor Memory Over Time')
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.graph_folder:
            path = os.path.join(self.graph_folder, f"{self.profile_name}{self._rank_suffix()}.png")
            plt.savefig(path, dpi=150)
            self._log(f"Tensor lifecycle graph saved to {path}")
        else:
            plt.show()

        plt.close()

    def _print_summary(self):
        if not self.snapshots:
            return
        max_mem = max(s.cumulative_bytes for s in self.snapshots) / (1024 * 1024)
        final_mem = self.snapshots[-1].cumulative_bytes / (1024 * 1024)
        self._log(f"[TensorLifecycleProfiler] Peak: {max_mem:.2f} MB, Final: {final_mem:.2f} MB")

    def __exit__(self, *args, **kwargs):
        if self._original_init:
            nn.Parameter.__init__ = self._original_init
        self._weak_refs.clear()
        self._print_summary()
        self._graph()
        self._unregister_instance()
