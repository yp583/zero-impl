import json
import re
import time
import torch
import numpy as np
from torch.profiler import profile, schedule, ProfilerActivity
from typing import Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import os
from matplotlib.patches import Patch

from engine.profilers.base import ZeroProfiler

# Category colors matching PyTorch's _memory_profiler.py
CATEGORY_COLORS = [
    "grey",           # Unknown/None (index 0)
    "darkgreen",      # PARAMETER
    "goldenrod",      # OPTIMIZER_STATE
    "black",          # INPUT
    "mediumpurple",   # TEMPORARY
    "red",            # ACTIVATION
    "mediumblue",     # GRADIENT
    "royalblue",      # AUTOGRAD_DETAIL
]

CATEGORY_NAMES = [
    "Unknown", "Parameter", "Optimizer", "Input",
    "Temporary", "Activation", "Gradient", "Autograd"
]

def generate_distinct_colors(n: int) -> list[str]:
    """Generate n maximally distinct colors using golden ratio hue spacing."""
    if n == 0:
        return []
    colors = []
    golden_ratio = 0.618033988749895
    hue = 0.0
    for _ in range(n):
        rgb = mcolors.hsv_to_rgb([hue, 0.85, 0.75])
        colors.append(mcolors.to_hex(rgb))
        hue = (hue + golden_ratio) % 1.0
    return colors


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


class PeakMemoryProfiler(ZeroProfiler):
    """Thin wrapper around PyTorch profiler for memory tracking.

    Uses torch.profiler with profile_memory=True to track tensor allocations.
    Can export Chrome trace (.json) and/or memory timeline (.html) files.
    """

    def __init__(
        self,
        output_folder: Optional[str] = None,
        profile_name: str = "peak_memory",
        device: str = "cpu",
        log_ranks: Optional[list[int]] = None,
        export_chrome_trace: bool = False,
        export_memory_timeline: bool = False,
        keep_raw_timeline: bool = False,
        clear_logs: bool = False,
    ):
        super().__init__(output_folder, profile_name, log_ranks, clear_logs)
        self.output_folder = output_folder
        self.profile_name = profile_name
        self.device = device
        self.use_cuda = device != "cpu" and torch.cuda.is_available()
        self.profiler = None
        self._export_chrome_trace = export_chrome_trace
        self._export_memory_timeline = export_memory_timeline
        self._keep_raw_timeline = keep_raw_timeline
        self._marks: list[tuple[float, str]] = []
        self._profile_start_ns: int = 0

    def __enter__(self):
        activities = [ProfilerActivity.CUDA] if self.use_cuda else [ProfilerActivity.CPU]

        self.profiler = profile(
            activities=activities,
            profile_memory=True,
            record_shapes=False,
            with_stack=False,
        )
        self.profiler.__enter__()
        self._profile_start_ns = time.time_ns()
        return self

    def step(self, _label: Optional[str] = None):
        if self.profiler:
            self.profiler.step()

    def __exit__(self, *args, **kwargs):
        assert self.profiler is not None
        self.profiler.__exit__(*args, **kwargs)
        self._export_timeline()

    def mark(self, label: str):
        super().mark(label)
        elapsed_ms = (time.time_ns() - self._profile_start_ns) / 1_000_000
        self._marks.append((elapsed_ms, label))

    def _export_timeline(self):
        assert self.profiler is not None
        if not self._should_log() or not self.output_folder:
            return
        os.makedirs(self.output_folder, exist_ok=True)
        base_path = os.path.join(self.output_folder, f"{self.profile_name}{self._rank_suffix()}")
        if self._export_chrome_trace:
            self.profiler.export_chrome_trace(f"{base_path}.json")
        if self._export_memory_timeline:
            self._build_memory_timeline(base_path)

    def _build_memory_timeline(self, base_path: str):
        assert self.profiler is not None
        if self._keep_raw_timeline:
            tmp_json = f"{base_path}_raw.json"
        else:
            tmp_json = f"/tmp/{self.profile_name}{self._rank_suffix()}_raw.json"
        device = f"cuda:0" if self.use_cuda else "cpu"

        try:
            self.profiler.export_memory_timeline(tmp_json, device=device)
        except Exception as e:
            self._log(f"Failed to export memory timeline: {e}")
            return

        with open(tmp_json) as f:
            data = json.load(f)

        times_us = data[0]
        sizes_by_category = data[1]

        if not times_us or not sizes_by_category:
            self._log("No memory data to plot")
            os.remove(tmp_json)
            return

        t0 = times_us[0] if times_us else 0
        times_ms = np.array([(t - t0) / 1000 for t in times_us])
        sizes = np.array(sizes_by_category)
        stacked = np.cumsum(sizes, axis=1) / (1024**3)  # Cumulative sum -> GiB

        fig, ax = plt.subplots(figsize=(14, 8))

        num_categories = min(stacked.shape[1], len(CATEGORY_COLORS))
        for i in range(num_categories):
            lower = stacked[:, i - 1] if i > 0 else np.zeros(len(times_ms))
            upper = stacked[:, i]
            ax.fill_between(times_ms, lower, upper, color=CATEGORY_COLORS[i], alpha=0.7)

        unique_labels = list(dict.fromkeys(label for _, label in self._marks))
        mark_colors = generate_distinct_colors(len(unique_labels))
        label_to_color = {label: mark_colors[i] for i, label in enumerate(unique_labels)}

        for mark_ms, label in self._marks:
            ax.axvline(x=mark_ms, color=label_to_color[label], linestyle='--', linewidth=2)

        marker_handles = [mlines.Line2D([], [], color=label_to_color[label], linestyle='--',
                                        linewidth=2, label=label) for label in unique_labels]

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Memory (GiB)')
        ax.set_title(f'{self.profile_name}: Memory Timeline')
        ax.grid(True, alpha=0.3)

        category_handles = [Patch(color=CATEGORY_COLORS[i], alpha=0.7, label=CATEGORY_NAMES[i])
                          for i in range(num_categories)]

        legend1 = ax.legend(handles=category_handles, loc='upper left', fontsize=8, title='Memory Categories')
        ax.add_artist(legend1)

        if marker_handles:
            ncol = min(4, len(marker_handles))
            ax.legend(handles=marker_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                     ncol=ncol, fontsize=7, title='Event Markers')

        plt.tight_layout()
        plt.savefig(f"{base_path}.png", dpi=150, bbox_inches='tight')
        self._log(f"Memory timeline saved to {base_path}.png")
        plt.close(fig)

        if not self._keep_raw_timeline and os.path.exists(tmp_json):
            os.remove(tmp_json)

    @staticmethod
    def mark_event(label: str):
        profiler = PeakMemoryProfiler.current()
        if profiler:
            profiler.mark(label)

