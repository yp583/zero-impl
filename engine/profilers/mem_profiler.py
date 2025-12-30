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
from engine.utils.distributed import (
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    DEFAULT_STACK_ORDER,
)
# Category colors matching PyTorch's _memory_profiler.py

def generate_distinct_colors(n: int) -> list[str]:
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
        category_stack_order: Optional[dict[str, int]] = None,
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
        self._category_stack_order = category_stack_order or DEFAULT_STACK_ORDER

    def __enter__(self):
        activities = [ProfilerActivity.CUDA] if self.use_cuda else [ProfilerActivity.CPU]

        self.profiler = profile(
            activities=activities,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
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

        events = list(self.profiler.events())
        self._log(f"Profiler captured {len(events)} events")

        category_scopes = self._extract_category_scopes(events)

        pytorch_timeline = self._get_pytorch_timeline(base_path)
        self._log(f"PyTorch timeline: {'available' if pytorch_timeline else 'not available'}")

        if pytorch_timeline is not None:
            times_ms, sizes_by_category = self._build_hybrid_timeline(
                pytorch_timeline, events, category_scopes
            )
        else:
            times_ms, sizes_by_category = self._build_categorized_timeline(
                events, category_scopes
            )

        self._log(f"Timeline has {len(times_ms)} data points")

        if len(times_ms) == 0:
            self._log("No memory data to plot")
            return

        sizes = np.array(sizes_by_category)
        num_categories = min(sizes.shape[1], len(CATEGORY_COLORS))

        # Build stacking order: list of original indices sorted by stack position
        stack_order = sorted(
            range(num_categories),
            key=lambda i: self._category_stack_order.get(CATEGORY_NAMES[i], i)
        )

        # Reorder columns by stack position
        reordered_sizes = sizes[:, stack_order]
        stacked = np.cumsum(reordered_sizes, axis=1) / (1024**3)

        fig, ax = plt.subplots(figsize=(14, 8))

        for stack_pos in range(num_categories):
            orig_idx = stack_order[stack_pos]
            lower = stacked[:, stack_pos - 1] if stack_pos > 0 else np.zeros(len(times_ms))
            upper = stacked[:, stack_pos]
            ax.fill_between(times_ms, lower, upper, color=CATEGORY_COLORS[orig_idx], alpha=0.7)

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

        # Legend in stack order (bottom to top)
        category_handles = [
            Patch(color=CATEGORY_COLORS[stack_order[i]], alpha=0.7, label=CATEGORY_NAMES[stack_order[i]])
            for i in range(num_categories)
        ]

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

    def _extract_category_scopes(self, events) -> list[tuple[int, int, int]]:
        """Extract [memory:Category] scopes from profiler events.

        Returns list of (start_us, end_us, category_index) tuples.
        """
        scopes = []
        for event in events:
            if event.name.startswith("[memory:"):
                category_name = event.name[8:-1]
                if category_name in CATEGORY_NAMES:
                    category_idx = CATEGORY_NAMES.index(category_name)
                    start_us = event.time_range.start
                    end_us = event.time_range.end
                    scopes.append((start_us, end_us, category_idx))
        return scopes

    def _get_category_at_time(self, time_us: int, scopes: list[tuple[int, int, int]]) -> int:
        """Find which category scope is active at a given time. Returns 0 (Unknown) if none."""
        for start_us, end_us, category_idx in scopes:
            if start_us <= time_us <= end_us:
                return category_idx
        return 0

    def _build_categorized_timeline(
        self, events, category_scopes: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Build memory timeline with custom categorization based on record_tensors_as scopes."""
        mem_key = "cuda_memory_usage" if self.use_cuda else "cpu_memory_usage"

        mem_events = []
        for event in events:
            mem_usage = getattr(event, mem_key, 0)
            if mem_usage != 0:
                time_us = event.time_range.start
                mem_events.append((time_us, mem_usage))

        if not mem_events:
            return np.array([]), []

        mem_events.sort(key=lambda x: x[0])
        t0 = mem_events[0][0]

        num_categories = len(CATEGORY_NAMES)
        current_by_category = [0] * num_categories

        times_ms = []
        sizes_by_category = []

        for time_us, mem_delta in mem_events:
            category_idx = self._get_category_at_time(time_us, category_scopes)
            current_by_category[category_idx] += mem_delta

            times_ms.append((time_us - t0) / 1000)
            sizes_by_category.append(list(current_by_category))

        return np.array(times_ms), sizes_by_category

    def _get_pytorch_timeline(self, base_path: str) -> dict | None:
        """Get PyTorch's automatic memory categorization."""
        tmp_json = f"/tmp/{self.profile_name}{self._rank_suffix()}_pytorch.json"
        device = "cuda:0" if self.use_cuda else "cpu"

        try:
            self.profiler.export_memory_timeline(tmp_json, device=device)
            with open(tmp_json) as f:
                data = json.load(f)
            os.remove(tmp_json)

            if not data[0] or not data[1]:
                return None

            return {"times_us": data[0], "sizes_by_category": data[1]}
        except Exception as e:
            self._log(f"Could not get PyTorch timeline: {e}")
            return None

    def _build_hybrid_timeline(
        self,
        pytorch_data: dict,
        events,
        category_scopes: list[tuple[int, int, int]],
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Build timeline using PyTorch's auto-detection, overriding with our scopes.

        - Outside any record_tensors_as scope: use PyTorch's automatic categorization
        - Inside a scope: attribute memory deltas to the specified category
        """
        times_us = pytorch_data["times_us"]
        pytorch_sizes = pytorch_data["sizes_by_category"]

        if not times_us:
            return np.array([]), []

        # Align timestamps between memory timeline and profiler events
        aligned_scopes = self._align_scope_timestamps(times_us, events, category_scopes)

        t0 = times_us[0]
        num_categories = len(CATEGORY_NAMES)
        pytorch_num_cats = len(pytorch_sizes[0]) if pytorch_sizes else 0

        current_by_category = [0] * num_categories
        prev_pytorch_row = [0] * pytorch_num_cats

        times_ms = []
        final_sizes = []

        for i, time_us in enumerate(times_us):
            row = pytorch_sizes[i]
            active_scope_category = self._get_category_at_time(time_us, aligned_scopes)

            # Calculate delta from previous timestamp
            deltas = [row[j] - prev_pytorch_row[j] for j in range(len(row))]
            prev_pytorch_row = list(row)

            if active_scope_category != 0:
                # Inside custom scope: all new allocations go to our category
                total_delta = sum(deltas)
                current_by_category[active_scope_category] += total_delta
            else:
                # Outside scope: apply PyTorch's categorization for each delta
                for j, delta in enumerate(deltas):
                    if j < num_categories:
                        current_by_category[j] += delta

            times_ms.append((time_us - t0) / 1000)
            final_sizes.append(list(current_by_category))

        return np.array(times_ms), final_sizes

    def _align_scope_timestamps(
        self,
        memory_times_us: list[int],
        events,
        category_scopes: list[tuple[int, int, int]],
    ) -> list[tuple[int, int, int]]:
        """Align profiler event timestamps to memory timeline timestamps.

        PyTorch's memory timeline and profiler events may use different time bases.
        This finds the offset by matching the earliest event timestamp to the
        earliest memory timestamp.
        """
        if not category_scopes or not memory_times_us:
            return category_scopes

        # Find earliest profiler event timestamp
        event_times = [e.time_range.start for e in events if hasattr(e, 'time_range')]
        if not event_times:
            return category_scopes

        min_event_time = min(event_times)
        min_memory_time = memory_times_us[0]

        # Calculate offset: memory_time = event_time + offset
        offset = min_memory_time - min_event_time

        # Apply offset to all scopes
        return [
            (start + offset, end + offset, cat_idx)
            for start, end, cat_idx in category_scopes
        ]

    @staticmethod
    def mark_event(label: str):
        profiler = PeakMemoryProfiler.current()
        if profiler:
            profiler.mark(label)

