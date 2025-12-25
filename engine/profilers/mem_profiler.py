import torch
from torch.profiler import profile, schedule, ProfilerActivity
from typing import Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import subprocess
import os

from engine.profilers.base import ZeroProfiler

try:
    import memray
    MEMRAY_AVAILABLE = True
except ImportError:
    MEMRAY_AVAILABLE = False


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
    Exports an interactive HTML memory timeline via export_memory_timeline().
    """

    def __init__(
        self,
        output_folder: Optional[str] = None,
        profile_name: str = "peak_memory",
        device: str = "cpu",
        log_ranks: Optional[list[int]] = None,
    ):
        super().__init__(output_folder, profile_name, log_ranks)
        self.output_folder = output_folder
        self.profile_name = profile_name
        self.device = device
        self.use_cuda = device != "cpu" and torch.cuda.is_available()
        self.profiler = None

    def __enter__(self):
        self._register_instance()
        activities = [ProfilerActivity.CUDA] if self.use_cuda else [ProfilerActivity.CPU]

        self.profiler = profile(
            activities=activities,
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )
        self.profiler.__enter__()
        return self

    def step(self, _label: Optional[str] = None):
        if self.profiler:
            self.profiler.step()

    def __exit__(self, *args, **kwargs):
        self.profiler.__exit__(*args, **kwargs)
        self._export_timeline()
        self._unregister_instance()

    def _export_timeline(self):
        if not self._should_log() or not self.output_folder:
            return
        os.makedirs(self.output_folder, exist_ok=True)
        path = os.path.join(self.output_folder, f"{self.profile_name}{self._rank_suffix()}.html")
        self.profiler.export_memory_timeline(path)
        self._log(f"[PeakMemoryProfiler] Memory timeline saved to {path}")


class MemoryProfiler(ZeroProfiler):
    def __init__(
        self,
        graph_folder: Optional[str] = None,
        record_shapes: bool = True,
        row_limit: int = 10,
        profile_name: str = "memory_profile",
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 3,
        log_ranks: Optional[list[int]] = None,
    ):
        super().__init__(graph_folder, profile_name, log_ranks)
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
        self._register_instance()
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
        self._unregister_instance()

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
        self._log(
            self.profiler.key_averages().table(
                sort_by="self_cpu_memory_usage",
                row_limit=self.row_limit
            )
        )

    def _graph(self):
        if not self._should_log():
            return
        if not self.result or not self.result.snapshots:
            self._log("No memory snapshots to graph")
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
            self._log(f"Memory graph saved to {path}")
        else:
            plt.show()

        plt.close()


class FlamegraphMemoryProfiler(ZeroProfiler):
    """Memory profiler that outputs flamegraphs.

    CPU mode: Uses memray to generate interactive HTML flamegraphs.
    CUDA mode: Uses PyTorch memory snapshots (.pickle) viewable at pytorch.org/memory_viz.

    Generates per-process output files for distributed training.
    Each rank gets its own output file with the rank number in the filename.
    """

    def __init__(
        self,
        output_folder: str = "./profiles",
        profile_name: str = "memory_flamegraph",
        device: str = "cpu",
        interactive: bool = True,
        track_leaks: bool = False,
        generate_temporal: bool = False,
        log_ranks: Optional[list[int]] = None,
    ):
        super().__init__(output_folder, profile_name, log_ranks)
        self.output_folder = output_folder
        self.profile_name = profile_name
        self.device = device
        self.interactive = interactive
        self.track_leaks = track_leaks
        self.generate_temporal = generate_temporal
        self.use_cuda = device != "cpu" and torch.cuda.is_available()

        if not self.use_cuda and not MEMRAY_AVAILABLE:
            raise ImportError(
                "memray is required for CPU FlamegraphMemoryProfiler. "
                "Install with: pip install memray"
            )

        self.tracker = None
        self.bin_path = None
        self.pickle_path = None

    def __enter__(self):
        self._register_instance()
        os.makedirs(self.output_folder, exist_ok=True)

        if self.use_cuda:
            self.pickle_path = os.path.join(
                self.output_folder,
                f"{self.profile_name}{self._rank_suffix()}.pickle"
            )
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="all",
            )
            self._log(f"[FlamegraphMemoryProfiler] Started CUDA tracking")
        else:
            self.bin_path = os.path.join(
                self.output_folder,
                f"{self.profile_name}{self._rank_suffix()}.bin"
            )
            destination = memray.FileDestination(self.bin_path, overwrite=True)
            self.tracker = memray.Tracker(destination=destination, native_traces=True)
            self.tracker.__enter__()
            self._log(f"[FlamegraphMemoryProfiler] Started CPU tracking")

        return self

    def __exit__(self, *args, **kwargs):
        if self.use_cuda:
            torch.cuda.memory._dump_snapshot(self.pickle_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            self._log(f"[FlamegraphMemoryProfiler] CUDA snapshot saved to {self.pickle_path}")
            self._log("[FlamegraphMemoryProfiler] View at https://pytorch.org/memory_viz")
        else:
            self.tracker.__exit__(*args, **kwargs)
            self._log(f"[FlamegraphMemoryProfiler] Stopped CPU tracking")
            if self.interactive:
                self._generate_flamegraph()
        self._unregister_instance()

    def _generate_flamegraph(self):
        html_path = os.path.join(
            self.output_folder,
            f"{self.profile_name}{self._rank_suffix()}.html"
        )

        cmd = ["memray", "flamegraph", "-o", html_path, "-f", "--inverted"]

        if self.track_leaks:
            cmd.append("--leaks")
        if self.generate_temporal:
            cmd.append("--temporal")

        cmd.append(self.bin_path)

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self._log(f"[FlamegraphMemoryProfiler] Flamegraph saved to {html_path}")
        except subprocess.CalledProcessError as e:
            self._log(f"[FlamegraphMemoryProfiler] Failed to generate flamegraph: {e.stderr.decode()}")
