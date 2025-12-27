from abc import ABC, abstractmethod
import glob
import os
import time
import torch.distributed as dist
from typing import Optional, TypeVar, Type, ClassVar, Self

class ZeroProfiler(ABC):
    """Abstract base class for all Zero profilers.

    Provides rank-aware file logging and singleton access per profiler type.
    When log_folder is specified, logs go to {log_folder}/{log_name}_rank{rank}.log.
    """

    _current: ClassVar[Self | None] = None 

    def __init__(
        self,
        log_folder: Optional[str] = None,
        log_name: str = "profiler",
        log_ranks: Optional[list[int]] = None,
        clear_logs: bool = False,
    ):
        self.log_folder = log_folder
        self.log_name = log_name
        self.log_ranks = log_ranks
        self._rank = self._get_rank()
        self._start_time = time.perf_counter()

        if clear_logs and log_folder and self._rank == 0:
            self._clear_existing_logs()

        type(self)._current = self

    @classmethod
    def current(cls) -> Self | None:
        return cls._current

    def _get_rank(self) -> int:
        if dist.is_initialized():
            return dist.get_rank()
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))

    def _is_distributed(self) -> bool:
        return dist.is_initialized()

    def _rank_suffix(self) -> str:
        return f"_rank{self._rank}" if self._is_distributed() else ""

    def _should_log(self) -> bool:
        if self.log_ranks is None:
            return True
        return self._rank in self.log_ranks

    def _get_log_path(self) -> Optional[str]:
        if not self.log_folder:
            return None
        return os.path.join(self.log_folder, f"{self.log_name}{self._rank_suffix()}.log")

    def _clear_existing_logs(self):
        if self.log_folder is None:
            return
        pattern = os.path.join(self.log_folder, f"{self.log_name}*")
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path):
                os.remove(file_path)

    def _log(self, message: str):
        if not self._should_log():
            return

        log_path = self._get_log_path()
        if log_path is not None:
            os.makedirs(self.log_folder, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(f"[Rank {self._rank}] {message}\n")
        else:
            print(f"[Rank {self._rank}] {message}")

    def mark(self, label: str):
        """Log a timestamped event marker."""
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self._log(f"[MARK] {elapsed_ms:.2f}ms - {label}")

    @abstractmethod
    def __enter__(self) -> 'ZeroProfiler': ...

    @abstractmethod
    def __exit__(self, *args, **kwargs): ...
