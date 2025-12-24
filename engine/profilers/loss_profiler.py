import torch
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Optional

from engine.profilers.base import ZeroProfiler


@dataclass
class LossSnapshot:
    step: int
    loss: float


class LossProfiler(ZeroProfiler):
    def __init__(
        self,
        graph_path: Optional[str] = None,
        log_ranks: Optional[list[int]] = None,
    ):
        log_folder = os.path.dirname(graph_path) if graph_path else None
        super().__init__(log_folder, "loss", log_ranks)
        self.graph_path = graph_path
        self.snapshots: list[LossSnapshot] = []
        self.step = 0

    def record(self, loss: torch.Tensor | float):
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        self.snapshots.append(LossSnapshot(step=self.step, loss=loss_val))
        self.step += 1

    def graph(self):
        if not self._should_log():
            return
        if not self.snapshots:
            self._log("No loss snapshots recorded")
            return

        steps = [s.step for s in self.snapshots]
        losses = [s.loss for s in self.snapshots]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, marker='o', linewidth=2, markersize=4)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.graph_path:
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            plt.savefig(self.graph_path)
            self._log(f"Loss graph saved to {self.graph_path}")
        else:
            plt.show()

        plt.close()

    def __enter__(self):
        self._register_instance()
        return self

    def __exit__(self, *args, **kwargs):
        self.graph()
        self._unregister_instance()
