import torch
import torch.nn as nn
from typing import List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os


@dataclass
class ModelSnapshot:
    trigger_module: str
    materialized_numel: int
    total_numel: int


class MetaParamCounter:
    def __init__(self, graph_folder=None):
        self.snapshots: Dict[str, List[ModelSnapshot]] = {}
        self.graph_folder = graph_folder
        self.models: Dict[str, nn.Module] = {}
        self.hooks = []

    def register_model(self, name: str, model: nn.Module):
        self.models[name] = model
        self.snapshots[name] = []

    def graph(self):
        if self.graph_folder:
            os.makedirs(self.graph_folder, exist_ok=True)

        for name, snapshots in self.snapshots.items():
            if not snapshots:
                print(f"No snapshots for {name}")
                continue

            plt.figure(figsize=(12, 6))

            labels = [s.trigger_module for s in snapshots]
            materialized = [s.materialized_numel for s in snapshots]
            total = [s.total_numel for s in snapshots]

            x_positions = range(len(snapshots))
            plt.plot(x_positions, materialized, marker='o', label='Materialized Parameters', linewidth=2)
            plt.plot(x_positions, total, marker='s', label='Total Parameters', linewidth=2)
            plt.xticks(x_positions, labels, rotation=45, ha='right')
            plt.xlabel('Module')
            plt.ylabel('Parameter Count')
            plt.title(f'{name}: Materialized vs Total Parameter Counts')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if self.graph_folder:
                path = os.path.join(self.graph_folder, f"{name}.png")
                plt.savefig(path)
                print(f"Graph saved to {path}")
            else:
                plt.show()

            plt.close()

    def __enter__(self):
        def forward_hook(module, *args):
            for name, model in self.models.items():
                all_modules = set(model.modules())
                if module not in all_modules:
                    continue

                materialized = 0
                total = 0
                for p in model.parameters(recurse=True):
                    total += p.numel()
                    if p.device.type != 'meta':
                        materialized += p.numel()

                self.snapshots[name].append(ModelSnapshot(
                    trigger_module=module.__class__.__name__,
                    materialized_numel=materialized,
                    total_numel=total
                ))

        hook = nn.modules.module.register_module_forward_hook(forward_hook)
        self.hooks.append(hook)
        return self

    def __exit__(self, *args, **kwargs):
        for hook in self.hooks:
            hook.remove()

        self.graph()
