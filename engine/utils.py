import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import torch.distributed as dist

from engine.communication import ShardedModuleState

def get_tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.shape().prod().item()

def has_direct_params(module: nn.Module) -> bool:
    return next(module.parameters(recurse=False), None) is not None


def rank0_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def get_module_tree(module: nn.Module, include_params: bool = False) -> dict[str, list[str]]:
    adj_list = {}

    def build_tree(mod: nn.Module, parent_name: str = ""):
        name = parent_name or mod.__class__.__name__
        children = []

        for child_name, child_mod in mod.named_children():
            full_name = f"{name}.{child_name}"
            children.append(full_name)
            build_tree(child_mod, full_name)

        if include_params:
            for param_name, _ in mod.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}"
                children.append(full_name)
                adj_list[full_name] = []

        adj_list[name] = children

    build_tree(module)
    return adj_list


def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, xcenter=0.5):
    pos = {}

    def _hierarchy(node, left, right, depth=0):
        pos[node] = ((left + right) / 2, -depth * vert_gap)
        children = list(G.successors(node))
        if children:
            dx = (right - left) / len(children)
            for i, child in enumerate(children):
                _hierarchy(child, left + i * dx, left + (i + 1) * dx, depth + 1)

    _hierarchy(root, 0, width)
    return pos

def graph_module(module: nn.Module, save_path: str = None, include_params: bool = False):
    adj_list = get_module_tree(module, include_params=include_params)
    G = nx.DiGraph(adj_list)
    root = module.__class__.__name__

    labels = {node: node.split('.')[-1] for node in G.nodes()}

    num_nodes = len(G.nodes())
    width = max(4.0, num_nodes * 0.5)
    fig_width = max(16, num_nodes * 1.5)

    pos = hierarchy_pos(G, root, width=width, vert_gap=0.4)

    plt.figure(figsize=(fig_width, 10))
    nx.draw(G, pos, labels=labels, node_color='lightblue',
            node_size=2000, font_size=8, font_weight='bold',
            arrows=True, arrowsize=15, edge_color='gray')
    plt.title('Module Tree')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

def get_shard_numels(model: nn.Module):
    model_numels = {}
    for name, param in model.named_parameters():
        sharded_param = getattr(param, "_shard_state", None)
        shard_numel = sharded_param.materialized.numel() if sharded_param is not None and sharded_param.materialized is not None else 0
        full_numel = param.numel()
        model_numels[name] = f"{shard_numel} / {full_numel}"
    return model_numels

def overlap(interval_a: tuple[int], interval_b: tuple[int]) -> tuple[int, int] | None:
    start_a, end_a = interval_a
    start_b, end_b = interval_b

    start = max(start_a, start_b)
    end = min(end_a, end_b)

    if start >= end:
        return None  # No overlap
    return (start, end)