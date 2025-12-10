import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from engine.utils.module_helpers import get_module_tree

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
