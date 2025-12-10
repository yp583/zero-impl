import torch
import torch.nn as nn

import os
from engine.utils import graph_module

class TestModel(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=4, num_heads=4):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.layer1(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        attn_output, _ = self.attn(x, x, x)
        attn_output = self.relu(attn_output)
        attn_output = attn_output.squeeze(1)
        x = self.layer2(attn_output)
        return x

if __name__ == "__main__":
    model = TestModel()
    # graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    # graph_module(model, save_path=f"{graph_dir}/module_tree.png", include_params=True)
    print([name for name, _ in model.named_parameters()])
