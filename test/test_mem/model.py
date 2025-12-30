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
        attn_output, _ = self.attn(x, x, x)
        attn_output = self.relu(attn_output)
        x = self.layer2(attn_output)
        # nn.CrossEntropyLoss expects (N, C) output for classification
        # We'll take the first time-step only for loss (reshape to (batch, C))
        return x[:, 0, :] if x.dim() == 3 else x

if __name__ == "__main__":
    model = TestModel()

    def attn_backward_hook(module: nn.Module, grad_input, grad_output):
        print("[Parameters]: ]\n", [pname for pname, _ in module.named_parameters()])
        print("[Grad Input]: \n", [gi.shape for gi in grad_input if gi is not None])
        print("[Grad Output]: \n", [go.shape for go in grad_output if go is not None])

    # Register full backward hook on the attn layer
    model.attn.register_full_backward_hook(attn_backward_hook)

    # Do a small forward and backward pass
    x = torch.randn(8, 4, 16)  # batch of 8, sequence len 4, input_dim=16
    out = model(x)
    target = torch.randint(0, 4, (8,))  # output_dim=4, 8 samples: shape [8]

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(out, target)
    loss.backward()

