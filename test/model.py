import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=4, num_heads=4):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.learnable = nn.Parameter(torch.Tensor(1))
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_output, _ = self.attn(x, x, x)
        attn_output = self.relu(attn_output)
        attn_output = attn_output.squeeze(1)
        x = self.layer2(attn_output)
        return x
