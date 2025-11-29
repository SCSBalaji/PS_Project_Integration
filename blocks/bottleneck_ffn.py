"""
Bottleneck Feed Forward Network module.
"""

import torch
import torch.nn as nn


class BottleneckFFN(nn.Module):
    """
    Bottleneck Feed Forward Network used in transformer variants.
    
    Args:
        inp (int): Input feature dimension (embedding dimension).
        oup (int): Output feature dimension.
        bottleneck_ratio (float): Reduction ratio for bottleneck hidden dimension.
        dropout (float): Dropout rate.
    """
    def __init__(self, inp: int, oup: int, bottleneck_ratio: float = 0.25, dropout: float = 0.1):
        super(BottleneckFFN, self).__init__()
        bottleneck_channels = max(1, int(inp * bottleneck_ratio))
        
        self.fc1 = nn.Linear(inp, bottleneck_channels)
        self.norm1 = nn.LayerNorm(bottleneck_channels)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(bottleneck_channels, oup)
        self.norm2 = nn.LayerNorm(oup)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        
        return x