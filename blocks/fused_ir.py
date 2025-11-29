"""
Fused Inverted Residual Block module.
"""

import torch
import torch.nn as nn


class FusedInvertedResidualBlock(nn.Module):
    """
    Fused Inverted Residual block:
    Combines expansion and depthwise convolutions into one fused conv for efficient computation,
    followed by a projection convolution to reduce channels.
    
    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        stride (int): Stride for the first convolution (default 1).
        expand_ratio (int): Expansion factor for hidden dimension (default 4).
    """
    def __init__(self, inp: int, oup: int, stride: int = 1, expand_ratio: int = 4):
        super(FusedInvertedResidualBlock, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (self.stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(inp, hidden_dim, kernel_size=3, stride=stride, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        else:
            hidden_dim = inp
        
        layers.append(
            nn.Conv2d(hidden_dim, oup, kernel_size=1 if expand_ratio != 1 else 3, 
                      stride=1, padding=0 if expand_ratio != 1 else 1, bias=False)
        )
        layers.append(nn.BatchNorm2d(oup))

        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.relu(self.block(x))