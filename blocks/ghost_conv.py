"""
Ghost Convolution module.
"""

import torch
import torch.nn as nn
import math


class GhostConv(nn.Module):
    """
    Ghost Convolution module: generates more feature maps from intrinsic ones
    using cheap operations, inspired by the GhostNet paper.
    
    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        kernel_size (int): Kernel size for primary convolution (default 1).
        ratio (int): Ratio for channels split between primary and cheap conv (default 2).
        dw_size (int): Kernel size for depthwise (cheap) convolution (default 3).
        stride (int): Stride for primary convolution (default 1).
        relu (bool): Whether to apply ReLU activations (default True).
    """
    def __init__(self, inp: int, oup: int, kernel_size: int = 1, ratio: int = 2, 
                 dw_size: int = 3, stride: int = 1, relu: bool = True):
        super(GhostConv, self).__init__()
        self.oup = oup
        assert kernel_size % 2 == 1, "Kernel size should be odd for symmetric padding"
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        self.cheap_op = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=dw_size, stride=1, 
                      padding=dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_op(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]