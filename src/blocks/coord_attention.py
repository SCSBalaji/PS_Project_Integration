"""
Coordinate Attention Module.

This module implements Coordinate Attention from the paper:
"Coordinate Attention for Efficient Mobile Network Design"
https://arxiv.org/abs/2103.02907

Coordinate Attention encodes channel relationships and long-range dependencies
with precise positional information, using two 1D feature encoding operations.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

from .utils import BaseBlock


class HSigmoid(nn.Module):
    """
    Hard Sigmoid activation function.
    
    Computes: ReLU6(x + 3) / 6
    
    This is a computationally efficient approximation of the sigmoid function,
    commonly used in mobile architectures.
    
    Args:
        inplace: Whether to perform the operation in-place. Default: True.
    
    Shape:
        - Input: (*)
        - Output: (*) same shape as input
    
    Examples:
        >>> hsigmoid = HSigmoid()
        >>> x = torch.randn(2, 64, 56, 56)
        >>> y = hsigmoid(x)
        >>> y.shape
        torch.Size([2, 64, 56, 56])
    """
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu6(x + 3.0) / 6.0


class HSwish(nn.Module):
    """
    Hard Swish activation function.
    
    Computes: x * HSigmoid(x) = x * ReLU6(x + 3) / 6
    
    This is a computationally efficient approximation of the Swish activation,
    commonly used in MobileNetV3 and efficient architectures.
    
    Args:
        inplace: Whether to perform the operation in-place. Default: True.
    
    Shape:
        - Input: (*)
        - Output: (*) same shape as input
    
    Examples:
        >>> hswish = HSwish()
        >>> x = torch.randn(2, 64, 56, 56)
        >>> y = hswish(x)
        >>> y.shape
        torch.Size([2, 64, 56, 56])
    """
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.hsigmoid = HSigmoid(inplace=inplace)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.hsigmoid(x)


class CoordAtt(BaseBlock):
    """
    Coordinate Attention Block.
    
    Encodes channel relationships and long-range dependencies with precise
    positional information by decomposing channel attention into two 1D
    feature encoding processes along the height and width directions.
    
    The attention mechanism:
    1. Applies adaptive average pooling along H and W separately
    2. Concatenates and processes through a shared 1x1 conv + BN + activation
    3. Splits and generates separate H and W attention maps
    4. Applies attention to input via element-wise multiplication
    
    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        reduction (int): Channel reduction ratio for bottleneck. Default: 32.
        debug (bool): Enable debug mode for shape printing. Default: False.
    
    Shape:
        - Input: (B, inp, H, W)
        - Output: (B, oup, H, W)
    
    Examples:
        >>> # Basic usage with same input/output channels
        >>> coord_att = CoordAtt(inp=64, oup=64)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> y = coord_att(x)
        >>> y.shape
        torch.Size([2, 64, 56, 56])
        
        >>> # With different input/output channels
        >>> coord_att = CoordAtt(inp=64, oup=128)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> y = coord_att(x)
        >>> y.shape
        torch.Size([2, 128, 56, 56])
        
        >>> # With custom reduction ratio
        >>> coord_att = CoordAtt(inp=128, oup=128, reduction=16)
        >>> x = torch.randn(2, 128, 28, 28)
        >>> y = coord_att(x)
        >>> y.shape
        torch.Size([2, 128, 28, 28])
    
    Note:
        The intermediate channel dimension (mip) is computed as:
        mip = max(8, inp // reduction)
        This ensures a minimum of 8 channels in the bottleneck.
    """
    
    def __init__(
        self,
        inp: int,
        oup: int,
        reduction: int = 32,
        debug: bool = False
    ):
        super().__init__(debug=debug)
        
        # Validate parameters
        assert inp > 0, f"Input channels must be positive, got {inp}"
        assert oup > 0, f"Output channels must be positive, got {oup}"
        assert reduction >= 1, f"Reduction must be >= 1, got {reduction}"
        
        # Store configuration
        self.inp = inp
        self.oup = oup
        self.reduction = reduction
        
        # Compute bottleneck channels (minimum 8)
        self.mip = max(8, inp // reduction)
        
        # Pooling operations for height and width
        # pool_h: (B, C, H, W) -> (B, C, H, 1)
        # pool_w: (B, C, H, W) -> (B, C, 1, W)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Shared 1x1 convolution for dimension reduction
        self.conv1 = nn.Conv2d(inp, self.mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.mip)
        self.act = HSwish()
        
        # Separate convolutions for height and width attention
        # Output channels = oup to match the output dimension
        self.conv_h = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mip, oup, kernel_size=1, stride=1, padding=0)
        
        # If inp != oup, we need a projection for the identity/skip connection
        if inp != oup:
            self.proj = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Coordinate Attention.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
        
        Returns:
            Output tensor of shape (B, C_out, H, W) with coordinate attention applied
        """
        self._debug_print("Input", x)
        
        # Store identity for residual-like connection
        identity = x
        
        n, c, h, w = x.size()
        
        # Aggregate spatial information
        # x_h: (B, C, H, 1) - average over width
        x_h = self.pool_h(x)
        self._debug_print("After pool_h", x_h)
        
        # x_w: (B, C, 1, W) -> (B, C, W, 1) for concatenation
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        self._debug_print("After pool_w (permuted)", x_w)
        
        # Concatenate along the spatial dimension (dim=2)
        # y: (B, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)
        self._debug_print("After concat", y)
        
        # Reduce channels and apply activation
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        self._debug_print("After conv1+bn1+act", y)
        
        # Split back to height and width components
        x_h, x_w = torch.split(y, [h, w], dim=2)
        self._debug_print("Split x_h", x_h)
        
        # Restore x_w shape: (B, mip, W, 1) -> (B, mip, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)
        self._debug_print("Split x_w (permuted back)", x_w)
        
        # Generate attention maps with sigmoid
        # a_h: (B, oup, H, 1)
        a_h = self.conv_h(x_h).sigmoid()
        # a_w: (B, oup, 1, W)
        a_w = self.conv_w(x_w).sigmoid()
        self._debug_print("Attention a_h", a_h)
        self._debug_print("Attention a_w", a_w)
        
        # Project identity if needed
        if self.proj is not None:
            identity = self.proj(identity)
            self._debug_print("After projection", identity)
        
        # Apply attention: element-wise multiplication
        # Broadcasting: (B, oup, H, W) * (B, oup, H, 1) * (B, oup, 1, W)
        out = identity * a_h * a_w
        self._debug_print("Output", out)
        
        return out
    
    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention maps without applying them.
        
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Tuple of (a_h, a_w) attention maps
            - a_h: Height attention of shape (B, oup, H, 1)
            - a_w: Width attention of shape (B, oup, 1, W)
        """
        n, c, h, w = x.size()
        
        # Aggregate spatial information
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split and generate attention maps
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return a_h, a_w
    
    def extra_repr(self) -> str:
        """Return string representation with configuration."""
        return (
            f"inp={self.inp}, oup={self.oup}, "
            f"reduction={self.reduction}, mip={self.mip}"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'inp': self.inp,
            'oup': self.oup,
            'reduction': self.reduction,
            'mip': self.mip,
        }