"""
Fused Inverted Residual Block Module.

This module implements the Fused Inverted Residual block inspired by
EfficientNetV2 and MobileNetV3. It fuses the expansion and depthwise
convolutions into a single 3×3 convolution for improved efficiency.

Reference:
    Tan & Le, "EfficientNetV2: Smaller Models and Faster Training"
    https://arxiv.org/abs/2104.00298
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .utils import BaseBlock


class FusedInvertedResidualBlock(BaseBlock):
    """
    Fused Inverted Residual Block.
    
    Combines expansion and depthwise convolutions into a single fused 3×3
    convolution, followed by a 1×1 projection. This approach reduces memory
    access overhead and is more efficient on modern accelerators.
    
    Standard Inverted Residual:
        Expand (1×1) → Depthwise (3×3) → Project (1×1)
    
    Fused Inverted Residual:
        Fused Expand+DW (3×3) → Project (1×1)
    
    Residual Connection:
        Applied when stride=1 AND inp=oup
    
    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        stride (int): Stride for spatial downsampling. Default: 1.
        expand_ratio (int): Expansion factor for hidden dimension. Default: 4.
        use_ghost (bool): Whether to use GhostConv for expansion. Default: False.
        debug (bool): Enable debug mode for shape printing. Default: False.
    
    Shape:
        - Input: (B, C_in, H, W)
        - Output: (B, C_out, H', W') where H' = H/stride, W' = W/stride
    
    Examples:
        >>> # With residual connection (stride=1, inp=oup)
        >>> fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, stride=1)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> y = fused_ir(x)
        >>> y.shape
        torch.Size([2, 64, 56, 56])
        
        >>> # Without residual (stride=2, downsampling)
        >>> fused_ir = FusedInvertedResidualBlock(inp=64, oup=128, stride=2)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> y = fused_ir(x)
        >>> y.shape
        torch.Size([2, 128, 28, 28])
    
    Note:
        - Residual connection is applied when stride=1 AND inp=oup
        - When expand_ratio=1, no expansion occurs (single 3×3 conv)
        - hidden_dim = inp * expand_ratio
    """
    
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        expand_ratio: int = 4,
        use_ghost: bool = False,
        debug: bool = False
    ):
        super().__init__(debug=debug)
        
        # Validate parameters
        assert inp > 0, f"Input channels must be positive, got {inp}"
        assert oup > 0, f"Output channels must be positive, got {oup}"
        assert stride >= 1, f"Stride must be >= 1, got {stride}"
        assert expand_ratio >= 1, f"Expand ratio must be >= 1, got {expand_ratio}"
        
        # Store configuration
        self.inp = inp
        self.oup = oup
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_ghost = use_ghost
        
        # Calculate hidden dimension
        self.hidden_dim = int(round(inp * expand_ratio))
        
        # Determine if residual connection should be used
        self.use_res_connect = (stride == 1) and (inp == oup)
        
        # Build layers
        layers = []
        
        if expand_ratio != 1:
            # Fused expansion: 3×3 conv that expands channels
            if use_ghost:
                # Use GhostConv for expansion (lazy import to avoid circular)
                from .ghost_conv import GhostConv
                layers.append(
                    GhostConv(inp, self.hidden_dim, kernel_size=1, ratio=2, 
                              dw_size=3, stride=stride, relu=True)
                )
                # Note: GhostConv handles stride, so we need projection without stride
                # But GhostConv with stride already downsamples, so projection is 1x1
                layers.append(
                    nn.Conv2d(self.hidden_dim, oup, kernel_size=1, stride=1, 
                              padding=0, bias=False)
                )
                layers.append(nn.BatchNorm2d(oup))
            else:
                # Standard fused convolution
                layers.append(
                    nn.Conv2d(inp, self.hidden_dim, kernel_size=3, stride=stride,
                              padding=1, bias=False)
                )
                layers.append(nn.BatchNorm2d(self.hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                
                # Projection: 1×1 conv to reduce channels
                layers.append(
                    nn.Conv2d(self.hidden_dim, oup, kernel_size=1, stride=1,
                              padding=0, bias=False)
                )
                layers.append(nn.BatchNorm2d(oup))
        else:
            # No expansion: single 3×3 conv
            layers.append(
                nn.Conv2d(inp, oup, kernel_size=3, stride=stride,
                          padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(oup))
        
        self.block = nn.Sequential(*layers)
        
        # Final activation (only used when no residual)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Fused Inverted Residual Block.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
        
        Returns:
            Output tensor of shape (B, C_out, H', W')
        """
        self._debug_print("Input", x)
        
        # Apply block
        out = self.block(x)
        self._debug_print("After block", out)
        
        # Apply residual connection if applicable
        if self.use_res_connect:
            out = x + out
            self._debug_print("After residual addition", out)
        else:
            out = self.relu(out)
            self._debug_print("After ReLU (no residual)", out)
        
        return out
    
    def get_residual_info(self) -> Dict[str, Any]:
        """
        Return information about residual connection configuration.
        
        Returns:
            Dictionary with residual connection details
        """
        return {
            'use_res_connect': self.use_res_connect,
            'stride': self.stride,
            'inp': self.inp,
            'oup': self.oup,
            'reason': self._get_residual_reason()
        }
    
    def _get_residual_reason(self) -> str:
        """Get human-readable reason for residual connection status."""
        if self.use_res_connect:
            return "Applied: stride=1 AND inp=oup"
        elif self.stride != 1:
            return f"Not applied: stride={self.stride} (must be 1)"
        else:
            return f"Not applied: inp={self.inp} != oup={self.oup}"
    
    def extra_repr(self) -> str:
        """Return string representation with configuration."""
        return (
            f"inp={self.inp}, oup={self.oup}, stride={self.stride}, "
            f"expand_ratio={self.expand_ratio}, hidden_dim={self.hidden_dim}, "
            f"use_res_connect={self.use_res_connect}, use_ghost={self.use_ghost}"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'inp': self.inp,
            'oup': self.oup,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'hidden_dim': self.hidden_dim,
            'use_res_connect': self.use_res_connect,
            'use_ghost': self.use_ghost,
        }