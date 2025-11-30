"""
Feed-Forward Network and Residual LayerNorm Modules.

This module implements:
1. BottleneckFFN - A parameter-efficient FFN with bottleneck design
2. ResidualLayerNormBlock - LayerNorm with residual connection

These are core components of the transformer stage in MobilePlantViT.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .utils import BaseBlock


class BottleneckFFN(BaseBlock):
    """
    Bottleneck Feed-Forward Network.
    
    Unlike standard transformer FFNs that expand channels (4× expansion),
    this implementation uses a bottleneck design that contracts channels first,
    reducing parameters and computation significantly.
    
    Standard Transformer FFN:
        Input (D) → Expand (4D) → GELU → Contract (D) → Output
    
    Bottleneck FFN (this implementation):
        Input (D) → Contract (D×ratio) → GELU → Expand (D) → Output
    
    Args:
        inp (int): Input feature dimension.
        oup (int): Output feature dimension.
        bottleneck_ratio (float): Ratio for bottleneck dimension. Default: 0.25.
        dropout (float): Dropout probability. Default: 0.1.
        debug (bool): Enable debug mode for shape printing. Default: False.
    
    Shape:
        - Input: (B, N, D_in)
        - Output: (B, N, D_out)
    
    Examples:
        >>> # Standard usage (same input/output dim)
        >>> ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.25)
        >>> x = torch.randn(2, 196, 256)
        >>> y = ffn(x)
        >>> y.shape
        torch.Size([2, 196, 256])
        
        >>> # Different input/output dimensions
        >>> ffn = BottleneckFFN(inp=256, oup=128, bottleneck_ratio=0.5)
        >>> x = torch.randn(2, 196, 256)
        >>> y = ffn(x)
        >>> y.shape
        torch.Size([2, 196, 128])
    
    Note:
        - bottleneck_channels = max(1, int(inp * bottleneck_ratio))
        - With ratio=0.25 and inp=256: bottleneck=64 (16× fewer params than 4× expansion)
    """
    
    def __init__(
        self,
        inp: int,
        oup: int,
        bottleneck_ratio: float = 0.25,
        dropout: float = 0.1,
        debug: bool = False
    ):
        super().__init__(debug=debug)
        
        # Validate parameters
        assert inp > 0, f"Input dimension must be positive, got {inp}"
        assert oup > 0, f"Output dimension must be positive, got {oup}"
        assert 0 < bottleneck_ratio <= 2.0, f"Bottleneck ratio should be in (0, 2.0], got {bottleneck_ratio}"
        assert 0 <= dropout < 1.0, f"Dropout must be in [0, 1), got {dropout}"
        
        # Store configuration
        self.inp = inp
        self.oup = oup
        self.bottleneck_ratio = bottleneck_ratio
        self.dropout_rate = dropout
        
        # Calculate bottleneck channels (minimum 1)
        self.bottleneck_channels = max(1, int(inp * bottleneck_ratio))
        
        # Contraction layer: inp → bottleneck
        self.fc1 = nn.Linear(inp, self.bottleneck_channels)
        self.norm1 = nn.LayerNorm(self.bottleneck_channels)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Expansion layer: bottleneck → oup
        self.fc2 = nn.Linear(self.bottleneck_channels, oup)
        self.norm2 = nn.LayerNorm(oup)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Bottleneck FFN.
        
        Args:
            x: Input tensor of shape (B, N, D_in)
        
        Returns:
            Output tensor of shape (B, N, D_out)
        """
        self._debug_print("Input", x)
        
        # Contraction path
        x = self.fc1(x)
        self._debug_print("After fc1 (contraction)", x)
        
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout1(x)
        self._debug_print("After norm1+GELU+dropout1", x)
        
        # Expansion path
        x = self.fc2(x)
        self._debug_print("After fc2 (expansion)", x)
        
        x = self.norm2(x)
        x = self.dropout2(x)
        self._debug_print("After norm2+dropout2", x)
        
        return x
    
    def extra_repr(self) -> str:
        """Return string representation with configuration."""
        return (
            f"inp={self.inp}, oup={self.oup}, "
            f"bottleneck_ratio={self.bottleneck_ratio}, "
            f"bottleneck_channels={self.bottleneck_channels}, "
            f"dropout={self.dropout_rate}"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'inp': self.inp,
            'oup': self.oup,
            'bottleneck_ratio': self.bottleneck_ratio,
            'bottleneck_channels': self.bottleneck_channels,
            'dropout': self.dropout_rate,
        }


class ResidualLayerNormBlock(BaseBlock):
    """
    Residual Block with Layer Normalization.
    
    Combines LayerNorm with residual (skip) connections. This is a fundamental
    building block in transformer architectures that:
    1. Stabilizes training through skip connections (gradient highway)
    2. Normalizes activations using LayerNorm
    3. Enables deep networks by allowing gradients to flow directly
    
    This implementation uses post-norm style:
        Y = LayerNorm(x) + residual
    
    Args:
        embed_dim (int): Embedding dimension for LayerNorm.
        debug (bool): Enable debug mode for shape printing. Default: False.
    
    Shape:
        - Input x: (B, N, D)
        - Input residual (optional): (B, N, D)
        - Output: (B, N, D)
    
    Examples:
        >>> # Default residual (uses x as residual)
        >>> res_ln = ResidualLayerNormBlock(embed_dim=256)
        >>> x = torch.randn(2, 196, 256)
        >>> y = res_ln(x)  # y = LayerNorm(x) + x
        >>> y.shape
        torch.Size([2, 196, 256])
        
        >>> # Explicit residual (typical transformer pattern)
        >>> x_before = torch.randn(2, 196, 256)  # Input to sublayer
        >>> x_after = sublayer(x_before)         # Output from sublayer
        >>> y = res_ln(x_after, residual=x_before)  # y = LayerNorm(x_after) + x_before
    
    Note:
        LayerNorm is stateless - same computation in train and eval modes.
    """
    
    def __init__(
        self,
        embed_dim: int,
        debug: bool = False
    ):
        super().__init__(debug=debug)
        
        # Validate parameters
        assert embed_dim > 0, f"Embedding dimension must be positive, got {embed_dim}"
        
        # Store configuration
        self.embed_dim = embed_dim
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Residual LayerNorm Block.
        
        Args:
            x: Input tensor of shape (B, N, D), typically output from a sublayer
            residual: Optional residual tensor of shape (B, N, D).
                     If None, uses x as the residual.
        
        Returns:
            Output tensor of shape (B, N, D): LayerNorm(x) + residual
        """
        self._debug_print("Input x", x)
        
        # Use x as residual if not provided
        if residual is None:
            residual = x
        
        self._debug_print("Residual", residual)
        
        # Apply LayerNorm
        x_norm = self.norm(x)
        self._debug_print("After LayerNorm", x_norm)
        
        # Add residual
        out = x_norm + residual
        self._debug_print("Output (norm + residual)", out)
        
        return out
    
    def extra_repr(self) -> str:
        """Return string representation with configuration."""
        return f"embed_dim={self.embed_dim}"
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'embed_dim': self.embed_dim,
        }