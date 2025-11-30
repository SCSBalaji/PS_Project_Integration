"""
Positional Encoding Module.

This module implements sinusoidal positional encoding as introduced in
"Attention Is All You Need" (Vaswani et al., 2017).

Positional encoding adds spatial position information to patch embeddings,
which is essential since transformer self-attention is permutation-invariant.

Reference:
    Vaswani et al., "Attention Is All You Need"
    https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional

from .utils import BaseBlock


class PositionalEncoding(BaseBlock):
    """
    Sinusoidal Positional Encoding.
    
    Adds fixed (non-learnable) positional information to patch embeddings
    using sine and cosine functions of different frequencies:
    
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where:
        - pos = position in sequence (0 to N-1)
        - i = dimension index (0 to d_model/2 - 1)
        - d_model = embed_dim
    
    Args:
        embed_dim (int): Embedding dimension (must match patch embedding output).
        max_len (int): Maximum sequence length supported. Default: 5000.
        debug (bool): Enable debug mode for shape printing.
    
    Shape:
        - Input: (B, N, D) — Patch embeddings
        - Output: (B, N, D) — Patch embeddings with positional information added
        
        Where N <= max_len
    
    Examples:
        >>> pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        >>> patches = torch.randn(2, 196, 256)
        >>> output = pos_enc(patches)
        >>> output.shape
        torch.Size([2, 196, 256])
        
        >>> # Verify position information is added
        >>> (output != patches).any()
        True
    
    Note:
        - Positional encodings are stored as a buffer (not a parameter)
        - No learnable parameters (parameter count = 0)
        - Supports variable sequence lengths up to max_len
        - Uses even indices for sin, odd indices for cos
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
        debug: bool = False
    ):
        super().__init__(debug=debug)
        
        # Validate parameters
        assert embed_dim > 0, f"embed_dim must be positive, got {embed_dim}"
        assert max_len > 0, f"max_len must be positive, got {max_len}"
        
        # Store configuration
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Precompute positional encoding matrix
        pe = self._create_positional_encoding(embed_dim, max_len)
        
        # Register as buffer (saved with model, moved to device, but not a parameter)
        self.register_buffer('pe', pe)
    
    def _create_positional_encoding(self, embed_dim: int, max_len: int) -> torch.Tensor:
        """
        Create the positional encoding matrix.
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
        
        Returns:
            Positional encoding tensor of shape (1, max_len, embed_dim)
        """
        # Initialize PE matrix
        pe = torch.zeros(max_len, embed_dim)
        
        # Position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Division term: 10000^(2i/d_model) computed as exp(2i * -log(10000)/d_model)
        # This is more numerically stable than direct computation
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        # Handle case where embed_dim is odd
        if embed_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Add batch dimension: (max_len, embed_dim) -> (1, max_len, embed_dim)
        pe = pe.unsqueeze(0)
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (B, N, D)
        
        Returns:
            Output tensor of shape (B, N, D) with positional information added
        """
        self._debug_print("Input", x)
        
        # Get sequence length
        seq_len = x.size(1)
        
        # Check sequence length is within bounds
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}. "
                f"Increase max_len or reduce sequence length."
            )
        
        # Slice positional encoding to match sequence length and add
        # pe: (1, max_len, D) -> (1, N, D)
        # Broadcasting handles batch dimension
        x = x + self.pe[:, :seq_len, :]
        
        self._debug_print("Output", x)
        
        return x
    
    def get_encoding(self, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Get the positional encoding matrix for visualization.
        
        Args:
            seq_len: Sequence length to return. If None, returns full matrix.
        
        Returns:
            Positional encoding tensor of shape (1, seq_len, embed_dim)
        """
        if seq_len is None:
            return self.pe
        return self.pe[:, :seq_len, :]
    
    def visualize_encoding(self, seq_len: int = 100) -> torch.Tensor:
        """
        Get positional encoding matrix suitable for visualization.
        
        Args:
            seq_len: Number of positions to visualize
        
        Returns:
            2D tensor of shape (seq_len, embed_dim) for heatmap visualization
        """
        return self.pe[0, :seq_len, :].clone()
    
    def extra_repr(self) -> str:
        """Return string representation with configuration."""
        return f"embed_dim={self.embed_dim}, max_len={self.max_len}"
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'embed_dim': self.embed_dim,
            'max_len': self.max_len,
            'learnable': False,
        }