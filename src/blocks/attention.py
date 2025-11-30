"""
Linear Differential Attention (LDA) Module.

This module implements the Linear Differential Attention mechanism inspired by
the DIFF Transformer paper. It computes attention as the difference between
two softmax attention maps, which helps cancel noise and irrelevant information.

Reference:
    "DIFF Transformer" - https://arxiv.org/abs/2410.05258
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from .utils import BaseBlock


class LinearDifferentialAttention(BaseBlock):
    """
    Linear Differential Attention (LDA) Block.
    
    Computes attention as the weighted difference between two attention maps:
    A_diff = α × (A1 - A2)
    
    This differential mechanism helps:
    - Cancel out noise common to both attention maps
    - Enhance meaningful patterns that differ between maps
    - Provide learnable noise cancellation via α parameter
    
    Args:
        embed_dim (int): Input/output embedding dimension.
        num_heads (int): Number of attention heads. Default: 8.
        dropout (float): Dropout probability on attention output. Default: 0.1.
        init (float): Initial value for α (stored as exp(init)). Default: 0.8.
        debug (bool): Enable debug mode for shape printing. Default: False.
    
    Shape:
        - Input: (B, N, D) where B=batch, N=sequence length, D=embed_dim
        - Output: (B, N, D) same shape as input
    
    Examples:
        >>> lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        >>> x = torch.randn(2, 196, 256)  # 14x14 patches, 256-dim embeddings
        >>> y = lda(x)
        >>> y.shape
        torch.Size([2, 196, 256])
    
    Note:
        - embed_dim must be divisible by num_heads
        - α is always positive due to exp() initialization
        - Q and K projections output 2×embed_dim (for Q1/Q2 and K1/K2)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        init: float = 0.8,
        debug: bool = False
    ):
        super().__init__(debug=debug)
        
        # Validate parameters
        assert embed_dim > 0, f"embed_dim must be positive, got {embed_dim}"
        assert num_heads > 0, f"num_heads must be positive, got {num_heads}"
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert 0.0 <= dropout < 1.0, f"dropout must be in [0, 1), got {dropout}"
        
        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_rate = dropout
        
        # Learnable differential weight (α)
        # Stored as exp(init) to ensure positivity
        self.alpha = nn.Parameter(torch.tensor(init).exp())
        
        # Projections
        # Q and K project to 2×embed_dim (for Q1/Q2 and K1/K2)
        self.q_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Normalization (GroupNorm with num_groups=num_heads for head-wise normalization)
        self.norm = nn.GroupNorm(num_groups=num_heads, num_channels=embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        # Xavier uniform initialization for projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Linear Differential Attention.
        
        Args:
            x: Input tensor of shape (B, N, D)
        
        Returns:
            Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape
        self._debug_print("Input", x)
        
        # Normalize input (GroupNorm expects (B, C, *) so we transpose)
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)
        self._debug_print("After GroupNorm", x_norm)
        
        # Project to Q, K, V
        q = self.q_proj(x_norm)  # (B, N, 2D)
        k = self.k_proj(x_norm)  # (B, N, 2D)
        v = self.v_proj(x_norm)  # (B, N, D)
        self._debug_print("Q projection", q)
        self._debug_print("K projection", k)
        self._debug_print("V projection", v)
        
        # Split Q and K into Q1/Q2 and K1/K2
        Q1, Q2 = q.chunk(2, dim=-1)  # Each (B, N, D)
        K1, K2 = k.chunk(2, dim=-1)  # Each (B, N, D)
        
        # Reshape for multi-head attention: (B, N, D) -> (B, heads, N, head_dim)
        Q1 = Q1.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        Q2 = Q2.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K1 = K1.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K2 = K2.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        self._debug_print("Q1 (multi-head)", Q1)
        self._debug_print("V (multi-head)", V)
        
        # Compute attention scores
        # A1 = softmax(Q1 @ K1.T / sqrt(head_dim))
        scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) * self.scaling  # (B, heads, N, N)
        scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) * self.scaling
        self._debug_print("Scores1", scores1)
        self._debug_print("Scores2", scores2)
        
        # Apply softmax
        A1 = F.softmax(scores1, dim=-1)  # (B, heads, N, N)
        A2 = F.softmax(scores2, dim=-1)
        self._debug_print("A1 (softmax)", A1)
        self._debug_print("A2 (softmax)", A2)
        
        # Differential attention: α × (A1 - A2)
        A_diff = self.alpha * (A1 - A2)
        self._debug_print("A_diff (differential)", A_diff)
        
        # Apply attention to values
        out = torch.matmul(A_diff, V)  # (B, heads, N, head_dim)
        self._debug_print("After attention @ V", out)
        
        # Reshape back: (B, heads, N, head_dim) -> (B, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        self._debug_print("After reshape", out)
        
        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)
        self._debug_print("Output", out)
        
        return out
    
    def get_attention_maps(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input tensor of shape (B, N, D)
        
        Returns:
            Tuple of (A1, A2, A_diff) attention maps, each of shape (B, heads, N, N)
        """
        B, N, D = x.shape
        
        # Normalize and project
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        
        # Split and reshape
        Q1, Q2 = q.chunk(2, dim=-1)
        K1, K2 = k.chunk(2, dim=-1)
        
        Q1 = Q1.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        Q2 = Q2.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K1 = K1.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K2 = K2.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention maps
        scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) * self.scaling
        scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) * self.scaling
        
        A1 = F.softmax(scores1, dim=-1)
        A2 = F.softmax(scores2, dim=-1)
        A_diff = self.alpha * (A1 - A2)
        
        return A1, A2, A_diff
    
    def get_alpha(self) -> float:
        """Return current alpha value."""
        return self.alpha.item()
    
    def extra_repr(self) -> str:
        """Return string representation with configuration."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, dropout={self.dropout_rate}, "
            f"alpha={self.alpha.item():.4f}"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout': self.dropout_rate,
            'alpha': self.alpha.item(),
        }


class NaiveFullAttention(nn.Module):
    """
    Naive Full Attention for reference/comparison.
    
    Standard scaled dot-product attention without differential mechanism.
    Used for testing and validation only.
    
    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        # Project
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        attn = F.softmax(scores, dim=-1)
        
        # Apply to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention maps for visualization."""
        B, N, D = x.shape
        
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        attn = F.softmax(scores, dim=-1)
        
        return attn