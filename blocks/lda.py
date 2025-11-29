"""
Linear Differential Attention module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDifferentialAttention(nn.Module):
    """
    Linear Differential Attention (LDA) block.
    
    Args:
        embed_dim (int): Input embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        init (float): Initialization scalar constant.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, init: float = 0.8):
        super(LinearDifferentialAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.alpha = nn.Parameter(torch.tensor(init).exp())
        
        self.q_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.GroupNorm(num_groups=num_heads, num_channels=embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)
        
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        Q1, Q2 = q.chunk(2, dim=-1)
        K1, K2 = k.chunk(2, dim=-1)
        
        Q1 = Q1.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        Q2 = Q2.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K1 = K1.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K2 = K2.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) * self.scaling
        scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) * self.scaling
        
        A1 = F.softmax(scores1, dim=-1)
        A2 = F.softmax(scores2, dim=-1)
        
        attn = self.alpha * (A1 - A2)
        
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out