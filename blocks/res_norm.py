"""
Residual Layer Normalization module.
"""

import torch
import torch.nn as nn


class ResidualLayerNormBlock(nn.Module):
    """
    Residual block combined with Layer Normalization.
    
    Args:
        embed_dim (int): Embedding dimension of the tokens.
    """
    def __init__(self, embed_dim: int):
        super(ResidualLayerNormBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        if residual is None:
            residual = x
        x = self.norm(x)
        return x + residual