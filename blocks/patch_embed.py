"""
Patch Embedding and Positional Encoding modules.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Convert spatial feature map into patch tokens.
    
    Args:
        in_channels (int): Number of input channels.
        embed_dim (int): Embedding dimension of output tokens.
        patch_size (int): Size of patches (height and width).
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """
    Add sinusoidal positional encoding to patch tokens.
    
    Args:
        embed_dim (int): Dimension of the embeddings.
        max_len (int): Maximum length of the sequence.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x