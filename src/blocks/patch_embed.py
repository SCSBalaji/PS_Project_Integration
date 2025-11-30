"""
Patch Embedding Module.

This module implements Patch Embedding for converting spatial feature maps
from the CNN stage into a sequence of patch tokens suitable for transformer
processing.

Reference:
    Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for 
    Image Recognition at Scale" (ViT)
    https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from .utils import BaseBlock


class PatchEmbedding(BaseBlock):
    """
    Patch Embedding Layer.
    
    Converts spatial feature maps into a sequence of patch tokens by:
    1. Dividing the feature map into non-overlapping patches
    2. Projecting each patch into the embedding dimension
    3. Flattening the spatial layout into sequence format
    
    This is implemented efficiently using a strided convolution where
    kernel_size = stride = patch_size.
    
    Args:
        in_channels (int): Number of input channels from CNN stage.
        embed_dim (int): Output embedding dimension for each patch token.
        patch_size (int): Size of each square patch (height and width).
        bias (bool): Whether to use bias in the projection convolution.
        debug (bool): Enable debug mode for shape printing.
    
    Shape:
        - Input: (B, C, H, W) — Spatial feature map from CNN
        - Output: (B, N, D) — Sequence of patch embeddings
        
        Where:
            - B = batch size
            - C = in_channels
            - H, W = spatial dimensions (should be divisible by patch_size)
            - N = num_patches = (H // patch_size) * (W // patch_size)
            - D = embed_dim
    
    Examples:
        >>> # Convert CNN features to patch tokens
        >>> patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        >>> cnn_output = torch.randn(2, 64, 56, 56)
        >>> tokens = patch_embed(cnn_output)
        >>> tokens.shape
        torch.Size([2, 196, 256])  # 196 = (56/4) * (56/4) = 14 * 14
        
        >>> # With different patch size
        >>> patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=2)
        >>> cnn_output = torch.randn(2, 64, 14, 14)
        >>> tokens = patch_embed(cnn_output)
        >>> tokens.shape
        torch.Size([2, 49, 256])  # 49 = (14/2) * (14/2) = 7 * 7
    
    Note:
        - H and W should be divisible by patch_size for exact patching
        - If not divisible, patches at the edge will be truncated
        - Patches are ordered row-major (left-to-right, top-to-bottom)
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        patch_size: int = 4,
        bias: bool = True,
        debug: bool = False
    ):
        super().__init__(debug=debug)
        
        # Validate parameters
        assert in_channels > 0, f"in_channels must be positive, got {in_channels}"
        assert embed_dim > 0, f"embed_dim must be positive, got {embed_dim}"
        assert patch_size > 0, f"patch_size must be positive, got {patch_size}"
        
        # Store configuration
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Projection convolution
        # Using kernel_size = stride = patch_size extracts non-overlapping patches
        # and projects them to embed_dim in one operation
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Convert spatial features to patch sequence.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Patch embeddings of shape (B, N, D)
        """
        self._debug_print("Input", x)
        
        B, C, H, W = x.shape
        
        # Apply projection convolution
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        self._debug_print("After projection conv", x)
        
        # Flatten spatial dimensions
        # (B, D, H', W') -> (B, D, N) where N = H' * W'
        x = x.flatten(2)
        self._debug_print("After flatten", x)
        
        # Transpose to sequence format
        # (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)
        self._debug_print("Output (transposed)", x)
        
        return x
    
    def get_num_patches(self, H: int, W: int) -> int:
        """
        Calculate the number of patches for given spatial dimensions.
        
        Args:
            H: Height of input feature map
            W: Width of input feature map
        
        Returns:
            Number of patches: (H // patch_size) * (W // patch_size)
        """
        return (H // self.patch_size) * (W // self.patch_size)
    
    def get_output_shape(self, H: int, W: int, batch_size: int = 1) -> Tuple[int, int, int]:
        """
        Get the output shape for given input spatial dimensions.
        
        Args:
            H: Height of input feature map
            W: Width of input feature map
            batch_size: Batch size
        
        Returns:
            Tuple of (batch_size, num_patches, embed_dim)
        """
        num_patches = self.get_num_patches(H, W)
        return (batch_size, num_patches, self.embed_dim)
    
    def extra_repr(self) -> str:
        """Return string representation with configuration."""
        return (
            f"in_channels={self.in_channels}, embed_dim={self.embed_dim}, "
            f"patch_size={self.patch_size}"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'in_channels': self.in_channels,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
        }