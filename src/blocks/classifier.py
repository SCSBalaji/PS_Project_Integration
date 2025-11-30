"""
Classification Head Modules.

This module implements:
1. GlobalAveragePooling - Aggregates sequence features into a single vector
2. ClassifierHead - Projects features to class probabilities

These are the final components in the MobilePlantViT pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .utils import BaseBlock


class GlobalAveragePooling(BaseBlock):
    """
    Global Average Pooling over the sequence dimension.
    
    Aggregates information from all patches/tokens into a single feature
    vector by computing the mean across the sequence dimension. This is
    a parameter-free operation that provides translation invariance.
    
    Why GAP instead of [CLS] token:
        - Parameter-free: No additional learnable token required
        - Translation invariance: Equal contribution from all positions
        - Simplicity: No special token handling needed
        - Robustness: Averages out noise from individual patches
    
    Args:
        debug (bool): Enable debug mode for shape printing. Default: False.
    
    Shape:
        - Input: (B, N, D) — Batch, Sequence Length, Embedding Dim
        - Output: (B, D) — Batch, Embedding Dim
    
    Examples:
        >>> gap = GlobalAveragePooling()
        >>> x = torch.randn(2, 196, 256)  # 196 patches, 256-dim embeddings
        >>> y = gap(x)
        >>> y.shape
        torch.Size([2, 256])
        
        >>> # Verify averaging
        >>> manual_avg = x.mean(dim=1)
        >>> torch.allclose(y, manual_avg)
        True
    
    Note:
        This is a parameter-free operation with 0 learnable parameters.
    """
    
    def __init__(self, debug: bool = False):
        super().__init__(debug=debug)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Global Average Pooling.
        
        Args:
            x: Input tensor of shape (B, N, D)
        
        Returns:
            Output tensor of shape (B, D)
        """
        self._debug_print("Input", x)
        
        # Average over sequence dimension (dim=1)
        out = x.mean(dim=1)
        
        self._debug_print("Output (after mean)", out)
        
        return out
    
    def extra_repr(self) -> str:
        """Return string representation."""
        return "dim=1 (sequence dimension)"
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'pooling_dim': 1,
            'parameters': 0,
        }


class ClassifierHead(BaseBlock):
    """
    Classification Head with Linear projection and Softmax.
    
    Maps the pooled feature vector to class probabilities. Consists of
    a single linear layer followed by softmax activation.
    
    Args:
        embed_dim (int): Input embedding dimension (from GAP output).
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability before classification. Default: 0.0.
        debug (bool): Enable debug mode for shape printing. Default: False.
    
    Shape:
        - Input: (B, D) — Batch, Embedding Dim
        - Output: (B, num_classes) — Batch, Class Probabilities
    
    Examples:
        >>> classifier = ClassifierHead(embed_dim=256, num_classes=38)
        >>> x = torch.randn(2, 256)  # Pooled features
        >>> probs = classifier(x)
        >>> probs.shape
        torch.Size([2, 38])
        >>> probs.sum(dim=-1)  # Each row sums to 1
        tensor([1., 1.])
        
        >>> # Get raw logits for CrossEntropyLoss
        >>> logits = classifier.get_logits(x)
        >>> logits.shape
        torch.Size([2, 38])
    
    Note:
        For training with nn.CrossEntropyLoss, use get_logits() method
        which returns raw scores before softmax.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        debug: bool = False
    ):
        super().__init__(debug=debug)
        
        # Validate parameters
        assert embed_dim > 0, f"Embedding dimension must be positive, got {embed_dim}"
        assert num_classes > 0, f"Number of classes must be positive, got {num_classes}"
        assert 0 <= dropout < 1.0, f"Dropout must be in [0, 1), got {dropout}"
        
        # Store configuration
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Classifier Head.
        
        Args:
            x: Input tensor of shape (B, D)
        
        Returns:
            Probability tensor of shape (B, num_classes)
        """
        self._debug_print("Input", x)
        
        # Apply dropout
        x = self.dropout(x)
        self._debug_print("After dropout", x)
        
        # Linear projection
        logits = self.fc(x)
        self._debug_print("Logits", logits)
        
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        self._debug_print("Probabilities", probs)
        
        return probs
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits without softmax.
        
        Use this for training with nn.CrossEntropyLoss which
        internally applies log-softmax.
        
        Args:
            x: Input tensor of shape (B, D)
        
        Returns:
            Logit tensor of shape (B, num_classes)
        """
        x = self.dropout(x)
        return self.fc(x)
    
    def extra_repr(self) -> str:
        """Return string representation with configuration."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_classes={self.num_classes}, "
            f"dropout={self.dropout_rate}"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
        }


class CombinedClassifier(BaseBlock):
    """
    Combined GAP + ClassifierHead for convenience.
    
    Combines Global Average Pooling and Classification Head into a
    single module for easier integration.
    
    Args:
        embed_dim (int): Input embedding dimension.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability. Default: 0.0.
        debug (bool): Enable debug mode. Default: False.
    
    Shape:
        - Input: (B, N, D) — Sequence from transformer
        - Output: (B, num_classes) — Class probabilities
    
    Examples:
        >>> classifier = CombinedClassifier(embed_dim=256, num_classes=38)
        >>> x = torch.randn(2, 196, 256)  # Transformer output
        >>> probs = classifier(x)
        >>> probs.shape
        torch.Size([2, 38])
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        debug: bool = False
    ):
        super().__init__(debug=debug)
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        self.gap = GlobalAveragePooling(debug=debug)
        self.head = ClassifierHead(embed_dim, num_classes, dropout, debug)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GAP and ClassifierHead.
        
        Args:
            x: Input tensor of shape (B, N, D)
        
        Returns:
            Probability tensor of shape (B, num_classes)
        """
        x = self.gap(x)
        x = self.head(x)
        return x
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits without softmax."""
        x = self.gap(x)
        return self.head.get_logits(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'class': self.__class__.__name__,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'gap_config': self.gap.get_config(),
            'head_config': self.head.get_config(),
        }