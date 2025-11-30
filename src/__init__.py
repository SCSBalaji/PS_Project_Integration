"""
MobilePlantViT Source Package.

This package contains:
- blocks: Individual building blocks for the MobilePlantViT architecture
- models: Complete model implementations including MobilePlantViT

Usage:
    from src.models import MobilePlantViT, MobilePlantViTConfig
    from src.models import mobileplant_vit_tiny, mobileplant_vit_small
    from src.blocks import GhostConv, FusedInvertedResidualBlock
"""

__version__ = "1.0.0"
__author__ = "MobilePlantViT Team"

# Package-level imports for convenience
from .models import (
    MobilePlantViT,
    MobilePlantViTConfig,
    mobileplant_vit_tiny,
    mobileplant_vit_small,
    mobileplant_vit_base,
    mobileplant_vit_large,
)

# Also expose key blocks at package level
from .blocks import (
    GhostConv,
    CoordAtt,
    FusedInvertedResidualBlock,
    LinearDifferentialAttention,
    PatchEmbedding,
    PositionalEncoding,
    BottleneckFFN,
    ResidualLayerNormBlock,
    GlobalAveragePooling,
    ClassifierHead,
)

__all__ = [
    # Models
    "MobilePlantViT",
    "MobilePlantViTConfig",
    "mobileplant_vit_tiny",
    "mobileplant_vit_small",
    "mobileplant_vit_base",
    "mobileplant_vit_large",
    # Blocks
    "GhostConv",
    "CoordAtt",
    "FusedInvertedResidualBlock",
    "LinearDifferentialAttention",
    "PatchEmbedding",
    "PositionalEncoding",
    "BottleneckFFN",
    "ResidualLayerNormBlock",
    "GlobalAveragePooling",
    "ClassifierHead",
]