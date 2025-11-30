"""
MobilePlantViT: Lightweight Hybrid CNN-Transformer for Plant Disease Classification.

This package provides:
- blocks: Individual neural network building blocks
- models: Complete model implementations
- utils: Testing and benchmarking utilities
"""

from . import blocks
from . import models
from . import utils

# Convenience imports
from .models import (
    MobilePlantViT,
    MobilePlantViTConfig,
    mobileplant_vit_tiny,
    mobileplant_vit_small,
    mobileplant_vit_base,
    mobileplant_vit_large,
)

__version__ = "0.1.0"

__all__ = [
    'blocks',
    'models',
    'utils',
    'MobilePlantViT',
    'MobilePlantViTConfig',
    'mobileplant_vit_tiny',
    'mobileplant_vit_small',
    'mobileplant_vit_base',
    'mobileplant_vit_large',
]