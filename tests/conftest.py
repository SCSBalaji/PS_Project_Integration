"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def device():
    """Provide the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_image_tensor():
    """Provide a sample image tensor (batch=1, channels=3, height=224, width=224)."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch_tensor():
    """Provide a batch of image tensors (batch=4, channels=3, height=224, width=224)."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_feature_map():
    """Provide a sample feature map tensor (batch=1, channels=64, height=56, width=56)."""
    return torch.randn(1, 64, 56, 56)


@pytest.fixture
def sample_sequence():
    """Provide a sample sequence tensor for transformer (batch=1, seq_len=196, embed_dim=256)."""
    return torch.randn(1, 196, 256)


@pytest.fixture
def num_classes():
    """Number of classes for PlantVillage dataset."""
    return 38


@pytest.fixture
def default_config():
    """Provide default configuration dictionary."""
    return {
        'reproducibility': {
            'seed': 42,
            'cudnn_deterministic': True,
            'cudnn_benchmark': False
        },
        'dataset': {
            'num_classes': 38,
            'image_size': 224
        },
        'model': {
            'name': 'MobilePlantViT'
        }
    }