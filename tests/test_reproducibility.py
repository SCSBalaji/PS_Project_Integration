"""
Tests for reproducibility utilities.
"""

import pytest
import torch
import numpy as np
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSetSeed:
    """Tests for seed setting functionality."""
    
    def test_python_random_reproducibility(self):
        """Test Python random is reproducible with same seed."""
        from utils import set_seed
        
        set_seed(42)
        values_1 = [random.random() for _ in range(10)]
        
        set_seed(42)
        values_2 = [random.random() for _ in range(10)]
        
        assert values_1 == values_2, "Python random should be reproducible"
    
    def test_numpy_random_reproducibility(self):
        """Test NumPy random is reproducible with same seed."""
        from utils import set_seed
        
        set_seed(42)
        values_1 = np.random.rand(10).tolist()
        
        set_seed(42)
        values_2 = np.random.rand(10).tolist()
        
        assert values_1 == values_2, "NumPy random should be reproducible"
    
    def test_torch_random_reproducibility(self):
        """Test PyTorch random is reproducible with same seed."""
        from utils import set_seed
        
        set_seed(42)
        values_1 = torch.rand(10).tolist()
        
        set_seed(42)
        values_2 = torch.rand(10).tolist()
        
        assert values_1 == values_2, "PyTorch random should be reproducible"
    
    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different random values."""
        from utils import set_seed
        
        set_seed(42)
        values_1 = torch.rand(10).tolist()
        
        set_seed(123)
        values_2 = torch.rand(10).tolist()
        
        assert values_1 != values_2, "Different seeds should produce different values"
    
    def test_set_seed_returns_config(self):
        """Test that set_seed returns seed configuration."""
        from utils import set_seed
        
        config = set_seed(42)
        
        assert isinstance(config, dict)
        assert config['seed'] == 42
        assert 'cudnn_deterministic' in config
        assert 'cudnn_benchmark' in config


class TestConfigLoading:
    """Tests for configuration loading."""
    
    def test_load_config_returns_dict(self):
        """Test that load_config returns a dictionary."""
        from utils import load_config
        
        config = load_config("config/defaults.yaml")
        assert isinstance(config, dict)
    
    def test_load_config_has_reproducibility(self):
        """Test that config has reproducibility section."""
        from utils import load_config
        
        config = load_config("config/defaults.yaml")
        assert 'reproducibility' in config
        assert 'seed' in config['reproducibility']
    
    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config."""
        from utils import load_config
        
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/config.yaml")
    
    def test_compute_config_hash(self):
        """Test config hash computation."""
        from utils import load_config, compute_config_hash
        
        config = load_config("config/defaults.yaml")
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        
        assert hash1 == hash2, "Same config should produce same hash"
        assert len(hash1) == 8, "Hash should be 8 characters"


class TestVerifyReproducibility:
    """Tests for reproducibility verification."""
    
    def test_verify_reproducibility_passes(self):
        """Test that verify_reproducibility returns True."""
        from utils import verify_reproducibility
        
        result = verify_reproducibility(seed=42)
        assert result is True


class TestRuntimeInfo:
    """Tests for runtime information collection."""
    
    def test_get_runtime_info_returns_dict(self):
        """Test that get_runtime_info returns a dictionary."""
        from utils import get_runtime_info
        
        info = get_runtime_info()
        assert isinstance(info, dict)
    
    def test_runtime_info_has_required_keys(self):
        """Test that runtime info has required keys."""
        from utils import get_runtime_info
        
        info = get_runtime_info()
        
        required_keys = [
            'torch_version',
            'cuda_available',
            'python_version',
            'timestamp'
        ]
        
        for key in required_keys:
            assert key in info, f"Missing required key: {key}"


class TestExperimentDirectory:
    """Tests for experiment directory creation."""
    
    def test_create_experiment_dir(self, tmp_path):
        """Test experiment directory creation."""
        from utils import create_experiment_dir
        
        exp_dir = create_experiment_dir(
            base_dir=str(tmp_path),
            experiment_name="test",
            experiment_type="debug"
        )
        
        assert exp_dir.exists()
        assert (exp_dir / "checkpoints").exists()
        assert (exp_dir / "logs").exists()
        assert (exp_dir / "artifacts").exists()