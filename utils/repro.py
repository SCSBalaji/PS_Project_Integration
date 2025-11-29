"""
Reproducibility utilities for ensuring consistent experiment results.
"""

import os
import random
import yaml
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, cudnn_deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        cudnn_deterministic: If True, makes CuDNN deterministic (slower but reproducible)
        cudnn_benchmark: If True, enables CuDNN auto-tuner (faster but non-deterministic)
    """
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # CuDNN settings
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ Seeds set to {seed}")
    print(f"   CuDNN deterministic: {cudnn_deterministic}")
    print(f"   CuDNN benchmark: {cudnn_benchmark}")


def load_config(config_path: str = "config/defaults.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Configuration loaded from: {config_path}")
    return config


def save_config(config: dict, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Configuration saved to: {save_path}")


def create_experiment_dir(base_dir: str = "experiments", experiment_name: str = None) -> Path:
    """
    Create a unique experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional name for the experiment
        
    Returns:
        Path to the created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{timestamp}_{experiment_name}"
    else:
        dir_name = timestamp
    
    experiment_dir = Path(base_dir) / dir_name
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Experiment directory created: {experiment_dir}")
    return experiment_dir


def initialize_run(config_path: str = "config/defaults.yaml", 
                   experiment_name: str = None) -> tuple:
    """
    Initialize a training run with proper reproducibility settings.
    
    This function:
    1. Loads the configuration
    2. Sets all random seeds
    3. Creates experiment directory
    4. Saves a copy of the configuration used
    
    Args:
        config_path: Path to the configuration file
        experiment_name: Optional name for the experiment
        
    Returns:
        Tuple of (config dict, experiment directory Path)
    """
    print("\n" + "=" * 80)
    print("  INITIALIZING EXPERIMENT RUN")
    print("=" * 80)
    
    # Load configuration
    config = load_config(config_path)
    
    # Set reproducibility seeds
    repro_config = config.get('reproducibility', {})
    set_seed(
        seed=repro_config.get('seed', 42),
        cudnn_deterministic=repro_config.get('cudnn_deterministic', True),
        cudnn_benchmark=repro_config.get('cudnn_benchmark', False)
    )
    
    # Create experiment directory
    exp_config = config.get('experiment', {})
    base_dir = exp_config.get('output_dir', 'experiments')
    exp_name = experiment_name or exp_config.get('name', 'run')
    
    experiment_dir = create_experiment_dir(base_dir, exp_name)
    
    # Add runtime metadata to config
    config['_runtime'] = {
        'experiment_dir': str(experiment_dir),
        'start_time': datetime.now().isoformat(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    }
    
    # Save configuration copy to experiment directory
    config_save_path = experiment_dir / "config_used.yaml"
    save_config(config, config_save_path)
    
    print(f"\n✅ Run initialized successfully!")
    print(f"   Experiment directory: {experiment_dir}")
    print("=" * 80 + "\n")
    
    return config, experiment_dir


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device object
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


# Quick test when run directly
if __name__ == "__main__":
    print("Testing reproducibility utilities...\n")
    
    # Test set_seed
    set_seed(42)
    
    # Test config loading (will fail if defaults.yaml doesn't exist)
    try:
        config = load_config()
        print(f"\nLoaded config with keys: {list(config.keys())}")
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("   Create config/defaults.yaml first!")
    
    # Test device detection
    device = get_device()
    
    print("\n✅ All reproducibility utilities working!")