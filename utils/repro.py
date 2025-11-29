"""
Reproducibility utilities for ensuring consistent experiment results.
"""

import os
import random
import yaml
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch


def set_seed(
    seed: int = 42,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
    warn_only: bool = False
) -> Dict[str, Any]:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        cudnn_deterministic: If True, makes CuDNN deterministic (slower but reproducible)
        cudnn_benchmark: If True, enables CuDNN auto-tuner (faster but non-deterministic)
        warn_only: If True, only warn about non-deterministic settings instead of enforcing
        
    Returns:
        Dictionary containing the seed configuration used
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
    
    # PyTorch deterministic algorithms (PyTorch 1.8+)
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(cudnn_deterministic)
        except RuntimeError as e:
            if warn_only:
                print(f"⚠️  Could not enable deterministic algorithms: {e}")
            else:
                raise
    
    # Build seed config for logging
    seed_config = {
        'seed': seed,
        'python_seed': seed,
        'numpy_seed': seed,
        'torch_seed': seed,
        'cuda_seed': seed if torch.cuda.is_available() else None,
        'cudnn_deterministic': cudnn_deterministic,
        'cudnn_benchmark': cudnn_benchmark,
        'pythonhashseed': str(seed)
    }
    
    print(f"✅ Seeds set to {seed}")
    print(f"   CuDNN deterministic: {cudnn_deterministic}")
    print(f"   CuDNN benchmark: {cudnn_benchmark}")
    
    return seed_config


def load_config(config_path: str = "config/defaults.yaml") -> Dict[str, Any]:
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


def save_config(config: Dict[str, Any], save_path: str) -> Path:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the configuration
        
    Returns:
        Path to saved configuration file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Configuration saved to: {save_path}")
    return save_path


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a hash of the configuration for quick comparison.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MD5 hash string of the configuration
    """
    # Remove runtime metadata for hash computation
    config_copy = {k: v for k, v in config.items() if not k.startswith('_')}
    config_str = json.dumps(config_copy, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def create_experiment_dir(
    base_dir: str = "experiments",
    experiment_name: Optional[str] = None,
    experiment_type: str = "run"
) -> Path:
    """
    Create a unique experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional name for the experiment
        experiment_type: Type of experiment (baseline, ablation, debug, etc.)
        
    Returns:
        Path to the created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{timestamp}_{experiment_type}_{experiment_name}"
    else:
        dir_name = f"{timestamp}_{experiment_type}"
    
    experiment_dir = Path(base_dir) / dir_name
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'logs/tensorboard', 'artifacts']
    for subdir in subdirs:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Experiment directory created: {experiment_dir}")
    return experiment_dir


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using device: {device}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ Using device: {device} (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"✅ Using device: {device}")
        print("   ⚠️  No GPU available. Training will be slow.")
    
    return device


def get_runtime_info() -> Dict[str, Any]:
    """
    Collect runtime environment information.
    
    Returns:
        Dictionary containing runtime information
    """
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None,
        'numpy_version': np.__version__,
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'timestamp': datetime.now().isoformat()
    }
    return info


def initialize_run(
    config_path: str = "config/defaults.yaml",
    experiment_name: Optional[str] = None,
    experiment_type: str = "baseline",
    run_dir: Optional[str] = None
) -> Tuple[Dict[str, Any], Path]:
    """
    Initialize a training run with proper reproducibility settings.
    
    This function:
    1. Loads the configuration
    2. Sets all random seeds
    3. Creates experiment directory
    4. Saves a copy of the configuration used
    5. Initializes logging metadata
    
    Args:
        config_path: Path to the configuration file
        experiment_name: Optional name for the experiment
        experiment_type: Type of experiment (baseline, ablation, hyperparam, debug, final)
        run_dir: Optional specific directory for the run (overrides auto-generation)
        
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
    seed_config = set_seed(
        seed=repro_config.get('seed', 42),
        cudnn_deterministic=repro_config.get('cudnn_deterministic', True),
        cudnn_benchmark=repro_config.get('cudnn_benchmark', False)
    )
    
    # Create experiment directory
    if run_dir:
        experiment_dir = Path(run_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ['checkpoints', 'logs', 'logs/tensorboard', 'artifacts']:
            (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
    else:
        exp_config = config.get('experiment', {})
        base_dir = exp_config.get('output_dir', 'experiments')
        exp_name = experiment_name or exp_config.get('name', 'run')
        experiment_dir = create_experiment_dir(base_dir, exp_name, experiment_type)
    
    # Add runtime metadata to config
    config['_runtime'] = get_runtime_info()
    config['_runtime']['experiment_dir'] = str(experiment_dir)
    config['_runtime']['experiment_type'] = experiment_type
    config['_runtime']['config_hash'] = compute_config_hash(config)
    
    # Add seed config to runtime
    config['_runtime']['seeds_used'] = seed_config
    
    # Save configuration copy to experiment directory
    config_save_path = experiment_dir / "config_used.yaml"
    save_config(config, config_save_path)
    
    # Get device
    device = get_device()
    config['_runtime']['device_used'] = str(device)
    
    print(f"\n✅ Run initialized successfully!")
    print(f"   Experiment directory: {experiment_dir}")
    print(f"   Config hash: {config['_runtime']['config_hash']}")
    print("=" * 80 + "\n")
    
    return config, experiment_dir


def verify_reproducibility(seed: int = 42, num_checks: int = 5) -> bool:
    """
    Verify that seed setting produces reproducible results.
    
    Args:
        seed: Seed to test
        num_checks: Number of random numbers to generate for verification
        
    Returns:
        True if reproducibility is verified
    """
    print("\n🔍 Verifying reproducibility...")
    
    # First run
    set_seed(seed)
    python_randoms_1 = [random.random() for _ in range(num_checks)]
    numpy_randoms_1 = np.random.rand(num_checks).tolist()
    torch_randoms_1 = torch.rand(num_checks).tolist()
    
    # Second run with same seed
    set_seed(seed)
    python_randoms_2 = [random.random() for _ in range(num_checks)]
    numpy_randoms_2 = np.random.rand(num_checks).tolist()
    torch_randoms_2 = torch.rand(num_checks).tolist()
    
    # Verify
    python_match = python_randoms_1 == python_randoms_2
    numpy_match = numpy_randoms_1 == numpy_randoms_2
    torch_match = torch_randoms_1 == torch_randoms_2
    
    all_match = python_match and numpy_match and torch_match
    
    print(f"   Python random: {'✅' if python_match else '❌'}")
    print(f"   NumPy random:  {'✅' if numpy_match else '❌'}")
    print(f"   PyTorch random: {'✅' if torch_match else '❌'}")
    print(f"   Overall: {'✅ Reproducibility verified!' if all_match else '❌ Reproducibility FAILED!'}")
    
    return all_match


# Quick test when run directly
if __name__ == "__main__":
    print("Testing reproducibility utilities...\n")
    
    # Test seed verification
    verify_reproducibility(42)
    
    # Test config loading
    try:
        config = load_config()
        print(f"\nLoaded config with keys: {list(config.keys())}")
        print(f"Config hash: {compute_config_hash(config)}")
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("   Create config/defaults.yaml first!")
    
    # Test device detection
    device = get_device()
    
    # Test runtime info
    print("\nRuntime info:")
    info = get_runtime_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ All reproducibility utilities working!")