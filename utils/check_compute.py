"""
Utility script to check available compute resources.
"""

import os
import sys
import platform
import shutil
from pathlib import Path


def check_python():
    """Check Python version and environment."""
    print("=" * 60)
    print("  PYTHON ENVIRONMENT")
    print("=" * 60)
    print(f"  Python Version: {sys.version}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Executable: {sys.executable}")
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"  Virtual Environment: {'Yes' if in_venv else 'No'}")
    print()


def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("=" * 60)
    print("  PYTORCH & CUDA")
    print("=" * 60)
    
    try:
        import torch
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    - VRAM: {vram_gb:.1f} GB")
                print(f"    - Compute Capability: {props.major}.{props.minor}")
                print(f"    - Multi-Processors: {props.multi_processor_count}")
        else:
            print("  ⚠️  No CUDA GPU available. Training will use CPU (slower).")
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("  ✅ Apple MPS (Metal) is available for acceleration.")
    
    except ImportError:
        print("  ❌ PyTorch not installed!")
        print("     Install with: pip install torch torchvision")
    
    print()


def check_disk_space():
    """Check available disk space."""
    print("=" * 60)
    print("  DISK SPACE")
    print("=" * 60)
    
    # Get current directory disk usage
    current_path = Path.cwd()
    total, used, free = shutil.disk_usage(current_path)
    
    total_gb = total / (1024 ** 3)
    used_gb = used / (1024 ** 3)
    free_gb = free / (1024 ** 3)
    
    print(f"  Drive: {current_path.drive or '/'}")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Used: {used_gb:.1f} GB ({100 * used / total:.1f}%)")
    print(f"  Free: {free_gb:.1f} GB")
    
    if free_gb < 10:
        print("  ⚠️  Warning: Less than 10 GB free. Consider freeing up space.")
    else:
        print("  ✅ Sufficient disk space available.")
    
    print()


def check_memory():
    """Check system memory."""
    print("=" * 60)
    print("  SYSTEM MEMORY")
    print("=" * 60)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        used_percent = mem.percent
        
        print(f"  Total RAM: {total_gb:.1f} GB")
        print(f"  Available: {available_gb:.1f} GB")
        print(f"  Used: {used_percent:.1f}%")
        
        if available_gb < 4:
            print("  ⚠️  Warning: Low available memory.")
        else:
            print("  ✅ Sufficient memory available.")
    
    except ImportError:
        print("  ℹ️  Install psutil for memory info: pip install psutil")
    
    print()


def check_dependencies():
    """Check if required dependencies are installed."""
    print("=" * 60)
    print("  DEPENDENCIES")
    print("=" * 60)
    
    required = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('tensorboard', 'TensorBoard'),
        ('pytest', 'pytest'),
    ]
    
    optional = [
        ('wandb', 'Weights & Biases'),
        ('psutil', 'psutil'),
        ('seaborn', 'Seaborn'),
    ]
    
    print("\n  Required:")
    for module, name in required:
        try:
            __import__(module)
            print(f"    ✅ {name}")
        except ImportError:
            print(f"    ❌ {name} - NOT INSTALLED")
    
    print("\n  Optional:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"    ✅ {name}")
        except ImportError:
            print(f"    ⚪ {name} - not installed")
    
    print()


def check_project_structure():
    """Check if project structure is correct."""
    print("=" * 60)
    print("  PROJECT STRUCTURE")
    print("=" * 60)
    
    required_paths = [
        'config/defaults.yaml',
        'utils/__init__.py',
        'utils/repro.py',
        'utils/logging_utils.py',
        'utils/experiment.py',
        'blocks/__init__.py',
        'tests/__init__.py',
        'tests/conftest.py',
        'requirements.txt',
        '.gitignore',
        'CONTRIBUTING.md',
    ]
    
    all_present = True
    for path in required_paths:
        exists = Path(path).exists()
        status = "✅" if exists else "❌"
        print(f"    {status} {path}")
        if not exists:
            all_present = False
    
    print()
    if all_present:
        print("  ✅ All required files present!")
    else:
        print("  ⚠️  Some required files are missing.")
    
    print()


def estimate_training_time():
    """Estimate training time based on available hardware."""
    print("=" * 60)
    print("  TRAINING TIME ESTIMATES")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
            # Rough estimates based on GPU
            if 'a100' in gpu_name or 'v100' in gpu_name:
                time_per_epoch = "2-3 min"
                recommended_batch = 128
            elif '3090' in gpu_name or '4090' in gpu_name or '3080' in gpu_name:
                time_per_epoch = "2-4 min"
                recommended_batch = 64
            elif '3060' in gpu_name or '3070' in gpu_name or '2080' in gpu_name:
                time_per_epoch = "3-5 min"
                recommended_batch = 64
            elif 't4' in gpu_name:
                time_per_epoch = "5-8 min"
                recommended_batch = 32
            else:
                time_per_epoch = "5-10 min"
                recommended_batch = 32 if vram >= 8 else 16
            
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {vram:.1f} GB")
            print(f"  Estimated time per epoch: {time_per_epoch}")
            print(f"  Recommended batch size: {recommended_batch}")
            print(f"  Estimated total (10 epochs): {time_per_epoch.split('-')[0]}0-{time_per_epoch.split('-')[1].replace(' min', '')}0 min")
        else:
            print("  ⚠️  No GPU available. CPU training will be slow.")
            print("  Estimated time per epoch: 30-60 min")
            print("  Recommended batch size: 16")
            print("  Consider using Google Colab or Kaggle for GPU access.")
    
    except ImportError:
        print("  ❌ PyTorch not installed, cannot estimate.")
    
    print()


def run_all_checks():
    """Run all compute checks."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + "  COMPUTE RESOURCE CHECK".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    check_python()
    check_pytorch()
    check_memory()
    check_disk_space()
    check_dependencies()
    check_project_structure()
    estimate_training_time()
    
    print("=" * 60)
    print("  CHECK COMPLETE")
    print("=" * 60)
    print("\n  Run 'python utils/check_compute.py' anytime to recheck.\n")


if __name__ == "__main__":
    run_all_checks()