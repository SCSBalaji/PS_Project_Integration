"""
Entrypoint initialization helper.

This module provides a single function that should be called at the start
of every training, evaluation, or inference script to ensure proper
reproducibility and experiment tracking.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.repro import (
    load_config,
    set_seed,
    save_config,
    get_device,
    get_runtime_info,
    compute_config_hash,
    create_experiment_dir,
    verify_reproducibility
)
from utils.logging_utils import ExperimentLogger
from utils.experiment import ExperimentManager


def init_training_run(
    config_path: str = "config/defaults.yaml",
    experiment_name: Optional[str] = None,
    experiment_type: str = "baseline",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "mobileplant-vit",
    verify_repro: bool = True
) -> Tuple[Dict[str, Any], Path, ExperimentLogger]:
    """
    Initialize a complete training run with all necessary components.
    
    This is the main entrypoint function that should be called at the start
    of every training script. It handles:
    
    1. Configuration loading
    2. Seed setting for reproducibility
    3. Experiment directory creation
    4. Logger initialization (TensorBoard/W&B)
    5. Device selection
    6. Config snapshot saving
    
    Args:
        config_path: Path to configuration YAML file
        experiment_name: Name for the experiment (used in directory naming)
        experiment_type: Type of experiment (baseline, ablation, hyperparam, debug, final)
        use_tensorboard: Whether to enable TensorBoard logging
        use_wandb: Whether to enable Weights & Biases logging
        wandb_project: W&B project name (if use_wandb=True)
        verify_repro: Whether to run reproducibility verification
        
    Returns:
        Tuple of:
        - config: Complete configuration dictionary with runtime metadata
        - experiment_dir: Path to the experiment directory
        - logger: ExperimentLogger instance for tracking
        
    Example:
        >>> config, exp_dir, logger = init_training_run(
        ...     experiment_name="plantvillage_baseline",
        ...     experiment_type="baseline"
        ... )
        >>> # ... training code ...
        >>> logger.finish()
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + "  INITIALIZING TRAINING RUN".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Step 1: Load configuration
    print("Step 1/6: Loading configuration...")
    config = load_config(config_path)
    
    # Step 2: Set seeds for reproducibility
    print("\nStep 2/6: Setting random seeds...")
    repro_config = config.get('reproducibility', {})
    seed_config = set_seed(
        seed=repro_config.get('seed', 42),
        cudnn_deterministic=repro_config.get('cudnn_deterministic', True),
        cudnn_benchmark=repro_config.get('cudnn_benchmark', False)
    )
    
    # Optional: Verify reproducibility
    if verify_repro:
        print("\nStep 2b/6: Verifying reproducibility...")
        if not verify_reproducibility(repro_config.get('seed', 42)):
            print("⚠️  Reproducibility verification failed! Results may vary between runs.")
    
    # Step 3: Create experiment directory
    print("\nStep 3/6: Creating experiment directory...")
    exp_manager = ExperimentManager(
        base_dir=config.get('experiment', {}).get('output_dir', 'experiments'),
        experiment_type=experiment_type,
        description=experiment_name or config.get('experiment', {}).get('name', 'run'),
        config=None  # Don't save config yet
    )
    experiment_dir = exp_manager.experiment_dir
    
    # Step 4: Get device
    print("\nStep 4/6: Detecting compute device...")
    device = get_device()
    
    # Step 5: Add runtime metadata
    print("\nStep 5/6: Collecting runtime metadata...")
    runtime_info = get_runtime_info()
    runtime_info['experiment_dir'] = str(experiment_dir)
    runtime_info['experiment_type'] = experiment_type
    runtime_info['experiment_name'] = experiment_name
    runtime_info['config_hash'] = compute_config_hash(config)
    runtime_info['device_used'] = str(device)
    runtime_info['seeds_used'] = seed_config
    
    config['_runtime'] = runtime_info
    
    # Save config snapshot
    config_save_path = experiment_dir / "config_used.yaml"
    save_config(config, config_save_path)
    
    # Create README for experiment
    exp_manager.create_readme(notes=f"Experiment type: {experiment_type}")
    
    # Step 6: Initialize logger
    print("\nStep 6/6: Initializing experiment logger...")
    logger = ExperimentLogger(
        experiment_dir=experiment_dir,
        config=config,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project
    )
    
    # Print summary
    print("\n" + "─" * 80)
    print("  INITIALIZATION COMPLETE")
    print("─" * 80)
    print(f"  Experiment: {exp_manager.experiment_name}")
    print(f"  Type: {experiment_type}")
    print(f"  Directory: {experiment_dir}")
    print(f"  Config hash: {runtime_info['config_hash']}")
    print(f"  Device: {device}")
    print(f"  Seed: {repro_config.get('seed', 42)}")
    print(f"  TensorBoard: {'Enabled' if use_tensorboard else 'Disabled'}")
    print(f"  W&B: {'Enabled' if use_wandb else 'Disabled'}")
    print("─" * 80 + "\n")
    
    return config, experiment_dir, logger


def init_evaluation_run(
    config_path: str = "config/defaults.yaml",
    checkpoint_path: Optional[str] = None,
    experiment_dir: Optional[str] = None
) -> Tuple[Dict[str, Any], Path]:
    """
    Initialize an evaluation run.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint to evaluate
        experiment_dir: Directory to save evaluation results
        
    Returns:
        Tuple of (config, output_dir)
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + "  INITIALIZING EVALUATION RUN".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Load configuration
    config = load_config(config_path)
    
    # Set seeds
    repro_config = config.get('reproducibility', {})
    set_seed(
        seed=repro_config.get('seed', 42),
        cudnn_deterministic=repro_config.get('cudnn_deterministic', True),
        cudnn_benchmark=repro_config.get('cudnn_benchmark', False)
    )
    
    # Create or use experiment directory
    if experiment_dir:
        output_dir = Path(experiment_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_experiment_dir(
            base_dir=config.get('experiment', {}).get('output_dir', 'experiments'),
            experiment_name="evaluation",
            experiment_type="eval"
        )
    
    # Add runtime info
    config['_runtime'] = get_runtime_info()
    config['_runtime']['checkpoint_path'] = checkpoint_path
    config['_runtime']['output_dir'] = str(output_dir)
    
    # Save config
    save_config(config, output_dir / "eval_config_used.yaml")
    
    print(f"\n✅ Evaluation run initialized")
    print(f"   Output directory: {output_dir}")
    if checkpoint_path:
        print(f"   Checkpoint: {checkpoint_path}")
    
    return config, output_dir


# Quick test
if __name__ == "__main__":
    print("Testing init_run utilities...\n")
    
    # Test training run initialization
    config, exp_dir, logger = init_training_run(
        experiment_name="test_init",
        experiment_type="debug",
        use_tensorboard=True,
        use_wandb=False,
        verify_repro=True
    )
    
    # Simulate some logging
    logger.log_epoch(
        epoch=0,
        train_loss=1.5,
        train_acc=0.3,
        val_loss=1.6,
        val_acc=0.28,
        lr=0.001
    )
    
    # Finish
    logger.finish()
    
    print("\n✅ init_run test complete!")
    print(f"   Test experiment created at: {exp_dir}")