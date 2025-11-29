"""
Experiment management utilities for consistent naming and artifact handling.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import yaml


class ExperimentManager:
    """
    Manages experiment directories, naming, and artifacts.
    
    Args:
        base_dir: Base directory for all experiments
        experiment_type: Type of experiment (baseline, ablation, hyperparam, debug, final)
        description: Short description of the experiment
        config: Configuration dictionary
    """
    
    VALID_TYPES = ['baseline', 'ablation', 'hyperparam', 'debug', 'final', 'test']
    
    def __init__(
        self,
        base_dir: str = "experiments",
        experiment_type: str = "baseline",
        description: str = "run",
        config: Optional[Dict[str, Any]] = None
    ):
        if experiment_type not in self.VALID_TYPES:
            print(f"⚠️  Unknown experiment type '{experiment_type}'. Valid types: {self.VALID_TYPES}")
        
        self.base_dir = Path(base_dir)
        self.experiment_type = experiment_type
        self.description = self._sanitize_name(description)
        self.config = config or {}
        
        # Generate experiment name
        self.timestamp = datetime.now().strftime("%Y%m%d")
        self.experiment_name = f"{self.timestamp}_{experiment_type}_{self.description}"
        self.experiment_dir = self.base_dir / self.experiment_name
        
        # Create directory structure
        self._create_directories()
        
        # Save initial config
        if config:
            self.save_config(config)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize experiment name for filesystem compatibility."""
        # Replace spaces and special chars with underscores
        sanitized = name.lower().replace(" ", "_").replace("-", "_")
        # Remove any non-alphanumeric chars except underscore
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
        return sanitized[:50]  # Limit length
    
    def _create_directories(self) -> None:
        """Create the experiment directory structure."""
        subdirs = ['checkpoints', 'logs/tensorboard', 'artifacts']
        
        for subdir in subdirs:
            (self.experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Experiment directory created: {self.experiment_dir}")
    
    def save_config(self, config: Dict[str, Any]) -> Path:
        """Save configuration to experiment directory."""
        config_path = self.experiment_dir / "config_used.yaml"
        
        # Add metadata
        config_with_meta = config.copy()
        config_with_meta['_experiment'] = {
            'name': self.experiment_name,
            'type': self.experiment_type,
            'description': self.description,
            'created_at': datetime.now().isoformat(),
            'directory': str(self.experiment_dir)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_with_meta, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Config saved: {config_path}")
        return config_path
    
    def save_artifact(self, name: str, data: Any, artifact_type: str = "json") -> Path:
        """
        Save an artifact to the artifacts directory.
        
        Args:
            name: Artifact filename (without extension)
            data: Data to save
            artifact_type: Type of artifact (json, yaml, text)
            
        Returns:
            Path to saved artifact
        """
        artifacts_dir = self.experiment_dir / "artifacts"
        
        if artifact_type == "json":
            path = artifacts_dir / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=4, default=str)
        elif artifact_type == "yaml":
            path = artifacts_dir / f"{name}.yaml"
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif artifact_type == "text":
            path = artifacts_dir / f"{name}.txt"
            with open(path, 'w') as f:
                f.write(str(data))
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        print(f"✅ Artifact saved: {path}")
        return path
    
    def get_checkpoint_path(self, name: str = "best_model") -> Path:
        """Get path for a checkpoint file."""
        return self.experiment_dir / "checkpoints" / f"{name}.pth"
    
    def get_tensorboard_dir(self) -> Path:
        """Get TensorBoard log directory."""
        return self.experiment_dir / "logs" / "tensorboard"
    
    def get_artifact_path(self, name: str) -> Path:
        """Get path for an artifact file."""
        return self.experiment_dir / "artifacts" / name
    
    def create_readme(self, notes: str = "") -> Path:
        """Create a README file for the experiment."""
        readme_path = self.experiment_dir / "README.md"
        
        content = f"""# Experiment: {self.experiment_name}

## Type
{self.experiment_type}

## Description
{self.description}

## Created
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Notes
{notes if notes else "No additional notes."}

## Files
- `config_used.yaml`: Configuration used for this run
- `checkpoints/`: Model checkpoints
- `logs/tensorboard/`: TensorBoard logs
- `artifacts/`: Training artifacts (plots, metrics, etc.)

## How to Reproduce

```bash
# Load config and run training
python train.py --config {self.experiment_dir}/config_used.yaml
```
"""
        
        with open(readme_path, 'w') as f:
            f.write(content)
        
        print(f"✅ README created: {readme_path}")
        return readme_path
    
    def finalize(self, final_metrics: Dict[str, Any]) -> None:
        """Finalize the experiment with final metrics."""
        # Save final summary
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_type': self.experiment_type,
            'completed_at': datetime.now().isoformat(),
            'metrics': final_metrics
        }
        self.save_artifact("final_summary", summary, "json")
        
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT FINALIZED: {self.experiment_name}")
        print(f"{'='*60}")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def list_experiments(base_dir: str = "experiments") -> list:
        """List all experiments in the base directory."""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        experiments = []
        for exp_dir in sorted(base_path.iterdir()):
            if exp_dir.is_dir() and (exp_dir / "config_used.yaml").exists():
                experiments.append(exp_dir.name)
        
        return experiments
    
    @staticmethod
    def load_experiment(experiment_path: str) -> Dict[str, Any]:
        """Load an existing experiment's config and summary."""
        exp_path = Path(experiment_path)
        
        result = {'path': str(exp_path)}
        
        # Load config
        config_path = exp_path / "config_used.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                result['config'] = yaml.safe_load(f)
        
        # Load summary
        summary_path = exp_path / "artifacts" / "final_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                result['summary'] = json.load(f)
        
        return result


def create_experiment(
    experiment_type: str = "baseline",
    description: str = "run",
    config_path: str = "config/defaults.yaml"
) -> ExperimentManager:
    """
    Convenience function to create a new experiment.
    
    Args:
        experiment_type: Type of experiment
        description: Short description
        config_path: Path to config file
        
    Returns:
        ExperimentManager instance
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment
    exp = ExperimentManager(
        base_dir=config.get('experiment', {}).get('output_dir', 'experiments'),
        experiment_type=experiment_type,
        description=description,
        config=config
    )
    
    # Create README
    exp.create_readme()
    
    return exp


# Quick test
if __name__ == "__main__":
    print("Testing ExperimentManager...\n")
    
    # Create a test experiment
    exp = create_experiment(
        experiment_type="debug",
        description="test run",
        config_path="config/defaults.yaml"
    )
    
    # Save some artifacts
    exp.save_artifact("test_metrics", {"accuracy": 0.95, "loss": 0.1})
    
    # Finalize
    exp.finalize({"best_accuracy": 0.95, "total_epochs": 10})
    
    # List experiments
    print("\nAll experiments:")
    for name in ExperimentManager.list_experiments():
        print(f"  - {name}")
    
    print("\n✅ ExperimentManager test complete!")