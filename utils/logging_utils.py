"""
Experiment tracking utilities for TensorBoard and Weights & Biases.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union

import torch
import numpy as np
import matplotlib.pyplot as plt

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")


class ExperimentLogger:
    """
    Unified experiment logger supporting TensorBoard and optionally W&B.
    
    Args:
        experiment_dir: Path to experiment directory
        config: Configuration dictionary
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B team/user name
        wandb_tags: List of tags for W&B run
    """
    
    def __init__(
        self,
        experiment_dir: Union[str, Path],
        config: Dict[str, Any],
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "mobileplant-vit",
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[list] = None
    ):
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Initialize TensorBoard
        self.tb_writer = None
        if self.use_tensorboard:
            tb_log_dir = self.experiment_dir / "logs" / "tensorboard"
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"✅ TensorBoard initialized: {tb_log_dir}")
            print(f"   Run: tensorboard --logdir={tb_log_dir.parent}")
        
        # Initialize W&B
        self.wandb_run = None
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("⚠️  W&B requested but not available. Skipping.")
            else:
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=config,
                    tags=wandb_tags,
                    dir=str(self.experiment_dir),
                    name=self.experiment_dir.name
                )
                print(f"✅ W&B initialized: {wandb.run.url}")
        
        # Track best metrics
        self.best_metrics = {
            'best_val_acc': 0.0,
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        # History for plotting
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log scalar values to TensorBoard and W&B.
        
        Args:
            scalars: Dictionary of metric names to values
            step: Current step (epoch or batch)
            prefix: Optional prefix for metric names
        """
        for name, value in scalars.items():
            full_name = f"{prefix}/{name}" if prefix else name
            
            # TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalar(full_name, value, step)
            
            # W&B
            if self.wandb_run:
                wandb.log({full_name: value}, step=step)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ) -> bool:
        """
        Log metrics for a complete epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            lr: Current learning rate
            
        Returns:
            True if this is a new best validation accuracy
        """
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
        
        # Log to trackers
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': lr
        }
        self.log_scalars(metrics, epoch)
        
        # Check for best model
        is_best = val_acc > self.best_metrics['best_val_acc']
        if is_best:
            self.best_metrics['best_val_acc'] = val_acc
            self.best_metrics['best_val_loss'] = val_loss
            self.best_metrics['best_epoch'] = epoch
        
        return is_best
    
    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int) -> None:
        """
        Log an image to TensorBoard and W&B.
        
        Args:
            tag: Name for the image
            image: Image tensor (C, H, W) or numpy array (H, W, C)
            step: Current step
        """
        if self.tb_writer:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1)
            self.tb_writer.add_image(tag, image, step)
        
        if self.wandb_run:
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
            wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        """
        Log a matplotlib figure to TensorBoard and W&B.
        
        Args:
            tag: Name for the figure
            figure: Matplotlib figure object
            step: Current step
        """
        if self.tb_writer:
            self.tb_writer.add_figure(tag, figure, step)
        
        if self.wandb_run:
            wandb.log({tag: wandb.Image(figure)}, step=step)
    
    def log_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: list,
        step: int,
        title: str = "Confusion Matrix"
    ) -> None:
        """
        Log a confusion matrix as a figure.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            step: Current step
            title: Title for the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xlabel='Predicted',
            ylabel='True',
            title=title
        )
        
        # Rotate labels if many classes
        if len(class_names) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        
        # Save to artifacts
        cm_path = self.experiment_dir / "artifacts" / f"confusion_matrix_epoch{step}.png"
        fig.savefig(cm_path, dpi=150, bbox_inches='tight')
        
        self.log_figure("confusion_matrix", fig, step)
        plt.close(fig)
        
        print(f"✅ Confusion matrix saved: {cm_path}")
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple) -> None:
        """
        Log model architecture graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (batch, channels, height, width)
        """
        if self.tb_writer:
            dummy_input = torch.randn(input_shape)
            device = next(model.parameters()).device
            dummy_input = dummy_input.to(device)
            
            try:
                self.tb_writer.add_graph(model, dummy_input)
                print("✅ Model graph logged to TensorBoard")
            except Exception as e:
                print(f"⚠️  Could not log model graph: {e}")
    
    def save_artifact(self, artifact_path: Union[str, Path], artifact_type: str = "file") -> None:
        """
        Save an artifact and log to W&B if enabled.
        
        Args:
            artifact_path: Path to the artifact file
            artifact_type: Type of artifact (file, model, dataset, etc.)
        """
        artifact_path = Path(artifact_path)
        
        if self.wandb_run and artifact_path.exists():
            artifact = wandb.Artifact(
                name=artifact_path.stem,
                type=artifact_type
            )
            artifact.add_file(str(artifact_path))
            wandb.log_artifact(artifact)
            print(f"✅ Artifact uploaded to W&B: {artifact_path.name}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> Path:
        """
        Save a model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_dir = self.experiment_dir / "checkpoints"
        latest_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ Best model saved: {best_path}")
            
            # Upload to W&B
            self.save_artifact(best_path, artifact_type="model")
        
        return latest_path
    
    def save_history(self) -> Path:
        """
        Save training history to JSON file.
        
        Returns:
            Path to saved history file
        """
        history_path = self.experiment_dir / "artifacts" / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"✅ Training history saved: {history_path}")
        return history_path
    
    def plot_training_curves(self, save: bool = True) -> plt.Figure:
        """
        Plot training and validation curves.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[2].plot(epochs, self.history['lr'], 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig_path = self.experiment_dir / "artifacts" / "training_curves.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"✅ Training curves saved: {fig_path}")
        
        return fig
    
    def finish(self) -> None:
        """
        Finalize logging and close all writers.
        """
        # Save final history
        self.save_history()
        
        # Plot and save training curves
        self.plot_training_curves()
        
        # Save final metrics summary
        summary = {
            'best_val_acc': self.best_metrics['best_val_acc'],
            'best_val_loss': self.best_metrics['best_val_loss'],
            'best_epoch': self.best_metrics['best_epoch'],
            'total_epochs': len(self.history['train_loss']),
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else None
        }
        
        summary_path = self.experiment_dir / "artifacts" / "final_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Close TensorBoard
        if self.tb_writer:
            self.tb_writer.close()
            print("✅ TensorBoard writer closed")
        
        # Close W&B
        if self.wandb_run:
            wandb.finish()
            print("✅ W&B run finished")
        
        print(f"\n{'='*60}")
        print("  EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        print(f"  Best Validation Accuracy: {self.best_metrics['best_val_acc']:.4f}")
        print(f"  Best Epoch: {self.best_metrics['best_epoch']}")
        print(f"  Experiment Directory: {self.experiment_dir}")
        print(f"{'='*60}\n")


# Quick test when run directly
if __name__ == "__main__":
    print("Testing logging utilities...\n")
    
    # Create a test experiment
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_experiment"
        test_dir.mkdir(parents=True)
        (test_dir / "logs").mkdir()
        (test_dir / "checkpoints").mkdir()
        (test_dir / "artifacts").mkdir()
        
        # Initialize logger
        logger = ExperimentLogger(
            experiment_dir=test_dir,
            config={'test': True},
            use_tensorboard=True,
            use_wandb=False
        )
        
        # Simulate training
        for epoch in range(3):
            is_best = logger.log_epoch(
                epoch=epoch,
                train_loss=1.0 - epoch * 0.2,
                train_acc=0.5 + epoch * 0.1,
                val_loss=1.1 - epoch * 0.2,
                val_acc=0.45 + epoch * 0.12,
                lr=0.001
            )
            print(f"Epoch {epoch}: is_best={is_best}")
        
        # Finish
        logger.finish()
    
    print("\n✅ Logging utilities test complete!")