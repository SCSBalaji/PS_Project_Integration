"""
Experiment logging utilities for tracking training progress.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch


class ExperimentLogger:
    """
    Unified experiment logger supporting TensorBoard and W&B.
    
    Args:
        experiment_dir: Directory to save logs and artifacts
        config: Configuration dictionary
        use_tensorboard: Whether to log to TensorBoard
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: W&B run name
    """
    
    def __init__(
        self,
        experiment_dir: str,
        config: Optional[Dict[str, Any]] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "mobileplant-vit",
        wandb_run_name: Optional[str] = None
    ):
        self.experiment_dir = Path(experiment_dir)
        self.config = config or {}
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Create necessary directories
        self.logs_dir = self.experiment_dir / "logs"
        self.tb_dir = self.logs_dir / "tensorboard"
        self.artifacts_dir = self.experiment_dir / "artifacts"
        
        # Ensure all directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history: Dict[str, List] = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Initialize TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            self._init_tensorboard()
        
        # Initialize W&B
        self.wandb_run = None
        if use_wandb:
            self._init_wandb(wandb_project, wandb_run_name)
        
        # Track best metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(self.tb_dir))
            print(f"✅ TensorBoard initialized: {self.tb_dir}")
            print(f"   Run: tensorboard --logdir={self.logs_dir}")
        except ImportError:
            print("⚠️  TensorBoard not available. Install with: pip install tensorboard")
            self.use_tensorboard = False
    
    def _init_wandb(self, project: str, run_name: Optional[str]):
        """Initialize Weights & Biases."""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=project,
                name=run_name or self.experiment_dir.name,
                config=self.config,
                dir=str(self.logs_dir),
                reinit=True
            )
            print(f"✅ W&B initialized: {project}/{self.wandb_run.name}")
        except ImportError:
            print("⚠️  W&B not available. Install with: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            self.use_wandb = False
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss (optional)
            val_acc: Validation accuracy (optional)
            lr: Learning rate (optional)
            extra_metrics: Additional metrics to log (optional)
        """
        # Store in history
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(lr)
        
        # Track best metrics
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        # Log to TensorBoard
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
            self.tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
            
            if val_loss is not None:
                self.tb_writer.add_scalar('Loss/val', val_loss, epoch)
            if val_acc is not None:
                self.tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
            if lr is not None:
                self.tb_writer.add_scalar('Learning_Rate', lr, epoch)
            
            if extra_metrics:
                for name, value in extra_metrics.items():
                    self.tb_writer.add_scalar(name, value, epoch)
        
        # Log to W&B
        if self.use_wandb and self.wandb_run:
            import wandb
            
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
            }
            
            if val_loss is not None:
                log_dict['val_loss'] = val_loss
            if val_acc is not None:
                log_dict['val_acc'] = val_acc
            if lr is not None:
                log_dict['learning_rate'] = lr
            if extra_metrics:
                log_dict.update(extra_metrics)
            
            wandb.log(log_dict)
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple = (1, 3, 224, 224)):
        """Log model graph to TensorBoard."""
        if self.use_tensorboard and self.tb_writer:
            try:
                dummy_input = torch.randn(input_shape)
                self.tb_writer.add_graph(model, dummy_input)
                print("✅ Model graph logged to TensorBoard")
            except Exception as e:
                print(f"⚠️  Could not log model graph: {e}")
    
    def log_images(self, tag: str, images: torch.Tensor, epoch: int):
        """Log images to TensorBoard."""
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_images(tag, images, epoch)
    
    def log_confusion_matrix(self, cm, class_names: List[str], epoch: int):
        """Log confusion matrix as image."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(12, 12))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            ax.set(
                xticks=np.arange(len(class_names)),
                yticks=np.arange(len(class_names)),
                xlabel='Predicted',
                ylabel='True',
                title=f'Confusion Matrix (Epoch {epoch})'
            )
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            fig.tight_layout()
            
            # Save to artifacts
            cm_path = self.artifacts_dir / f'confusion_matrix_epoch_{epoch}.png'
            fig.savefig(cm_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Log to TensorBoard
            if self.use_tensorboard and self.tb_writer:
                self.tb_writer.add_figure('Confusion_Matrix', fig, epoch)
            
            print(f"✅ Confusion matrix saved: {cm_path}")
            
        except ImportError:
            print("⚠️  Matplotlib not available for confusion matrix plotting")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters with associated metrics."""
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_hparams(hparams, metrics)
    
    def save_history(self) -> Path:
        """Save training history to JSON file."""
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        history_path = self.artifacts_dir / 'training_history.json'
        
        # Format history for saving
        formatted_history = {
            'epochs': [
                {
                    'epoch': e,
                    'train_loss': tl,
                    'train_acc': ta,
                    'val_loss': vl,
                    'val_acc': va,
                    'learning_rate': lr
                }
                for e, tl, ta, vl, va, lr in zip(
                    self.history['epochs'],
                    self.history['train_loss'],
                    self.history['train_acc'],
                    self.history['val_loss'],
                    self.history['val_acc'],
                    self.history['learning_rate']
                )
            ],
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.history['epochs'])
        }
        
        with open(history_path, 'w') as f:
            json.dump(formatted_history, f, indent=4)
        
        print(f"✅ Training history saved: {history_path}")
        return history_path
    
    def save_training_history(self) -> Path:
        """Alias for save_history() for backward compatibility."""
        return self.save_history()
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        is_best: bool = False,
        filename: str = "checkpoint.pth"
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            loss: Current loss
            is_best: Whether this is the best model so far
            filename: Checkpoint filename
            
        Returns:
            Path to saved checkpoint
        """
        checkpoints_dir = self.experiment_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = checkpoints_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ Best model saved: {best_path}")
        
        return checkpoint_path
    
    def finish(self):
        """Finalize logging and close writers."""
        # Save training history
        if self.history['epochs']:
            self.save_history()
        
        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()
            print("✅ TensorBoard writer closed")
        
        # Finish W&B run
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()
            print("✅ W&B run finished")
        
        print(f"\n📁 Experiment artifacts saved to: {self.experiment_dir}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training run."""
        return {
            'total_epochs': len(self.history['epochs']),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else None,
            'experiment_dir': str(self.experiment_dir)
        }


# Test when run directly
if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("Testing ExperimentLogger...\n")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="logger_test_")
    
    try:
        # Create logger
        logger = ExperimentLogger(
            experiment_dir=temp_dir,
            config={'test': True, 'seed': 42},
            use_tensorboard=True,
            use_wandb=False
        )
        
        # Log some epochs
        for epoch in range(5):
            logger.log_epoch(
                epoch=epoch,
                train_loss=1.0 - epoch * 0.15,
                train_acc=0.5 + epoch * 0.1,
                val_loss=1.1 - epoch * 0.12,
                val_acc=0.48 + epoch * 0.09,
                lr=0.001
            )
        
        # Get summary
        summary = logger.get_summary()
        print(f"\nTraining summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Finish
        logger.finish()
        
        print("\n✅ ExperimentLogger test complete!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)