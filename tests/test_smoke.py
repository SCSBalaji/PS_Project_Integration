"""
Smoke tests for end-to-end pipeline validation.

These tests verify that the entire pipeline works correctly:
- Configuration loading
- Seed setting
- Model creation
- Forward/backward pass
- Checkpoint saving/loading
- Logging

These tests use small/dummy data and run quickly for CI.
"""

import pytest
import torch
import torch.nn as nn
import sys
import json
import tempfile
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEndToEndPipeline:
    """End-to-end smoke tests for the full training pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Create temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp(prefix="smoke_test_")
        yield
        # Cleanup after test
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_pipeline_forward_pass(self):
        """Test complete forward pass through all blocks."""
        from utils import set_seed, load_config
        from blocks import (
            GhostConv, FusedInvertedResidualBlock, CoordAtt,
            PatchEmbedding, PositionalEncoding, LinearDifferentialAttention,
            ResidualLayerNormBlock, BottleneckFFN, ClassifierHead, GlobalAveragePooling
        )
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Load config
        config = load_config("config/defaults.yaml")
        num_classes = config['dataset']['num_classes']
        
        # Create dummy input (batch_size=2, channels=3, height=224, width=224)
        x = torch.randn(2, 3, 224, 224)
        
        # Forward through each block
        # 1. GhostConv
        ghost = GhostConv(inp=3, oup=64)
        x = ghost(x)
        assert x.shape == (2, 64, 224, 224), f"GhostConv output shape mismatch: {x.shape}"
        
        # 2. FusedInvertedResidual
        fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, stride=1)
        x = fused_ir(x)
        assert x.shape == (2, 64, 224, 224), f"FusedIR output shape mismatch: {x.shape}"
        
        # 3. CoordAtt
        coord_att = CoordAtt(inp=64, oup=64)
        x = coord_att(x)
        assert x.shape == (2, 64, 224, 224), f"CoordAtt output shape mismatch: {x.shape}"
        
        # 4. PatchEmbedding
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=14)
        x = patch_embed(x)
        expected_seq_len = (224 // 14) ** 2  # 256
        assert x.shape == (2, expected_seq_len, 256), f"PatchEmbed output shape mismatch: {x.shape}"
        
        # 5. PositionalEncoding
        pos_enc = PositionalEncoding(embed_dim=256)
        x = pos_enc(x)
        assert x.shape == (2, expected_seq_len, 256), f"PosEnc output shape mismatch: {x.shape}"
        
        # 6. LinearDifferentialAttention
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = lda(x)
        assert x.shape == (2, expected_seq_len, 256), f"LDA output shape mismatch: {x.shape}"
        
        # 7. ResidualLayerNorm
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = res_ln(x)
        assert x.shape == (2, expected_seq_len, 256), f"ResLN output shape mismatch: {x.shape}"
        
        # 8. BottleneckFFN
        ffn = BottleneckFFN(inp=256, oup=256)
        x = ffn(x)
        assert x.shape == (2, expected_seq_len, 256), f"FFN output shape mismatch: {x.shape}"
        
        # 9. GlobalAveragePooling
        gap = GlobalAveragePooling()
        x = gap(x)
        assert x.shape == (2, 256), f"GAP output shape mismatch: {x.shape}"
        
        # 10. ClassifierHead
        classifier = ClassifierHead(embed_dim=256, num_classes=num_classes)
        x = classifier(x)
        assert x.shape == (2, num_classes), f"Classifier output shape mismatch: {x.shape}"
        
        # Verify output is valid probability distribution
        assert torch.allclose(x.sum(dim=1), torch.ones(2), atol=1e-5), "Output should sum to 1"
        assert (x >= 0).all(), "Probabilities should be non-negative"
        
        print("✅ Full pipeline forward pass successful!")
    
    def test_gradient_flow_through_pipeline(self):
        """Test that gradients flow through the entire pipeline."""
        from utils import set_seed
        from blocks import (
            GhostConv, FusedInvertedResidualBlock, CoordAtt,
            PatchEmbedding, PositionalEncoding, LinearDifferentialAttention,
            ResidualLayerNormBlock, BottleneckFFN, ClassifierHead, GlobalAveragePooling
        )
        
        set_seed(42)
        
        # Create input with gradient tracking
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        
        # Build mini-pipeline
        model = nn.Sequential(
            GhostConv(inp=3, oup=64),
            FusedInvertedResidualBlock(inp=64, oup=64),
            CoordAtt(inp=64, oup=64),
        )
        
        # Forward pass
        out = model(x)
        
        # Compute dummy loss and backward
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None, "Input should have gradients"
        assert not torch.isnan(x.grad).any(), "Gradients should not be NaN"
        assert not torch.isinf(x.grad).any(), "Gradients should not be Inf"
        
        print("✅ Gradient flow test successful!")
    
    def test_checkpoint_save_load(self):
        """Test model checkpoint saving and loading."""
        from utils import set_seed
        from blocks import GhostConv, FusedInvertedResidualBlock, CoordAtt
        
        set_seed(42)
        
        # Create a simple model
        model = nn.Sequential(
            GhostConv(inp=3, oup=64),
            FusedInvertedResidualBlock(inp=64, oup=64),
            CoordAtt(inp=64, oup=64),
        )
        
        # Create dummy input
        x = torch.randn(1, 3, 224, 224)
        
        # Get output before save
        output_before = model(x).detach().clone()
        
        # Save checkpoint
        checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 5,
            'loss': 0.5
        }, checkpoint_path)
        
        # Create new model and load checkpoint
        model_new = nn.Sequential(
            GhostConv(inp=3, oup=64),
            FusedInvertedResidualBlock(inp=64, oup=64),
            CoordAtt(inp=64, oup=64),
        )
        
        checkpoint = torch.load(checkpoint_path)
        model_new.load_state_dict(checkpoint['model_state_dict'])
        
        # Get output after load
        output_after = model_new(x).detach()
        
        # Verify outputs match
        assert torch.allclose(output_before, output_after, atol=1e-6), \
            "Model output should be identical after checkpoint load"
        
        assert checkpoint['epoch'] == 5
        assert checkpoint['loss'] == 0.5
        
        print("✅ Checkpoint save/load test successful!")
    
    def test_training_step_simulation(self):
        """Simulate a single training step."""
        from utils import set_seed
        from blocks import (
            GhostConv, FusedInvertedResidualBlock, CoordAtt,
            PatchEmbedding, LinearDifferentialAttention,
            BottleneckFFN, ClassifierHead, GlobalAveragePooling
        )
        
        set_seed(42)
        
        # Create mini model
        class MiniModel(nn.Module):
            def __init__(self, num_classes=38):
                super().__init__()
                self.ghost = GhostConv(inp=3, oup=64)
                self.fused = FusedInvertedResidualBlock(inp=64, oup=64)
                self.coord = CoordAtt(inp=64, oup=64)
                self.patch = PatchEmbedding(in_channels=64, embed_dim=128, patch_size=14)
                self.lda = LinearDifferentialAttention(embed_dim=128, num_heads=4)
                self.ffn = BottleneckFFN(inp=128, oup=128)
                self.gap = GlobalAveragePooling()
                self.classifier = ClassifierHead(embed_dim=128, num_classes=num_classes)
            
            def forward(self, x):
                x = self.ghost(x)
                x = self.fused(x)
                x = self.coord(x)
                x = self.patch(x)
                x = self.lda(x)
                x = self.ffn(x)
                x = self.gap(x)
                x = self.classifier(x)
                return x
        
        # Create model, optimizer, loss
        model = MiniModel(num_classes=38)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy batch
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 38, (batch_size,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Note: ClassifierHead already applies softmax, so we use NLLLoss or 
        # need to get logits. For simplicity, we'll compute loss differently
        # since CrossEntropyLoss expects logits, not probabilities
        log_probs = torch.log(outputs + 1e-8)  # Convert probabilities to log-probs
        loss = nn.NLLLoss()(log_probs, labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
        
        # Optimizer step
        optimizer.step()
        
        print(f"✅ Training step simulation successful! Loss: {loss.item():.4f}")
    
    def test_config_snapshot_saving(self):
        """Test that config is properly saved with experiments."""
        from utils import load_config, save_config, compute_config_hash
        import yaml
        
        # Load config
        config = load_config("config/defaults.yaml")
        original_hash = compute_config_hash(config)
        
        # Save to temp directory
        save_path = Path(self.temp_dir) / "config_used.yaml"
        save_config(config, save_path)
        
        # Load saved config
        with open(save_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Verify hash matches
        loaded_hash = compute_config_hash(loaded_config)
        assert original_hash == loaded_hash, "Config hash should match after save/load"
        
        print("✅ Config snapshot saving test successful!")
    
    def test_reproducibility_across_runs(self):
        """Test that same seed produces identical results."""
        from utils import set_seed
        from blocks import GhostConv, FusedInvertedResidualBlock
        
        def run_model(seed):
            set_seed(seed)
            model = nn.Sequential(
                GhostConv(inp=3, oup=64),
                FusedInvertedResidualBlock(inp=64, oup=64),
            )
            x = torch.randn(1, 3, 64, 64)  # Smaller for speed
            return model(x).detach().clone()
        
        # Run twice with same seed
        output1 = run_model(42)
        output2 = run_model(42)
        
        # Should be identical
        assert torch.allclose(output1, output2), "Same seed should produce identical results"
        
        # Run with different seed
        output3 = run_model(123)
        
        # Should be different
        assert not torch.allclose(output1, output3), "Different seed should produce different results"
        
        print("✅ Reproducibility test successful!")


class TestLoggingIntegration:
    """Tests for logging functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="logging_test_")
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_experiment_logger_creation(self):
        """Test ExperimentLogger can be created."""
        from utils import load_config
        from utils.logging_utils import ExperimentLogger
        
        config = load_config("config/defaults.yaml")
        
        logger = ExperimentLogger(
            experiment_dir=self.temp_dir,
            config=config,
            use_tensorboard=True,
            use_wandb=False
        )
        
        assert logger is not None
        
        # Log some metrics
        logger.log_epoch(
            epoch=0,
            train_loss=1.5,
            train_acc=0.3,
            val_loss=1.6,
            val_acc=0.28,
            lr=0.001
        )
        
        # Finish logging
        logger.finish()
        
        # Check TensorBoard directory was created
        tb_dir = Path(self.temp_dir) / "tensorboard"
        assert tb_dir.exists() or (Path(self.temp_dir) / "logs" / "tensorboard").exists()
        
        print("✅ Experiment logger test successful!")
    
    def test_training_history_saving(self):
        """Test that training history is saved correctly."""
        from utils import load_config
        from utils.logging_utils import ExperimentLogger
        
        config = load_config("config/defaults.yaml")
        
        logger = ExperimentLogger(
            experiment_dir=self.temp_dir,
            config=config,
            use_tensorboard=False,
            use_wandb=False
        )
        
        # Log multiple epochs
        for epoch in range(3):
            logger.log_epoch(
                epoch=epoch,
                train_loss=1.5 - epoch * 0.3,
                train_acc=0.3 + epoch * 0.2,
                val_loss=1.6 - epoch * 0.25,
                val_acc=0.28 + epoch * 0.18,
                lr=0.001
            )
        
        # Save history
        history_path = logger.save_training_history()
        
        # Load and verify
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        assert len(history['epochs']) == 3
        assert history['epochs'][-1]['train_acc'] == pytest.approx(0.7, rel=0.1)
        
        logger.finish()
        print("✅ Training history saving test successful!")


class TestExperimentManager:
    """Tests for experiment management."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="exp_manager_test_")
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_experiment_creation(self):
        """Test experiment directory creation."""
        from utils import load_config
        from utils.experiment import ExperimentManager
        
        config = load_config("config/defaults.yaml")
        
        exp = ExperimentManager(
            base_dir=self.temp_dir,
            experiment_type="debug",
            description="smoke_test",
            config=config
        )
        
        # Check directory structure
        assert exp.experiment_dir.exists()
        assert (exp.experiment_dir / "checkpoints").exists()
        assert (exp.experiment_dir / "artifacts").exists()
        assert (exp.experiment_dir / "config_used.yaml").exists()
        
        print("✅ Experiment manager test successful!")
    
    def test_artifact_saving(self):
        """Test artifact saving functionality."""
        from utils import load_config
        from utils.experiment import ExperimentManager
        
        config = load_config("config/defaults.yaml")
        
        exp = ExperimentManager(
            base_dir=self.temp_dir,
            experiment_type="debug",
            description="artifact_test",
            config=config
        )
        
        # Save various artifacts
        exp.save_artifact("metrics", {"accuracy": 0.95, "loss": 0.1}, "json")
        exp.save_artifact("notes", "This is a test run", "text")
        
        # Verify files exist
        assert (exp.experiment_dir / "artifacts" / "metrics.json").exists()
        assert (exp.experiment_dir / "artifacts" / "notes.txt").exists()
        
        print("✅ Artifact saving test successful!")


class TestQuickSanityCheck:
    """Quick sanity checks that should always pass."""
    
    def test_imports(self):
        """Test that all required imports work."""
        # Utils
        from utils import set_seed, load_config, save_config
        from utils import get_device, get_runtime_info
        from utils import ExperimentLogger, ExperimentManager
        
        # Blocks
        from blocks import GhostConv, FusedInvertedResidualBlock, CoordAtt
        from blocks import PatchEmbedding, PositionalEncoding
        from blocks import LinearDifferentialAttention, ResidualLayerNormBlock
        from blocks import BottleneckFFN, ClassifierHead, GlobalAveragePooling
        
        print("✅ All imports successful!")
    
    def test_config_exists(self):
        """Test that default config file exists."""
        config_path = Path("config/defaults.yaml")
        assert config_path.exists(), "Default config file should exist"
        print("✅ Config file exists!")
    
    def test_pytorch_available(self):
        """Test PyTorch is available and working."""
        import torch
        
        # Basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.matmul(x, y)
        
        assert z.shape == (2, 2)
        print(f"✅ PyTorch {torch.__version__} working!")
    
    def test_device_detection(self):
        """Test device detection works."""
        from utils import get_device
        
        device = get_device()
        assert device is not None
        print(f"✅ Device detected: {device}")