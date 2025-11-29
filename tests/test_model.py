"""
Integration tests for the full MobilePlantViT model.
"""

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMobilePlantViTModel:
    """Integration tests for the full model."""
    
    def test_model_forward_pass(self, sample_image_tensor, num_classes):
        """Test full model forward pass."""
        from blocks import MobilePlantViTModel
        
        model = MobilePlantViTModel(num_classes=num_classes)
        output = model(sample_image_tensor)
        
        assert output.shape == (1, num_classes), f"Expected (1, {num_classes}), got {output.shape}"
    
    def test_model_batch_forward(self, sample_batch_tensor, num_classes):
        """Test model with batch input."""
        from blocks import MobilePlantViTModel
        
        model = MobilePlantViTModel(num_classes=num_classes)
        output = model(sample_batch_tensor)
        
        assert output.shape == (4, num_classes), f"Expected (4, {num_classes}), got {output.shape}"
    
    def test_model_output_probabilities(self, sample_image_tensor, num_classes):
        """Test that model outputs valid probabilities."""
        from blocks import MobilePlantViTModel
        
        model = MobilePlantViTModel(num_classes=num_classes)
        output = model(sample_image_tensor)
        
        # Should sum to 1
        assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-5)
        
        # Should be non-negative
        assert (output >= 0).all()
    
    def test_model_gradient_flow(self, sample_image_tensor, num_classes):
        """Test that gradients flow through entire model."""
        from blocks import MobilePlantViTModel
        
        model = MobilePlantViTModel(num_classes=num_classes)
        sample_image_tensor.requires_grad = True
        
        output = model(sample_image_tensor)
        loss = output.sum()
        loss.backward()
        
        assert sample_image_tensor.grad is not None
    
    def test_model_device_transfer(self, sample_image_tensor, num_classes, device):
        """Test model works on available device."""
        from blocks import MobilePlantViTModel
        
        model = MobilePlantViTModel(num_classes=num_classes).to(device)
        input_tensor = sample_image_tensor.to(device)
        
        output = model(input_tensor)
        
        assert output.device == device


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_seed_reproducibility(self, sample_image_tensor, num_classes):
        """Test that setting seed produces reproducible results."""
        from blocks import MobilePlantViTModel
        from utils import set_seed
        
        # First run
        set_seed(42)
        model1 = MobilePlantViTModel(num_classes=num_classes)
        output1 = model1(sample_image_tensor)
        
        # Second run with same seed
        set_seed(42)
        model2 = MobilePlantViTModel(num_classes=num_classes)
        output2 = model2(sample_image_tensor)
        
        assert torch.allclose(output1, output2), "Same seed should produce same output"


class TestConfigLoading:
    """Tests for configuration system."""
    
    def test_load_config(self):
        """Test configuration loading."""
        from utils import load_config
        
        config = load_config("config/defaults.yaml")
        
        assert 'reproducibility' in config
        assert 'seed' in config['reproducibility']
    
    def test_config_has_required_keys(self):
        """Test configuration has all required keys."""
        from utils import load_config
        
        config = load_config("config/defaults.yaml")
        
        required_sections = ['reproducibility', 'dataset', 'dataloader', 'training', 'model']
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"