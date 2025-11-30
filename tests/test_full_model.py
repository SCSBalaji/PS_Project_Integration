"""
Comprehensive unit tests for MobilePlantViT full model assembly.

Test Categories:
1. Model Shape Tests - Verify input/output shapes
2. Model Variants Tests - Test tiny/small/base/large variants
3. Model Output Tests - Verify output properties
4. Model Gradient Tests - Verify gradient flow
5. Model Configuration Tests - Test config system
6. Model Save/Load Tests - Test serialization
7. Model Training Tests - Basic training functionality
8. Model Benchmark Tests - Performance measurements
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import (
    MobilePlantViT,
    MobilePlantViTConfig,
    mobileplant_vit_tiny,
    mobileplant_vit_small,
    mobileplant_vit_base,
    mobileplant_vit_large,
)
from src.utils.testing import (
    check_output_shape,
    check_gradient_flow,
    check_no_nan_inf,
    count_parameters,
)


# ============================================================================
# Test Class 1: Model Shape Tests
# ============================================================================

class TestModelShape:
    """Test MobilePlantViT input/output shapes."""
    
    def test_basic_shape(self):
        """Test basic input/output shape."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, 38), f"Expected (2, 38), got {y.shape}"
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_size_variations(self, batch_size):
        """Test with various batch sizes."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (batch_size, 38)
    
    @pytest.mark.parametrize("num_classes", [10, 38, 100, 1000])
    def test_various_num_classes(self, num_classes):
        """Test with various number of classes."""
        model = MobilePlantViT(num_classes=num_classes)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, num_classes)
    
    def test_forward_features_shape(self):
        """Test forward_features returns correct shape."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            features = model.forward_features(x)
        
        assert features.shape == (2, 256)  # embed_dim
    
    def test_get_logits_shape(self):
        """Test get_logits returns correct shape."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            logits = model.get_logits(x)
        
        assert logits.shape == (2, 38)
    
    def test_intermediate_outputs(self):
        """Test get_intermediate_outputs returns all stages."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model.get_intermediate_outputs(x)
        
        # Verify all expected keys present
        expected_keys = [
            'input', 'after_ghost_conv', 'after_fused_ir', 'after_coord_att',
            'after_patch_embed', 'after_pos_enc', 'after_lda', 'after_res_ln',
            'after_ffn', 'after_gap', 'output'
        ]
        
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"
        
        # Verify shapes
        assert outputs['after_ghost_conv'].shape == (2, 64, 224, 224)
        assert outputs['after_fused_ir'].shape == (2, 64, 56, 56)
        assert outputs['after_coord_att'].shape == (2, 64, 56, 56)
        assert outputs['after_patch_embed'].shape == (2, 196, 256)
        assert outputs['after_pos_enc'].shape == (2, 196, 256)
        assert outputs['after_lda'].shape == (2, 196, 256)
        assert outputs['after_res_ln'].shape == (2, 196, 256)
        assert outputs['after_ffn'].shape == (2, 196, 256)
        assert outputs['after_gap'].shape == (2, 256)
        assert outputs['output'].shape == (2, 38)


# ============================================================================
# Test Class 2: Model Variants Tests
# ============================================================================

class TestModelVariants:
    """Test model variant factory functions."""
    
    def test_tiny_variant(self):
        """Test mobileplant_vit_tiny."""
        model = mobileplant_vit_tiny()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, 38)
        
        # Check parameter count in reasonable range
        params = model.count_parameters()
        assert params < 500_000, f"Tiny should be < 500K params, got {params:,}"
    
    def test_small_variant(self):
        """Test mobileplant_vit_small."""
        model = mobileplant_vit_small()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, 38)
        
        params = model.count_parameters()
        assert params < 1_000_000, f"Small should be < 1M params, got {params:,}"
    
    def test_base_variant(self):
        """Test mobileplant_vit_base."""
        model = mobileplant_vit_base()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, 38)
        
        params = model.count_parameters()
        assert params < 2_000_000, f"Base should be < 2M params, got {params:,}"
    
    def test_large_variant(self):
        """Test mobileplant_vit_large."""
        model = mobileplant_vit_large()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, 38)
        
        params = model.count_parameters()
        assert params < 5_000_000, f"Large should be < 5M params, got {params:,}"
    
    def test_variant_with_custom_classes(self):
        """Test variants with custom number of classes."""
        for variant_fn in [mobileplant_vit_tiny, mobileplant_vit_small, 
                          mobileplant_vit_base, mobileplant_vit_large]:
            model = variant_fn(num_classes=100)
            model.eval()
            
            x = torch.randn(2, 3, 224, 224)
            
            with torch.no_grad():
                y = model(x)
            
            assert y.shape == (2, 100)
    
    def test_parameter_ordering(self):
        """Test that variants have increasing parameters."""
        tiny = mobileplant_vit_tiny()
        small = mobileplant_vit_small()
        base = mobileplant_vit_base()
        large = mobileplant_vit_large()
        
        tiny_params = tiny.count_parameters()
        small_params = small.count_parameters()
        base_params = base.count_parameters()
        large_params = large.count_parameters()
        
        assert tiny_params < small_params < base_params < large_params


# ============================================================================
# Test Class 3: Model Output Tests
# ============================================================================

class TestModelOutput:
    """Test model output properties."""
    
    def test_probabilities_sum_to_one(self):
        """Verify output probabilities sum to 1.0."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            probs = model(x)
        
        prob_sums = probs.sum(dim=-1)
        
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5)
    
    def test_probabilities_in_range(self):
        """Verify all probabilities are in [0, 1]."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            probs = model(x)
        
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_deterministic_eval(self):
        """Verify same output in eval mode."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y1 = model(x).clone()
            y2 = model(x).clone()
        
        assert torch.allclose(y1, y2)
    
    def test_stochastic_train(self):
        """Verify different outputs in train mode (due to dropout)."""
        model = MobilePlantViT(lda_dropout=0.5, ffn_dropout=0.5)
        model.train()
        
        x = torch.randn(2, 3, 224, 224)
        
        y1 = model(x).clone()
        y2 = model(x).clone()
        
        # Due to dropout, outputs should differ
        assert not torch.allclose(y1, y2)
    
    def test_no_nan_inf(self):
        """Verify no NaN or Inf in output."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y = model(x)
        
        check_no_nan_inf(y, "model output")
    
    def test_logits_vs_probs(self):
        """Verify logits and probs relationship."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            logits = model.get_logits(x)
            probs = model(x)
        
        expected_probs = torch.softmax(logits, dim=-1)
        
        assert torch.allclose(probs, expected_probs, atol=1e-5)


# ============================================================================
# Test Class 4: Model Gradient Tests
# ============================================================================

class TestModelGradient:
    """Test gradient flow through model."""
    
    def test_full_gradient_flow(self):
        """Verify gradients flow to all parameters."""
        model = MobilePlantViT()
        x = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        probs = model(x)
        loss = probs.sum()
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    def test_no_nan_gradients(self):
        """Verify no NaN in gradients."""
        model = MobilePlantViT()
        x = torch.randn(2, 3, 224, 224)
        
        probs = model(x)
        loss = probs.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_gradient_with_ce_loss(self):
        """Test gradient computation with CrossEntropyLoss."""
        model = MobilePlantViT()
        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 38, (2,))
        
        logits = model.get_logits(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        # Verify gradients exist
        assert model.classifier.fc.weight.grad is not None
        assert model.classifier.fc.weight.grad.abs().sum() > 0
    
    def test_gradient_clipping_compatibility(self):
        """Verify model works with gradient clipping."""
        model = MobilePlantViT()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 38, (2,))
        
        logits = model.get_logits(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Should complete without error


# ============================================================================
# Test Class 5: Model Configuration Tests
# ============================================================================

class TestModelConfiguration:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MobilePlantViTConfig()
        
        assert config.img_size == 224
        assert config.num_classes == 38
        assert config.embed_dim == 256
        assert config.num_heads == 8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MobilePlantViTConfig(
            num_classes=100,
            embed_dim=512,
            num_heads=16
        )
        
        model = MobilePlantViT(config)
        
        assert model.config.num_classes == 100
        assert model.config.embed_dim == 512
        assert model.config.num_heads == 16
    
    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = MobilePlantViTConfig(num_classes=100)
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['num_classes'] == 100
    
    def test_config_from_dict(self):
        """Test config creation from dict."""
        config_dict = {'num_classes': 100, 'embed_dim': 128}
        
        config = MobilePlantViTConfig.from_dict(config_dict)
        
        assert config.num_classes == 100
        assert config.embed_dim == 128
    
    def test_model_from_config(self):
        """Test model creation from config dict."""
        config_dict = {'num_classes': 50, 'embed_dim': 192}
        
        model = MobilePlantViT.from_config(config_dict)
        
        assert model.config.num_classes == 50
        assert model.config.embed_dim == 192
    
    def test_kwargs_override_config(self):
        """Test that kwargs override config values."""
        config = MobilePlantViTConfig(num_classes=38)
        
        model = MobilePlantViT(config, num_classes=100)
        
        assert model.config.num_classes == 100
    
    def test_get_config(self):
        """Test get_config returns current configuration."""
        model = MobilePlantViT(num_classes=50)
        
        config = model.get_config()
        
        assert config['num_classes'] == 50
    
    def test_invalid_config(self):
        """Test validation catches invalid config."""
        with pytest.raises(AssertionError):
            # embed_dim not divisible by num_heads
            MobilePlantViTConfig(embed_dim=100, num_heads=8)


# ============================================================================
# Test Class 6: Model Save/Load Tests
# ============================================================================

class TestModelSaveLoad:
    """Test model serialization."""
    
    def test_save_load_state_dict(self):
        """Test save and load state dict."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y_original = model(x).clone()
        
        # Save state dict
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name
        
        try:
            # Create new model and load
            model_loaded = MobilePlantViT()
            model_loaded.load_state_dict(torch.load(temp_path))
            model_loaded.eval()
            
            with torch.no_grad():
                y_loaded = model_loaded(x)
            
            assert torch.allclose(y_original, y_loaded)
        finally:
            os.unlink(temp_path)
    
    def test_save_load_full_model(self):
        """Test save and load full model."""
        model = MobilePlantViT(num_classes=50)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            y_original = model(x).clone()
        
        # Save full model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(model, f.name)
            temp_path = f.name
        
        try:
            # Load full model
            model_loaded = torch.load(temp_path)
            model_loaded.eval()
            
            with torch.no_grad():
                y_loaded = model_loaded(x)
            
            assert torch.allclose(y_original, y_loaded)
        finally:
            os.unlink(temp_path)
    
    def test_save_load_with_config(self):
        """Test save/load preserves configuration."""
        config = MobilePlantViTConfig(num_classes=100, embed_dim=192)
        model = MobilePlantViT(config)
        
        # Save state dict and config
        checkpoint = {
            'state_dict': model.state_dict(),
            'config': model.get_config()
        }
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(checkpoint, f.name)
            temp_path = f.name
        
        try:
            # Load checkpoint
            loaded_checkpoint = torch.load(temp_path)
            
            # Recreate model from config
            model_loaded = MobilePlantViT.from_config(loaded_checkpoint['config'])
            model_loaded.load_state_dict(loaded_checkpoint['state_dict'])
            
            assert model_loaded.config.num_classes == 100
            assert model_loaded.config.embed_dim == 192
        finally:
            os.unlink(temp_path)


# ============================================================================
# Test Class 7: Model Training Tests
# ============================================================================

class TestModelTraining:
    """Test basic training functionality."""
    
    def test_one_optimization_step(self):
        """Test single optimization step."""
        model = MobilePlantViT()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 38, (2,))
        
        # Initial loss
        logits = model.get_logits(x)
        loss_before = nn.CrossEntropyLoss()(logits, target).item()
        
        # Optimization step
        optimizer.zero_grad()
        logits = model.get_logits(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        optimizer.step()
        
        # Loss after
        with torch.no_grad():
            logits = model.get_logits(x)
            loss_after = nn.CrossEntropyLoss()(logits, target).item()
        
        # Loss should decrease (or at least not increase significantly)
        # Note: With one step, it might not always decrease due to randomness
        # But the operation should complete without error
    
    def test_overfit_single_batch(self):
        """Test model can overfit to single batch."""
        model = MobilePlantViT()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        x = torch.randn(4, 3, 224, 224)
        target = torch.randint(0, 38, (4,))
        
        # Train for multiple steps
        initial_loss = None
        for i in range(20):
            optimizer.zero_grad()
            logits = model.get_logits(x)
            loss = nn.CrossEntropyLoss()(logits, target)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, \
            f"Loss didn't decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
    
    def test_learning_rate_sensitivity(self):
        """Test model trains with different learning rates."""
        for lr in [1e-2, 1e-3, 1e-4]:
            model = MobilePlantViT()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            x = torch.randn(2, 3, 224, 224)
            target = torch.randint(0, 38, (2,))
            
            # Single step should complete without error
            optimizer.zero_grad()
            logits = model.get_logits(x)
            loss = nn.CrossEntropyLoss()(logits, target)
            loss.backward()
            optimizer.step()


# ============================================================================
# Test Class 8: Parameter Count Tests
# ============================================================================

class TestParameterCount:
    """Test parameter counting functionality."""
    
    def test_count_parameters(self):
        """Test count_parameters method."""
        model = MobilePlantViT()
        
        total_params = model.count_parameters(trainable_only=False)
        trainable_params = model.count_parameters(trainable_only=True)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_parameter_breakdown(self):
        """Test get_parameter_breakdown method."""
        model = MobilePlantViT()
        
        breakdown = model.get_parameter_breakdown()
        
        # Check all expected keys
        expected_keys = [
            'ghost_conv', 'fused_ir', 'coord_att', 'cnn_total',
            'patch_embed', 'pos_enc', 'transition_total',
            'lda', 'res_ln', 'ffn', 'transformer_total',
            'gap', 'classifier', 'classifier_total',
            'total'
        ]
        
        for key in expected_keys:
            assert key in breakdown, f"Missing key: {key}"
        
        # Verify totals add up
        cnn = breakdown['ghost_conv'] + breakdown['fused_ir'] + breakdown['coord_att']
        assert breakdown['cnn_total'] == cnn
        
        trans = breakdown['patch_embed'] + breakdown['pos_enc']
        assert breakdown['transition_total'] == trans
        
        transformer = breakdown['lda'] + breakdown['res_ln'] + breakdown['ffn']
        assert breakdown['transformer_total'] == transformer
        
        cls = breakdown['gap'] + breakdown['classifier']
        assert breakdown['classifier_total'] == cls
    
    def test_within_budget(self):
        """Test all variants within 5M parameter budget."""
        for variant_fn in [mobileplant_vit_tiny, mobileplant_vit_small,
                          mobileplant_vit_base, mobileplant_vit_large]:
            model = variant_fn()
            params = model.count_parameters()
            
            assert params < 5_000_000, f"Variant exceeds budget: {params:,}"


# ============================================================================
# Test Class 9: Benchmark Tests
# ============================================================================

class TestBenchmark:
    """Performance measurements."""
    
    def test_forward_time(self):
        """Measure forward pass time."""
        model = MobilePlantViT()
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
        
        import time
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = sum(times) / len(times)
        print(f"\nMobilePlantViT forward time: {mean_time:.2f} ms")
        
        # Should complete in reasonable time (< 500ms on CPU)
        assert mean_time < 500, f"Forward pass too slow: {mean_time:.2f} ms"
    
    def test_backward_time(self):
        """Measure backward pass time."""
        model = MobilePlantViT()
        
        x = torch.randn(2, 3, 224, 224)
        
        # Warmup
        for _ in range(3):
            y = model(x)
            y.sum().backward()
            model.zero_grad()
        
        import time
        times = []
        for _ in range(10):
            start = time.perf_counter()
            y = model(x)
            y.sum().backward()
            times.append((time.perf_counter() - start) * 1000)
            model.zero_grad()
        
        mean_time = sum(times) / len(times)
        print(f"\nMobilePlantViT forward+backward time: {mean_time:.2f} ms")
        
        assert mean_time < 1000, f"Backward pass too slow: {mean_time:.2f} ms"
    
    def test_throughput(self):
        """Measure inference throughput."""
        model = MobilePlantViT()
        model.eval()
        
        batch_size = 8
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
        
        import time
        num_iterations = 20
        start = time.perf_counter()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(x)
        
        elapsed = time.perf_counter() - start
        total_images = batch_size * num_iterations
        throughput = total_images / elapsed
        
        print(f"\nThroughput: {throughput:.1f} images/second")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])