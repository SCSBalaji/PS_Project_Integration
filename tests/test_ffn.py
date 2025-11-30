"""
Comprehensive unit tests for Bottleneck FFN and Residual LayerNorm.

Test Categories:
1. Shape Tests - Verify output shapes for various configurations
2. Bottleneck Tests - Verify bottleneck dimension calculations
3. Dropout Tests - Verify train/eval behavior
4. Residual Tests - Verify residual connection logic
5. Gradient Tests - Verify gradient flow
6. Stateless Tests - Verify LayerNorm stateless behavior
7. Benchmark Tests - Performance measurements
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.blocks import BottleneckFFN, ResidualLayerNormBlock
from src.utils.testing import (
    check_output_shape,
    check_gradient_flow,
    check_no_nan_inf,
    count_parameters,
)


# ============================================================================
# Test Class 1: BottleneckFFN Shape Tests
# ============================================================================

class TestBottleneckFFNShape:
    """Test output shapes for various configurations."""
    
    def test_basic_shape(self):
        """Test basic shape transformation."""
        ffn = BottleneckFFN(inp=256, oup=256)
        x = torch.randn(2, 196, 256)
        
        y = ffn(x)
        
        assert y.shape == (2, 196, 256)
    
    def test_different_inp_oup(self):
        """Test with different input and output dimensions."""
        ffn = BottleneckFFN(inp=256, oup=128)
        x = torch.randn(2, 196, 256)
        
        y = ffn(x)
        
        assert y.shape == (2, 196, 128)
    
    def test_increase_dim(self):
        """Test with output larger than input."""
        ffn = BottleneckFFN(inp=128, oup=256)
        x = torch.randn(2, 196, 128)
        
        y = ffn(x)
        
        assert y.shape == (2, 196, 256)
    
    @pytest.mark.parametrize("seq_len", [16, 49, 100, 196, 400])
    def test_various_seq_lengths(self, seq_len):
        """Test with various sequence lengths."""
        ffn = BottleneckFFN(inp=256, oup=256)
        x = torch.randn(2, seq_len, 256)
        
        y = ffn(x)
        
        assert y.shape == (2, seq_len, 256)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_size_variations(self, batch_size):
        """Test with various batch sizes."""
        ffn = BottleneckFFN(inp=256, oup=256)
        x = torch.randn(batch_size, 196, 256)
        
        y = ffn(x)
        
        assert y.shape == (batch_size, 196, 256)
    
    @pytest.mark.parametrize("embed_dim", [64, 128, 256, 512])
    def test_various_embed_dims(self, embed_dim):
        """Test with various embedding dimensions."""
        ffn = BottleneckFFN(inp=embed_dim, oup=embed_dim)
        x = torch.randn(2, 49, embed_dim)
        
        y = ffn(x)
        
        assert y.shape == (2, 49, embed_dim)


# ============================================================================
# Test Class 2: Bottleneck Ratio Tests
# ============================================================================

class TestBottleneckRatio:
    """Test bottleneck dimension calculations."""
    
    def test_ratio_025(self):
        """Test default ratio of 0.25."""
        ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.25)
        
        assert ffn.bottleneck_channels == 64
        assert ffn.fc1.out_features == 64
        assert ffn.fc2.in_features == 64
    
    def test_ratio_05(self):
        """Test ratio of 0.5."""
        ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.5)
        
        assert ffn.bottleneck_channels == 128
    
    def test_ratio_01(self):
        """Test small ratio of 0.1."""
        ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.1)
        
        assert ffn.bottleneck_channels == 25
    
    def test_ratio_10(self):
        """Test ratio of 1.0 (no compression)."""
        ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=1.0)
        
        assert ffn.bottleneck_channels == 256
    
    def test_minimum_bottleneck(self):
        """Test minimum bottleneck channels is 1."""
        ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.001)
        
        assert ffn.bottleneck_channels >= 1
    
    def test_very_small_input(self):
        """Test with very small input dimension."""
        ffn = BottleneckFFN(inp=8, oup=8, bottleneck_ratio=0.25)
        
        assert ffn.bottleneck_channels == 2
        
        x = torch.randn(2, 49, 8)
        y = ffn(x)
        assert y.shape == (2, 49, 8)
    
    def test_ratio_affects_params(self):
        """Test that smaller ratio means fewer parameters."""
        ffn_025 = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.25)
        ffn_05 = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.5)
        ffn_10 = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=1.0)
        
        params_025 = count_parameters(ffn_025)['total']
        params_05 = count_parameters(ffn_05)['total']
        params_10 = count_parameters(ffn_10)['total']
        
        assert params_025 < params_05 < params_10


# ============================================================================
# Test Class 3: Dropout Tests
# ============================================================================

class TestBottleneckFFNDropout:
    """Test dropout behavior in train/eval modes."""
    
    def test_dropout_train_mode(self):
        """Test that dropout is active in train mode."""
        ffn = BottleneckFFN(inp=256, oup=256, dropout=0.5)
        ffn.train()
        
        x = torch.randn(2, 196, 256)
        
        # Run multiple times - outputs should differ due to dropout
        outputs = []
        for _ in range(5):
            y = ffn(x)
            outputs.append(y.clone())
        
        # Check that not all outputs are identical
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same, "Outputs should differ in train mode due to dropout"
    
    def test_dropout_eval_mode(self):
        """Test that dropout is inactive in eval mode."""
        ffn = BottleneckFFN(inp=256, oup=256, dropout=0.5)
        ffn.eval()
        
        x = torch.randn(2, 196, 256)
        
        with torch.no_grad():
            y1 = ffn(x).clone()
            y2 = ffn(x).clone()
        
        assert torch.allclose(y1, y2), "Outputs should be identical in eval mode"
    
    def test_dropout_zero(self):
        """Test with dropout=0 (no dropout)."""
        ffn = BottleneckFFN(inp=256, oup=256, dropout=0.0)
        ffn.train()
        
        x = torch.randn(2, 196, 256)
        
        y1 = ffn(x).clone()
        y2 = ffn(x).clone()
        
        # With dropout=0, outputs should be identical even in train mode
        assert torch.allclose(y1, y2)
    
    @pytest.mark.parametrize("dropout_rate", [0.1, 0.3, 0.5])
    def test_various_dropout_rates(self, dropout_rate):
        """Test that various dropout rates work."""
        ffn = BottleneckFFN(inp=256, oup=256, dropout=dropout_rate)
        x = torch.randn(2, 49, 256)
        
        ffn.train()
        y_train = ffn(x)
        
        ffn.eval()
        with torch.no_grad():
            y_eval = ffn(x)
        
        # Both should have valid shapes
        assert y_train.shape == (2, 49, 256)
        assert y_eval.shape == (2, 49, 256)


# ============================================================================
# Test Class 4: ResidualLayerNormBlock Shape Tests
# ============================================================================

class TestResLNShape:
    """Test Residual LayerNorm shapes."""
    
    def test_basic_shape(self):
        """Test basic shape preservation."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 196, 256)
        
        y = res_ln(x)
        
        assert y.shape == (2, 196, 256)
    
    def test_with_explicit_residual(self):
        """Test with explicit residual tensor."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 196, 256)
        residual = torch.randn(2, 196, 256)
        
        y = res_ln(x, residual=residual)
        
        assert y.shape == (2, 196, 256)
    
    @pytest.mark.parametrize("seq_len", [16, 49, 100, 196])
    def test_various_seq_lengths(self, seq_len):
        """Test with various sequence lengths."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, seq_len, 256)
        
        y = res_ln(x)
        
        assert y.shape == (2, seq_len, 256)
    
    @pytest.mark.parametrize("embed_dim", [64, 128, 256, 512])
    def test_various_embed_dims(self, embed_dim):
        """Test with various embedding dimensions."""
        res_ln = ResidualLayerNormBlock(embed_dim=embed_dim)
        x = torch.randn(2, 49, embed_dim)
        
        y = res_ln(x)
        
        assert y.shape == (2, 49, embed_dim)


# ============================================================================
# Test Class 5: Residual Connection Tests
# ============================================================================

class TestResidualConnection:
    """Test residual connection logic."""
    
    def test_default_residual_is_input(self):
        """Test that default residual is the input itself."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        
        # y = LayerNorm(x) + x
        y = res_ln(x)
        
        # Manual computation
        y_manual = res_ln.norm(x) + x
        
        assert torch.allclose(y, y_manual, atol=1e-6)
    
    def test_explicit_residual(self):
        """Test with explicit residual."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        residual = torch.randn(2, 49, 256)
        
        # y = LayerNorm(x) + residual
        y = res_ln(x, residual=residual)
        
        # Manual computation
        y_manual = res_ln.norm(x) + residual
        
        assert torch.allclose(y, y_manual, atol=1e-6)
    
    def test_zero_residual(self):
        """Test with zero residual."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        zero_residual = torch.zeros(2, 49, 256)
        
        y = res_ln(x, residual=zero_residual)
        
        # Should be just LayerNorm(x)
        y_manual = res_ln.norm(x)
        
        assert torch.allclose(y, y_manual, atol=1e-6)
    
    def test_residual_addition_verified(self):
        """Verify residual is actually added."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        
        y_default = res_ln(x)  # Uses x as residual
        zero_res = torch.zeros_like(x)
        y_zero = res_ln(x, residual=zero_res)  # No residual
        
        # Difference should be approximately x
        diff = y_default - y_zero
        assert torch.allclose(diff, x, atol=1e-5)


# ============================================================================
# Test Class 6: Stateless Tests
# ============================================================================

class TestStatelessBehavior:
    """Test that LayerNorm is stateless (train == eval)."""
    
    def test_layernorm_stateless(self):
        """Test LayerNorm produces same output in train/eval."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        
        res_ln.train()
        y_train = res_ln(x).clone()
        
        res_ln.eval()
        with torch.no_grad():
            y_eval = res_ln(x).clone()
        
        assert torch.allclose(y_train, y_eval, atol=1e-6), \
            "LayerNorm should be stateless (train == eval)"
    
    def test_no_running_stats(self):
        """Verify LayerNorm has no running statistics."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        
        # LayerNorm should not have running_mean or running_var
        assert not hasattr(res_ln.norm, 'running_mean') or res_ln.norm.running_mean is None
        assert not hasattr(res_ln.norm, 'running_var') or res_ln.norm.running_var is None


# ============================================================================
# Test Class 7: Gradient Tests
# ============================================================================

class TestFFNGradient:
    """Test gradient flow for FFN blocks."""
    
    def test_bottleneck_ffn_gradient_flow(self):
        """Verify all BottleneckFFN parameters receive gradients."""
        ffn = BottleneckFFN(inp=256, oup=256)
        x = torch.randn(2, 49, 256)
        
        grad_info = check_gradient_flow(ffn, x)
        
        for name, has_grad in grad_info.items():
            assert has_grad, f"Parameter {name} did not receive gradient"
    
    def test_resln_gradient_flow(self):
        """Verify all ResLN parameters receive gradients."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        
        grad_info = check_gradient_flow(res_ln, x)
        
        for name, has_grad in grad_info.items():
            assert has_grad, f"Parameter {name} did not receive gradient"
    
    def test_gradient_through_bottleneck(self):
        """Verify gradients flow through bottleneck."""
        ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.1)
        x = torch.randn(2, 49, 256, requires_grad=True)
        
        y = ffn(x)
        loss = y.sum()
        loss.backward()
        
        # Input should receive gradient
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
    
    def test_gradient_through_residual(self):
        """Verify gradients flow through residual path."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256, requires_grad=True)
        residual = torch.randn(2, 49, 256, requires_grad=True)
        
        y = res_ln(x, residual=residual)
        loss = y.sum()
        loss.backward()
        
        # Both inputs should receive gradients
        assert x.grad is not None
        assert residual.grad is not None
        assert x.grad.abs().sum() > 0
        assert residual.grad.abs().sum() > 0
    
    def test_gradient_magnitude(self):
        """Check gradients are not exploding or vanishing."""
        ffn = BottleneckFFN(inp=256, oup=256)
        x = torch.randn(2, 49, 256)
        
        y = ffn(x)
        loss = y.mean()
        loss.backward()
        
        for name, param in ffn.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm > 1e-10, f"Gradient for {name} is vanishing"
                assert grad_norm < 1e5, f"Gradient for {name} is exploding"


# ============================================================================
# Test Class 8: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_token(self):
        """Test with single token sequence."""
        ffn = BottleneckFFN(inp=256, oup=256)
        x = torch.randn(2, 1, 256)
        
        y = ffn(x)
        
        assert y.shape == (2, 1, 256)
    
    def test_single_batch(self):
        """Test with batch size 1."""
        ffn = BottleneckFFN(inp=256, oup=256)
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        
        x = torch.randn(1, 49, 256)
        
        y_ffn = ffn(x)
        y_ln = res_ln(x)
        
        assert y_ffn.shape == (1, 49, 256)
        assert y_ln.shape == (1, 49, 256)
    
    def test_no_nan_inf_ffn(self):
        """Verify FFN output contains no NaN or Inf."""
        ffn = BottleneckFFN(inp=256, oup=256)
        ffn.eval()
        
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y = ffn(x)
        
        assert check_no_nan_inf(y, "FFN output")
    
    def test_no_nan_inf_resln(self):
        """Verify ResLN output contains no NaN or Inf."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y = res_ln(x)
        
        assert check_no_nan_inf(y, "ResLN output")
    
    def test_get_config_ffn(self):
        """Test FFN configuration retrieval."""
        ffn = BottleneckFFN(inp=256, oup=128, bottleneck_ratio=0.25, dropout=0.1)
        
        config = ffn.get_config()
        
        assert config['class'] == 'BottleneckFFN'
        assert config['inp'] == 256
        assert config['oup'] == 128
        assert config['bottleneck_ratio'] == 0.25
        assert config['bottleneck_channels'] == 64
    
    def test_get_config_resln(self):
        """Test ResLN configuration retrieval."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        
        config = res_ln.get_config()
        
        assert config['class'] == 'ResidualLayerNormBlock'
        assert config['embed_dim'] == 256


# ============================================================================
# Test Class 9: Benchmark Tests
# ============================================================================

class TestBenchmark:
    """Performance measurements."""
    
    def test_ffn_forward_time(self):
        """Measure FFN forward pass time."""
        ffn = BottleneckFFN(inp=256, oup=256)
        ffn.eval()
        x = torch.randn(2, 196, 256)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = ffn(x)
        
        import time
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = ffn(x)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = sum(times) / len(times)
        print(f"\nFFN forward time: {mean_time:.3f} ms")
        
        assert mean_time < 100, f"FFN too slow: {mean_time:.3f} ms"
    
    def test_resln_forward_time(self):
        """Measure ResLN forward pass time."""
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 196, 256)
        
        import time
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = res_ln(x)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = sum(times) / len(times)
        print(f"\nResLN forward time: {mean_time:.3f} ms")
        
        assert mean_time < 50, f"ResLN too slow: {mean_time:.3f} ms"
    
    def test_parameter_count(self):
        """Verify parameter counts."""
        ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.25)
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        
        ffn_params = count_parameters(ffn)
        resln_params = count_parameters(res_ln)
        
        print(f"\nFFN parameters: {ffn_params['total']:,}")
        print(f"ResLN parameters: {resln_params['total']:,}")
        
        # ResLN should have exactly 2 * embed_dim parameters (weight + bias)
        assert resln_params['total'] == 2 * 256


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])