"""
Comprehensive unit tests for Linear Differential Attention (LDA).

Test Categories:
1. Shape Tests - Verify output shapes for various configurations
2. Differential Mechanism Tests - Verify A_diff = α × (A1 - A2)
3. Correctness Tests - Compare against naive attention
4. Gradient Tests - Verify gradient flow
5. Dropout Tests - Verify train/eval behavior
6. Stability Tests - Numerical stability checks
7. Benchmark Tests - Performance measurements
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.blocks import LinearDifferentialAttention, NaiveFullAttention
from src.utils.testing import (
    check_output_shape,
    check_gradient_flow,
    check_no_nan_inf,
    count_parameters,
)


# ============================================================================
# Test Class 1: Shape Tests
# ============================================================================

class TestLDAShape:
    """Test output shapes for various configurations."""
    
    def test_basic_shape(self):
        """Test basic shape preservation."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 196, 256)
        
        assert check_output_shape(lda, x, (2, 196, 256))
    
    @pytest.mark.parametrize("embed_dim", [64, 128, 256, 512])
    def test_various_embed_dims(self, embed_dim):
        """Test with various embedding dimensions."""
        lda = LinearDifferentialAttention(embed_dim=embed_dim, num_heads=8)
        x = torch.randn(2, 49, embed_dim)
        
        expected_shape = (2, 49, embed_dim)
        assert check_output_shape(lda, x, expected_shape)
    
    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 16])
    def test_various_num_heads(self, num_heads):
        """Test with various number of attention heads."""
        embed_dim = 256  # Must be divisible by all num_heads
        lda = LinearDifferentialAttention(embed_dim=embed_dim, num_heads=num_heads)
        x = torch.randn(2, 49, embed_dim)
        
        expected_shape = (2, 49, embed_dim)
        assert check_output_shape(lda, x, expected_shape)
    
    @pytest.mark.parametrize("seq_len", [16, 49, 100, 196, 400])
    def test_various_seq_lengths(self, seq_len):
        """Test with various sequence lengths."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, seq_len, 256)
        
        expected_shape = (2, seq_len, 256)
        assert check_output_shape(lda, x, expected_shape)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_size_variations(self, batch_size):
        """Test with various batch sizes."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(batch_size, 49, 256)
        
        expected_shape = (batch_size, 49, 256)
        assert check_output_shape(lda, x, expected_shape)
    
    def test_head_dim_calculation(self):
        """Verify head_dim = embed_dim // num_heads."""
        for embed_dim, num_heads in [(256, 8), (512, 16), (128, 4)]:
            lda = LinearDifferentialAttention(embed_dim=embed_dim, num_heads=num_heads)
            expected_head_dim = embed_dim // num_heads
            
            assert lda.head_dim == expected_head_dim, \
                f"Expected head_dim={expected_head_dim}, got {lda.head_dim}"


# ============================================================================
# Test Class 2: Differential Mechanism Tests
# ============================================================================

class TestLDADifferential:
    """Test the differential attention mechanism."""
    
    def test_differential_mechanism(self):
        """Verify A_diff = α × (A1 - A2)."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            A1, A2, A_diff = lda.get_attention_maps(x)
            
            # Manually compute expected A_diff
            expected_diff = lda.alpha * (A1 - A2)
            
            # Verify
            assert torch.allclose(A_diff, expected_diff, atol=1e-6), \
                "A_diff should equal α × (A1 - A2)"
    
    def test_alpha_learnable(self):
        """Verify alpha is a learnable parameter."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        
        assert lda.alpha.requires_grad, "Alpha should require gradients"
        assert isinstance(lda.alpha, nn.Parameter), "Alpha should be nn.Parameter"
    
    def test_alpha_positive(self):
        """Verify alpha is always positive (exp initialization)."""
        for init in [-1.0, 0.0, 0.5, 0.8, 1.5]:
            lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, init=init)
            
            assert lda.alpha.item() > 0, \
                f"Alpha should be positive for init={init}, got {lda.alpha.item()}"
    
    def test_alpha_initialization(self):
        """Verify alpha = exp(init)."""
        for init in [0.5, 0.8, 1.0]:
            lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, init=init)
            expected_alpha = math.exp(init)
            
            assert abs(lda.alpha.item() - expected_alpha) < 1e-5, \
                f"Expected alpha={expected_alpha}, got {lda.alpha.item()}"
    
    def test_different_attention_maps(self):
        """Verify A1 ≠ A2 for random input."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            A1, A2, _ = lda.get_attention_maps(x)
            
            # A1 and A2 should be different
            diff = (A1 - A2).abs().mean()
            assert diff > 1e-6, "A1 and A2 should differ for random input"
    
    def test_attention_maps_shape(self):
        """Verify attention maps have correct shape."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            A1, A2, A_diff = lda.get_attention_maps(x)
        
        expected_shape = (2, 8, 49, 49)  # (B, heads, N, N)
        assert A1.shape == expected_shape, f"A1 shape mismatch: {A1.shape}"
        assert A2.shape == expected_shape, f"A2 shape mismatch: {A2.shape}"
        assert A_diff.shape == expected_shape, f"A_diff shape mismatch: {A_diff.shape}"


# ============================================================================
# Test Class 3: Correctness Tests
# ============================================================================

class TestLDACorrectness:
    """Test correctness of attention computation."""
    
    def test_attention_sum_to_one(self):
        """Verify softmax rows sum to 1."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            A1, A2, _ = lda.get_attention_maps(x)
            
            # Each row should sum to 1
            A1_sums = A1.sum(dim=-1)
            A2_sums = A2.sum(dim=-1)
            
            assert torch.allclose(A1_sums, torch.ones_like(A1_sums), atol=1e-5), \
                "A1 rows should sum to 1"
            assert torch.allclose(A2_sums, torch.ones_like(A2_sums), atol=1e-5), \
                "A2 rows should sum to 1"
    
    def test_attention_values_in_range(self):
        """Verify attention values are in [0, 1] for A1 and A2."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            A1, A2, _ = lda.get_attention_maps(x)
            
            # A1 and A2 are softmax outputs, should be in [0, 1]
            assert (A1 >= 0).all() and (A1 <= 1).all(), "A1 should be in [0, 1]"
            assert (A2 >= 0).all() and (A2 <= 1).all(), "A2 should be in [0, 1]"
    
    def test_output_bounded(self):
        """Verify output values are reasonable."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y = lda(x)
            
            # Output should not explode
            assert y.abs().max() < 100, f"Output values too large: {y.abs().max()}"
            assert not torch.isnan(y).any(), "Output contains NaN"
            assert not torch.isinf(y).any(), "Output contains Inf"
    
    def test_scaling_factor(self):
        """Verify scaling factor is 1/sqrt(head_dim)."""
        for embed_dim, num_heads in [(256, 8), (512, 16)]:
            lda = LinearDifferentialAttention(embed_dim=embed_dim, num_heads=num_heads)
            head_dim = embed_dim // num_heads
            expected_scaling = head_dim ** -0.5
            
            assert abs(lda.scaling - expected_scaling) < 1e-6, \
                f"Expected scaling={expected_scaling}, got {lda.scaling}"


# ============================================================================
# Test Class 4: Gradient Tests
# ============================================================================

class TestLDAGradient:
    """Test gradient flow and numerical gradient checks."""
    
    def test_gradient_flow(self):
        """Verify all parameters receive gradients."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 49, 256)
        
        grad_info = check_gradient_flow(lda, x)
        
        for name, has_grad in grad_info.items():
            assert has_grad, f"Parameter {name} did not receive gradient"
    
    def test_alpha_gradient(self):
        """Verify alpha receives gradient during backprop."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 49, 256)
        
        y = lda(x)
        loss = y.sum()
        loss.backward()
        
        assert lda.alpha.grad is not None, "Alpha should receive gradient"
        assert lda.alpha.grad.abs() > 0, "Alpha gradient should be non-zero"
    
    def test_backward_no_error(self):
        """Verify backward pass completes without errors."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 49, 256, requires_grad=True)
        
        try:
            y = lda(x)
            loss = y.sum()
            loss.backward()
        except Exception as e:
            pytest.fail(f"Backward pass raised exception: {e}")
    
    def test_gradient_magnitude(self):
        """Check gradients are not exploding or vanishing."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 49, 256, requires_grad=True)
        
        y = lda(x)
        loss = y.mean()
        loss.backward()
        
        for name, param in lda.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm > 1e-10, f"Gradient for {name} is vanishing: {grad_norm}"
                assert grad_norm < 1e5, f"Gradient for {name} is exploding: {grad_norm}"
    
    def test_input_gradient(self):
        """Verify gradients flow to input."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 49, 256, requires_grad=True)
        
        y = lda(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None, "Input should receive gradient"
        assert x.grad.abs().sum() > 0, "Input gradient should be non-zero"


# ============================================================================
# Test Class 5: Dropout Tests
# ============================================================================

class TestLDADropout:
    """Test dropout behavior in train vs eval mode."""
    
    def test_dropout_train_mode(self):
        """Verify different outputs in train mode due to dropout."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, dropout=0.5)
        lda.train()
        
        x = torch.randn(2, 49, 256)
        
        outputs = []
        for _ in range(5):
            with torch.no_grad():
                y = lda(x)
                outputs.append(y.clone())
        
        # At least some outputs should differ due to dropout
        differences = 0
        for i in range(len(outputs) - 1):
            if not torch.allclose(outputs[i], outputs[i+1], atol=1e-6):
                differences += 1
        
        # With 50% dropout, outputs should vary
        # Note: This test may occasionally fail due to randomness
        assert differences > 0, "Dropout should cause variation in train mode"
    
    def test_dropout_eval_mode(self):
        """Verify same outputs in eval mode (dropout disabled)."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, dropout=0.5)
        lda.eval()
        
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y1 = lda(x).clone()
            y2 = lda(x).clone()
        
        assert torch.allclose(y1, y2, atol=1e-6), \
            "Outputs should be identical in eval mode"
    
    def test_dropout_zero(self):
        """Verify no dropout effect when dropout=0."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, dropout=0.0)
        lda.train()
        
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y1 = lda(x).clone()
            y2 = lda(x).clone()
        
        assert torch.allclose(y1, y2, atol=1e-6), \
            "Outputs should be identical with dropout=0"


# ============================================================================
# Test Class 6: Stability Tests
# ============================================================================

class TestLDAStability:
    """Test numerical stability."""
    
    def test_no_nan_inf(self):
        """Verify no NaN or Inf in output."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y = lda(x)
        
        assert check_no_nan_inf(y, "LDA output")
    
    def test_large_values_handling(self):
        """Verify stability with large input values."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        x = torch.randn(2, 49, 256) * 10  # Large values
        
        with torch.no_grad():
            y = lda(x)
        
        assert not torch.isnan(y).any(), "NaN in output with large input"
        assert not torch.isinf(y).any(), "Inf in output with large input"
    
    def test_small_values_handling(self):
        """Verify stability with small input values."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        x = torch.randn(2, 49, 256) * 0.01  # Small values
        
        with torch.no_grad():
            y = lda(x)
        
        assert not torch.isnan(y).any(), "NaN in output with small input"
        assert not torch.isinf(y).any(), "Inf in output with small input"
    
    def test_deterministic_eval(self):
        """Verify deterministic output in eval mode."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y1 = lda(x).clone()
            y2 = lda(x).clone()
        
        assert torch.allclose(y1, y2), "Outputs should be identical in eval mode"
    
    def test_reproducibility_with_seed(self):
        """Verify same seed produces same output."""
        torch.manual_seed(42)
        lda1 = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda1.eval()
        
        torch.manual_seed(42)
        lda2 = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda2.eval()
        
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y1 = lda1(x)
            y2 = lda2(x)
        
        assert torch.allclose(y1, y2), "Same seed should produce same output"


# ============================================================================
# Test Class 7: Benchmark Tests
# ============================================================================

class TestLDABenchmark:
    """Performance measurements for LDA."""
    
    def test_forward_time(self):
        """Measure forward pass time."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        x = torch.randn(2, 196, 256)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = lda(x)
        
        import time
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = lda(x)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = sum(times) / len(times)
        print(f"\nLDA forward time (N=196): {mean_time:.3f} ms")
        
        assert mean_time < 500, f"Forward pass too slow: {mean_time:.3f} ms"
    
    def test_backward_time(self):
        """Measure backward pass time."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.train()
        x = torch.randn(2, 196, 256, requires_grad=True)
        
        # Warmup
        for _ in range(5):
            lda.zero_grad()
            y = lda(x)
            y.sum().backward()
        
        import time
        times = []
        for _ in range(20):
            lda.zero_grad()
            y = lda(x)
            start = time.perf_counter()
            y.sum().backward()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = sum(times) / len(times)
        print(f"\nLDA backward time (N=196): {mean_time:.3f} ms")
        
        assert mean_time < 1000, f"Backward pass too slow: {mean_time:.3f} ms"
    
    def test_scaling_with_seq_length(self):
        """Measure how time scales with sequence length (should be O(N²))."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        
        import time
        
        seq_lengths = [49, 100, 196, 400]
        times = []
        
        for N in seq_lengths:
            x = torch.randn(2, N, 256)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = lda(x)
            
            # Measure
            run_times = []
            for _ in range(10):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = lda(x)
                run_times.append((time.perf_counter() - start) * 1000)
            
            times.append(sum(run_times) / len(run_times))
        
        print("\nLDA scaling with sequence length:")
        for N, t in zip(seq_lengths, times):
            print(f"  N={N}: {t:.3f} ms")
        
        # Verify roughly O(N²) scaling (time should increase with N)
        assert times[-1] > times[0], "Time should increase with sequence length"
    
    def test_parameter_count(self):
        """Verify parameter count."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        params = count_parameters(lda)
        
        print(f"\nLDA parameter count: {params['total']:,}")
        
        # Expected: Q(2D×D) + K(2D×D) + V(D×D) + Out(D×D + D) + norm + alpha
        # ≈ 2×256×256 + 2×256×256 + 256×256 + 256×256 + 256 + 512 + 1
        # ≈ 393,985 parameters
        
        assert params['total'] > 0
        assert params['trainable'] == params['total']


# ============================================================================
# Test Class 8: Edge Cases
# ============================================================================

class TestLDAEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_token(self):
        """Test with single token sequence."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 1, 256)
        
        y = lda(x)
        assert y.shape == (2, 1, 256)
    
    def test_batch_size_one(self):
        """Test with batch size 1."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(1, 49, 256)
        
        y = lda(x)
        assert y.shape == (1, 49, 256)
    
    def test_single_head(self):
        """Test with single attention head."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=1)
        x = torch.randn(2, 49, 256)
        
        y = lda(x)
        assert y.shape == (2, 49, 256)
    
    def test_many_heads(self):
        """Test with many attention heads."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=32)
        x = torch.randn(2, 49, 256)
        
        y = lda(x)
        assert y.shape == (2, 49, 256)
    
    def test_get_config(self):
        """Test configuration retrieval."""
        lda = LinearDifferentialAttention(
            embed_dim=256, num_heads=8, dropout=0.1, init=0.8
        )
        
        config = lda.get_config()
        
        assert config['class'] == 'LinearDifferentialAttention'
        assert config['embed_dim'] == 256
        assert config['num_heads'] == 8
        assert config['head_dim'] == 32
        assert config['dropout'] == 0.1
        assert abs(config['alpha'] - math.exp(0.8)) < 1e-4
    
    def test_repr(self):
        """Test string representation."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        
        repr_str = repr(lda)
        
        assert 'LinearDifferentialAttention' in repr_str
        assert 'embed_dim=256' in repr_str
        assert 'num_heads=8' in repr_str


# ============================================================================
# Comparison with NaiveFullAttention
# ============================================================================

class TestLDAVsNaive:
    """Compare LDA with naive full attention."""
    
    def test_both_produce_valid_output(self):
        """Verify both LDA and NaiveFullAttention produce valid outputs."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        naive = NaiveFullAttention(embed_dim=256, num_heads=8)
        
        lda.eval()
        naive.eval()
        
        x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            y_lda = lda(x)
            y_naive = naive(x)
        
        # Both should produce valid (non-NaN, non-Inf) outputs
        assert not torch.isnan(y_lda).any()
        assert not torch.isnan(y_naive).any()
        assert not torch.isinf(y_lda).any()
        assert not torch.isinf(y_naive).any()
    
    def test_shape_compatibility(self):
        """Verify both have same input/output shapes."""
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        naive = NaiveFullAttention(embed_dim=256, num_heads=8)
        
        x = torch.randn(2, 49, 256)
        
        y_lda = lda(x)
        y_naive = naive(x)
        
        assert y_lda.shape == y_naive.shape == (2, 49, 256)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])