"""
Comprehensive unit tests for Patch Embedding and Positional Encoding.

Test Categories:
1. Patch Embedding Shape Tests
2. Patch Embedding Edge Cases
3. Positional Encoding Shape Tests
4. Positional Encoding Value Tests
5. Combined Transition Pipeline Tests
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.blocks import PatchEmbedding, PositionalEncoding
from src.utils.testing import (
    check_output_shape,
    check_gradient_flow,
    check_no_nan_inf,
    count_parameters,
)


# ============================================================================
# Test Class 1: Patch Embedding Shape Tests
# ============================================================================

class TestPatchEmbedShape:
    """Test output shapes for Patch Embedding."""
    
    def test_basic_shape(self):
        """Test basic shape transformation."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        x = torch.randn(2, 64, 56, 56)
        
        y = patch_embed(x)
        
        # N = (56/4) * (56/4) = 14 * 14 = 196
        expected_shape = (2, 196, 256)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    def test_num_patches_calculation(self):
        """Test that num_patches is calculated correctly."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        
        # Test various spatial dimensions
        test_cases = [
            ((56, 56), 196),   # 14 * 14
            ((28, 28), 49),    # 7 * 7
            ((14, 14), 9),     # Cannot get 14/4 = 3.5, so 3 * 3 = 9? No, floor division
            ((112, 112), 784), # 28 * 28
        ]
        
        for (H, W), expected_patches in test_cases:
            actual = patch_embed.get_num_patches(H, W)
            # Recalculate expected with floor division
            expected = (H // 4) * (W // 4)
            assert actual == expected, f"For ({H}, {W}): expected {expected}, got {actual}"
    
    @pytest.mark.parametrize("patch_size", [2, 4, 7, 14])
    def test_various_patch_sizes(self, patch_size):
        """Test with various patch sizes."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=patch_size)
        
        # Use spatial size divisible by all patch sizes
        H = W = 28
        x = torch.randn(2, 64, H, W)
        
        y = patch_embed(x)
        
        expected_patches = (H // patch_size) * (W // patch_size)
        expected_shape = (2, expected_patches, 256)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    @pytest.mark.parametrize("in_channels", [32, 64, 128, 256])
    def test_various_input_channels(self, in_channels):
        """Test with various input channel counts."""
        patch_embed = PatchEmbedding(in_channels=in_channels, embed_dim=256, patch_size=4)
        x = torch.randn(2, in_channels, 28, 28)
        
        y = patch_embed(x)
        
        expected_shape = (2, 49, 256)  # 7 * 7 = 49
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    @pytest.mark.parametrize("embed_dim", [128, 256, 384, 512])
    def test_various_embed_dims(self, embed_dim):
        """Test with various embedding dimensions."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=embed_dim, patch_size=4)
        x = torch.randn(2, 64, 28, 28)
        
        y = patch_embed(x)
        
        expected_shape = (2, 49, embed_dim)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_size_variations(self, batch_size):
        """Test with various batch sizes."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        x = torch.randn(batch_size, 64, 28, 28)
        
        y = patch_embed(x)
        
        expected_shape = (batch_size, 49, 256)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    def test_rectangular_input(self):
        """Test with non-square spatial dimensions."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        x = torch.randn(2, 64, 28, 56)  # H != W
        
        y = patch_embed(x)
        
        # N = (28/4) * (56/4) = 7 * 14 = 98
        expected_shape = (2, 98, 256)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"


# ============================================================================
# Test Class 2: Patch Embedding Edge Cases
# ============================================================================

class TestPatchEmbedEdgeCases:
    """Test edge cases for Patch Embedding."""
    
    def test_non_divisible_spatial(self):
        """Test when spatial dims not divisible by patch_size."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        
        # 30 is not divisible by 4: 30 // 4 = 7
        x = torch.randn(2, 64, 30, 30)
        
        y = patch_embed(x)
        
        # Should use floor division: 7 * 7 = 49 patches
        expected_shape = (2, 49, 256)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    def test_minimum_spatial(self):
        """Test with minimum spatial dimensions equal to patch_size."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        x = torch.randn(2, 64, 4, 4)  # H = W = patch_size
        
        y = patch_embed(x)
        
        # N = 1 * 1 = 1 patch
        expected_shape = (2, 1, 256)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    def test_single_patch(self):
        """Test resulting in a single patch."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=7)
        x = torch.randn(2, 64, 7, 7)
        
        y = patch_embed(x)
        
        expected_shape = (2, 1, 256)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    def test_small_patch_size(self):
        """Test with very small patch size."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=1)
        x = torch.randn(2, 64, 14, 14)
        
        y = patch_embed(x)
        
        # N = 14 * 14 = 196 patches (each pixel is a patch)
        expected_shape = (2, 196, 256)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    def test_large_patch_size(self):
        """Test with large patch size."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=14)
        x = torch.randn(2, 64, 56, 56)
        
        y = patch_embed(x)
        
        # N = (56/14) * (56/14) = 4 * 4 = 16
        expected_shape = (2, 16, 256)
        assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    
    def test_gradient_flow(self):
        """Verify gradients flow through patch embedding."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        x = torch.randn(2, 64, 28, 28, requires_grad=True)
        
        y = patch_embed(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None, "Input gradient is None"
        
        grad_info = check_gradient_flow(patch_embed, torch.randn(2, 64, 28, 28))
        for name, has_grad in grad_info.items():
            assert has_grad, f"Parameter {name} did not receive gradient"
    
    def test_no_nan_inf(self):
        """Verify output contains no NaN or Inf."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        patch_embed.eval()
        
        x = torch.randn(2, 64, 28, 28)
        
        with torch.no_grad():
            y = patch_embed(x)
        
        assert check_no_nan_inf(y, "PatchEmbedding output")


# ============================================================================
# Test Class 3: Positional Encoding Shape Tests
# ============================================================================

class TestPositionalEncodingShape:
    """Test output shapes for Positional Encoding."""
    
    def test_output_same_shape(self):
        """Test that output shape equals input shape."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        x = torch.randn(2, 196, 256)
        
        y = pos_enc(x)
        
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    
    @pytest.mark.parametrize("seq_len", [16, 49, 100, 196, 400, 1000])
    def test_various_seq_lengths(self, seq_len):
        """Test with various sequence lengths."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        x = torch.randn(2, seq_len, 256)
        
        y = pos_enc(x)
        
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    
    def test_max_len_boundary(self):
        """Test with sequence length equal to max_len."""
        max_len = 500
        pos_enc = PositionalEncoding(embed_dim=256, max_len=max_len)
        x = torch.randn(2, max_len, 256)
        
        y = pos_enc(x)
        
        assert y.shape == x.shape
    
    def test_exceeds_max_len_raises_error(self):
        """Test that exceeding max_len raises an error."""
        max_len = 100
        pos_enc = PositionalEncoding(embed_dim=256, max_len=max_len)
        x = torch.randn(2, max_len + 1, 256)
        
        with pytest.raises(ValueError, match="exceeds maximum length"):
            pos_enc(x)
    
    @pytest.mark.parametrize("embed_dim", [64, 128, 256, 384, 512])
    def test_various_embed_dims(self, embed_dim):
        """Test with various embedding dimensions."""
        pos_enc = PositionalEncoding(embed_dim=embed_dim, max_len=5000)
        x = torch.randn(2, 49, embed_dim)
        
        y = pos_enc(x)
        
        assert y.shape == x.shape
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_size_variations(self, batch_size):
        """Test with various batch sizes."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        x = torch.randn(batch_size, 49, 256)
        
        y = pos_enc(x)
        
        assert y.shape == x.shape


# ============================================================================
# Test Class 4: Positional Encoding Value Tests
# ============================================================================

class TestPositionalEncodingValues:
    """Test positional encoding values and properties."""
    
    def test_position_info_added(self):
        """Test that positional information is actually added."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        x = torch.randn(2, 49, 256)
        
        y = pos_enc(x)
        
        # Output should differ from input
        diff = (y - x).abs().mean()
        assert diff > 0, "Positional encoding did not change the values"
    
    def test_sinusoidal_pattern(self):
        """Verify the sinusoidal pattern in positional encoding."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=100)
        
        # Get the PE buffer
        pe = pos_enc.pe[0]  # (max_len, embed_dim)
        
        # Check first position is (sin(0), cos(0), sin(0), cos(0), ...) = (0, 1, 0, 1, ...)
        # sin(0) = 0, cos(0) = 1
        assert abs(pe[0, 0].item() - 0.0) < 1e-5, "PE[0,0] should be sin(0)=0"
        assert abs(pe[0, 1].item() - 1.0) < 1e-5, "PE[0,1] should be cos(0)=1"
        
        # Verify sin/cos pattern at a non-zero position
        pos = 10
        dim_idx = 2
        
        # Expected: sin(pos / 10000^(2*dim_idx/embed_dim))
        div_term = 10000 ** (2 * dim_idx / 256)
        expected_sin = math.sin(pos / div_term)
        expected_cos = math.cos(pos / div_term)
        
        assert abs(pe[pos, 2*dim_idx].item() - expected_sin) < 1e-5, \
            f"Sin pattern mismatch at pos={pos}, dim={2*dim_idx}"
        assert abs(pe[pos, 2*dim_idx + 1].item() - expected_cos) < 1e-5, \
            f"Cos pattern mismatch at pos={pos}, dim={2*dim_idx + 1}"
    
    def test_no_learnable_params(self):
        """Test that positional encoding has no learnable parameters."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        num_params = sum(p.numel() for p in pos_enc.parameters())
        
        assert num_params == 0, f"Expected 0 parameters, got {num_params}"
    
    def test_pe_buffer_shape(self):
        """Test that PE buffer has correct shape."""
        embed_dim = 256
        max_len = 5000
        pos_enc = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
        
        expected_shape = (1, max_len, embed_dim)
        assert pos_enc.pe.shape == expected_shape, \
            f"Expected PE shape {expected_shape}, got {pos_enc.pe.shape}"
    
    def test_pe_is_buffer_not_parameter(self):
        """Test that PE is registered as buffer, not parameter."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        # Check it's in named_buffers
        buffer_names = [name for name, _ in pos_enc.named_buffers()]
        assert 'pe' in buffer_names, "PE should be registered as a buffer"
        
        # Check it's not in named_parameters
        param_names = [name for name, _ in pos_enc.named_parameters()]
        assert 'pe' not in param_names, "PE should not be a parameter"
    
    def test_different_positions_different_values(self):
        """Test that different positions have different encodings."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        pe = pos_enc.pe[0]
        
        # Check that position 0 and position 1 are different
        diff = (pe[0] - pe[1]).abs().sum()
        assert diff > 0, "Different positions should have different encodings"
    
    def test_deterministic_encoding(self):
        """Test that encoding is deterministic (same every time)."""
        pos_enc1 = PositionalEncoding(embed_dim=256, max_len=5000)
        pos_enc2 = PositionalEncoding(embed_dim=256, max_len=5000)
        
        assert torch.allclose(pos_enc1.pe, pos_enc2.pe), \
            "Positional encoding should be deterministic"
    
    def test_odd_embed_dim(self):
        """Test with odd embedding dimension."""
        pos_enc = PositionalEncoding(embed_dim=255, max_len=100)
        x = torch.randn(2, 49, 255)
        
        y = pos_enc(x)
        
        assert y.shape == x.shape
        assert not torch.isnan(y).any(), "NaN values in output with odd embed_dim"


# ============================================================================
# Test Class 5: Combined Transition Pipeline Tests
# ============================================================================

class TestCombinedTransition:
    """Test the combined PatchEmbedding → PositionalEncoding pipeline."""
    
    def test_patch_to_pos_pipeline(self):
        """Test that PatchEmbed output can be fed to PosEnc."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        # CNN output
        cnn_output = torch.randn(2, 64, 56, 56)
        
        # Pipeline
        patches = patch_embed(cnn_output)
        output = pos_enc(patches)
        
        expected_shape = (2, 196, 256)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_shape_through_pipeline(self):
        """Verify shapes at each step of the pipeline."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        # Input: CNN features
        x = torch.randn(2, 64, 56, 56)
        print(f"CNN output:      {x.shape}")
        
        # After PatchEmbed
        patches = patch_embed(x)
        print(f"After PatchEmbed: {patches.shape}")
        assert patches.shape == (2, 196, 256)
        
        # After PosEnc
        output = pos_enc(patches)
        print(f"After PosEnc:     {output.shape}")
        assert output.shape == (2, 196, 256)
    
    def test_pipeline_gradient_flow(self):
        """Test gradient flow through entire pipeline."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        x = torch.randn(2, 64, 56, 56, requires_grad=True)
        
        patches = patch_embed(x)
        output = pos_enc(patches)
        loss = output.sum()
        loss.backward()
        
        # Check input gradient
        assert x.grad is not None, "Input gradient is None"
        assert x.grad.abs().sum() > 0, "Input gradient is zero"
    
    def test_pipeline_no_nan_inf(self):
        """Test that pipeline produces no NaN/Inf values."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        patch_embed.eval()
        pos_enc.eval()
        
        x = torch.randn(2, 64, 56, 56)
        
        with torch.no_grad():
            patches = patch_embed(x)
            output = pos_enc(patches)
        
        assert check_no_nan_inf(output, "Pipeline output")
    
    def test_pipeline_deterministic(self):
        """Test that pipeline is deterministic in eval mode."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        patch_embed.eval()
        pos_enc.eval()
        
        x = torch.randn(2, 64, 56, 56)
        
        with torch.no_grad():
            y1 = pos_enc(patch_embed(x)).clone()
            y2 = pos_enc(patch_embed(x)).clone()
        
        assert torch.allclose(y1, y2), "Pipeline should be deterministic"
    
    def test_different_spatial_sizes(self):
        """Test pipeline with different spatial sizes."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        spatial_sizes = [(28, 28), (56, 56), (112, 112), (56, 28)]
        
        for H, W in spatial_sizes:
            x = torch.randn(2, 64, H, W)
            
            patches = patch_embed(x)
            output = pos_enc(patches)
            
            expected_patches = (H // 4) * (W // 4)
            expected_shape = (2, expected_patches, 256)
            
            assert output.shape == expected_shape, \
                f"For ({H}, {W}): expected {expected_shape}, got {output.shape}"


# ============================================================================
# Test Class 6: Parameter Count and Config Tests
# ============================================================================

class TestConfigAndParameters:
    """Test configuration and parameter counting."""
    
    def test_patch_embed_parameter_count(self):
        """Test PatchEmbedding parameter count."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        
        params = count_parameters(patch_embed)
        
        # Expected: Conv2d weights + bias
        # Weight: in_channels * embed_dim * patch_size * patch_size
        # Bias: embed_dim
        expected_weight = 64 * 256 * 4 * 4
        expected_bias = 256
        expected_total = expected_weight + expected_bias
        
        assert params['total'] == expected_total, \
            f"Expected {expected_total} params, got {params['total']}"
    
    def test_pos_enc_parameter_count(self):
        """Test PositionalEncoding parameter count is 0."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        params = count_parameters(pos_enc)
        
        assert params['total'] == 0, f"Expected 0 params, got {params['total']}"
    
    def test_patch_embed_config(self):
        """Test PatchEmbedding configuration retrieval."""
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        
        config = patch_embed.get_config()
        
        assert config['class'] == 'PatchEmbedding'
        assert config['in_channels'] == 64
        assert config['embed_dim'] == 256
        assert config['patch_size'] == 4
    
    def test_pos_enc_config(self):
        """Test PositionalEncoding configuration retrieval."""
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        config = pos_enc.get_config()
        
        assert config['class'] == 'PositionalEncoding'
        assert config['embed_dim'] == 256
        assert config['max_len'] == 5000
        assert config['learnable'] == False


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])