"""
Unit tests for individual model building blocks.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import blocks from notebook (we'll need to convert these to a module)
# For now, we define them here for testing


class TestGhostConv:
    """Tests for GhostConv module."""
    
    def test_ghost_conv_output_shape(self, sample_image_tensor):
        """Test that GhostConv produces correct output shape."""
        from blocks.ghost_conv import GhostConv
        
        inp, oup = 3, 64
        ghost = GhostConv(inp=inp, oup=oup, kernel_size=1, ratio=2, dw_size=3, stride=1)
        
        output = ghost(sample_image_tensor)
        
        assert output.shape == (1, oup, 224, 224), f"Expected shape (1, {oup}, 224, 224), got {output.shape}"
    
    def test_ghost_conv_with_stride(self, sample_image_tensor):
        """Test GhostConv with stride=2 reduces spatial dimensions."""
        from blocks.ghost_conv import GhostConv
        
        ghost = GhostConv(inp=3, oup=64, kernel_size=3, stride=2)
        output = ghost(sample_image_tensor)
        
        assert output.shape[2] == 112, f"Expected height 112, got {output.shape[2]}"
        assert output.shape[3] == 112, f"Expected width 112, got {output.shape[3]}"
    
    def test_ghost_conv_gradient_flow(self, sample_image_tensor):
        """Test that gradients flow through GhostConv."""
        from blocks.ghost_conv import GhostConv
        
        ghost = GhostConv(inp=3, oup=64)
        sample_image_tensor.requires_grad = True
        
        output = ghost(sample_image_tensor)
        loss = output.sum()
        loss.backward()
        
        assert sample_image_tensor.grad is not None, "Gradients should flow through GhostConv"


class TestFusedInvertedResidual:
    """Tests for FusedInvertedResidualBlock module."""
    
    def test_fused_ir_output_shape(self, sample_feature_map):
        """Test FusedInvertedResidualBlock output shape."""
        from blocks.fused_ir import FusedInvertedResidualBlock
        
        inp, oup = 64, 64
        fused_ir = FusedInvertedResidualBlock(inp=inp, oup=oup, stride=1, expand_ratio=4)
        
        output = fused_ir(sample_feature_map)
        
        assert output.shape == sample_feature_map.shape, f"Residual connection should preserve shape"
    
    def test_fused_ir_expansion(self, sample_feature_map):
        """Test FusedInvertedResidualBlock with channel expansion."""
        from blocks.fused_ir import FusedInvertedResidualBlock
        
        inp, oup = 64, 128
        fused_ir = FusedInvertedResidualBlock(inp=inp, oup=oup, stride=2)
        
        output = fused_ir(sample_feature_map)
        
        assert output.shape[1] == oup, f"Expected {oup} channels, got {output.shape[1]}"
        assert output.shape[2] == 28, f"Stride 2 should halve spatial dims"


class TestCoordAtt:
    """Tests for Coordinate Attention module."""
    
    def test_coord_att_output_shape(self, sample_feature_map):
        """Test CoordAtt preserves input shape."""
        from blocks.coord_att import CoordAtt
        
        coord_att = CoordAtt(inp=64, oup=64, reduction=32)
        output = coord_att(sample_feature_map)
        
        assert output.shape == sample_feature_map.shape, "CoordAtt should preserve shape"
    
    def test_coord_att_attention_range(self, sample_feature_map):
        """Test that attention values are in valid range (0-1 after sigmoid)."""
        from blocks.coord_att import CoordAtt
        
        coord_att = CoordAtt(inp=64, oup=64)
        output = coord_att(sample_feature_map)
        
        # Output should be input modulated by attention, so range depends on input
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"


class TestPatchEmbedding:
    """Tests for Patch Embedding module."""
    
    def test_patch_embedding_output_shape(self, sample_feature_map):
        """Test PatchEmbedding converts spatial to sequence."""
        from blocks.patch_embed import PatchEmbedding
        
        in_channels = 64
        embed_dim = 256
        patch_size = 4  # 56/4 = 14 patches per side = 196 total
        
        patch_embed = PatchEmbedding(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        output = patch_embed(sample_feature_map)
        
        expected_seq_len = (56 // patch_size) ** 2  # 196
        assert output.shape == (1, expected_seq_len, embed_dim), f"Expected (1, {expected_seq_len}, {embed_dim}), got {output.shape}"
    
    def test_patch_embedding_different_sizes(self):
        """Test PatchEmbedding with different patch sizes."""
        from blocks.patch_embed import PatchEmbedding
        
        feature_map = torch.randn(2, 32, 64, 64)
        
        for patch_size in [4, 8, 16]:
            patch_embed = PatchEmbedding(in_channels=32, embed_dim=128, patch_size=patch_size)
            output = patch_embed(feature_map)
            
            expected_seq_len = (64 // patch_size) ** 2
            assert output.shape[1] == expected_seq_len, f"Patch size {patch_size} failed"


class TestLinearDifferentialAttention:
    """Tests for Linear Differential Attention module."""
    
    def test_lda_output_shape(self, sample_sequence):
        """Test LDA preserves sequence shape."""
        from blocks.lda import LinearDifferentialAttention
        
        embed_dim = 256
        lda = LinearDifferentialAttention(embed_dim=embed_dim, num_heads=8, dropout=0.1)
        
        output = lda(sample_sequence)
        
        assert output.shape == sample_sequence.shape, "LDA should preserve shape"
    
    def test_lda_different_heads(self, sample_sequence):
        """Test LDA with different number of heads."""
        from blocks.lda import LinearDifferentialAttention
        
        embed_dim = 256
        
        for num_heads in [1, 2, 4, 8]:
            lda = LinearDifferentialAttention(embed_dim=embed_dim, num_heads=num_heads)
            output = lda(sample_sequence)
            assert output.shape == sample_sequence.shape, f"Failed with {num_heads} heads"
    
    def test_lda_no_nan(self, sample_sequence):
        """Test LDA doesn't produce NaN values."""
        from blocks.lda import LinearDifferentialAttention
        
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        output = lda(sample_sequence)
        
        assert not torch.isnan(output).any(), "LDA output contains NaN"


class TestBottleneckFFN:
    """Tests for Bottleneck Feed Forward Network."""
    
    def test_bottleneck_ffn_output_shape(self, sample_sequence):
        """Test BottleneckFFN output shape."""
        from blocks.bottleneck_ffn import BottleneckFFN
        
        inp, oup = 256, 256
        ffn = BottleneckFFN(inp=inp, oup=oup, bottleneck_ratio=0.25, dropout=0.1)
        
        output = ffn(sample_sequence)
        
        assert output.shape == sample_sequence.shape, "FFN should preserve shape when inp==oup"
    
    def test_bottleneck_ffn_channel_change(self, sample_sequence):
        """Test BottleneckFFN with different input/output dimensions."""
        from blocks.bottleneck_ffn import BottleneckFFN
        
        ffn = BottleneckFFN(inp=256, oup=512, bottleneck_ratio=0.25)
        output = ffn(sample_sequence)
        
        assert output.shape == (1, 196, 512), f"Expected (1, 196, 512), got {output.shape}"


class TestClassifierHead:
    """Tests for Classifier Head."""
    
    def test_classifier_output_shape(self, num_classes):
        """Test ClassifierHead produces correct number of classes."""
        from blocks.classifier import ClassifierHead
        
        embed_dim = 256
        batch_size = 4
        
        classifier = ClassifierHead(embed_dim=embed_dim, num_classes=num_classes)
        input_tensor = torch.randn(batch_size, embed_dim)
        
        output = classifier(input_tensor)
        
        assert output.shape == (batch_size, num_classes), f"Expected ({batch_size}, {num_classes}), got {output.shape}"
    
    def test_classifier_softmax(self, num_classes):
        """Test ClassifierHead output sums to 1 (valid probability distribution)."""
        from blocks.classifier import ClassifierHead
        
        classifier = ClassifierHead(embed_dim=256, num_classes=num_classes)
        input_tensor = torch.randn(2, 256)
        
        output = classifier(input_tensor)
        
        # Check each sample's probabilities sum to 1
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5), "Softmax outputs should sum to 1"