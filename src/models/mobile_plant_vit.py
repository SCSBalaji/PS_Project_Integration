"""
MobilePlantViT: A Lightweight Hybrid CNN-Transformer for Plant Disease Classification.

This module implements the complete MobilePlantViT architecture, combining:
- CNN backbone with GhostConv, Fused-IR, and Coordinate Attention
- Patch embedding for CNN-to-Transformer transition
- Linear Differential Attention (LDA) transformer
- Bottleneck FFN with residual connections
- Global Average Pooling and classification head

The architecture is optimized for mobile deployment with < 5M parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

# Import all blocks
from ..blocks import (
    GhostConv,
    FusedInvertedResidualBlock,
    CoordAtt,
    PatchEmbedding,
    PositionalEncoding,
    LinearDifferentialAttention,
    ResidualLayerNormBlock,
    BottleneckFFN,
    GlobalAveragePooling,
    ClassifierHead,
)


@dataclass
class MobilePlantViTConfig:
    """
    Configuration class for MobilePlantViT model.
    
    Attributes:
        img_size: Input image size (assumes square images).
        in_channels: Number of input channels (3 for RGB).
        num_classes: Number of output classes.
        
        # CNN Stage
        ghost_out_channels: Output channels from GhostConv.
        ghost_ratio: Ghost ratio for GhostConv.
        ghost_dw_size: Depthwise kernel size in GhostConv.
        fused_ir_out_channels: Output channels from Fused-IR.
        fused_ir_stride: Stride for Fused-IR (controls downsampling).
        fused_ir_expand_ratio: Expansion ratio for Fused-IR.
        coord_att_reduction: Reduction ratio for Coordinate Attention.
        
        # Transition Stage
        embed_dim: Embedding dimension for transformer.
        patch_size: Patch size for patch embedding.
        max_seq_len: Maximum sequence length for positional encoding.
        
        # Transformer Stage
        num_heads: Number of attention heads in LDA.
        lda_dropout: Dropout rate in LDA.
        lda_init: Initialization value for LDA alpha.
        ffn_bottleneck_ratio: Bottleneck ratio for FFN.
        ffn_dropout: Dropout rate in FFN.
        
        # Classifier
        classifier_dropout: Dropout before classifier.
    """
    # Input
    img_size: int = 224
    in_channels: int = 3
    num_classes: int = 38
    
    # CNN Stage
    ghost_out_channels: int = 64
    ghost_ratio: int = 2
    ghost_dw_size: int = 3
    fused_ir_out_channels: int = 64
    fused_ir_stride: int = 4
    fused_ir_expand_ratio: int = 4
    coord_att_reduction: int = 32
    
    # Transition Stage
    embed_dim: int = 256
    patch_size: int = 4
    max_seq_len: int = 5000
    
    # Transformer Stage
    num_heads: int = 8
    lda_dropout: float = 0.1
    lda_init: float = 0.8
    ffn_bottleneck_ratio: float = 0.25
    ffn_dropout: float = 0.1
    
    # Classifier
    classifier_dropout: float = 0.0
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.img_size > 0, "img_size must be positive"
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.num_classes > 0, "num_classes must be positive"
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        assert 0 <= self.lda_dropout < 1, "lda_dropout must be in [0, 1)"
        assert 0 <= self.ffn_dropout < 1, "ffn_dropout must be in [0, 1)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'img_size': self.img_size,
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'ghost_out_channels': self.ghost_out_channels,
            'ghost_ratio': self.ghost_ratio,
            'ghost_dw_size': self.ghost_dw_size,
            'fused_ir_out_channels': self.fused_ir_out_channels,
            'fused_ir_stride': self.fused_ir_stride,
            'fused_ir_expand_ratio': self.fused_ir_expand_ratio,
            'coord_att_reduction': self.coord_att_reduction,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
            'max_seq_len': self.max_seq_len,
            'num_heads': self.num_heads,
            'lda_dropout': self.lda_dropout,
            'lda_init': self.lda_init,
            'ffn_bottleneck_ratio': self.ffn_bottleneck_ratio,
            'ffn_dropout': self.ffn_dropout,
            'classifier_dropout': self.classifier_dropout,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MobilePlantViTConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class MobilePlantViT(nn.Module):
    """
    MobilePlantViT: Lightweight Hybrid CNN-Transformer for Plant Disease Classification.
    
    Architecture Overview:
    ```
    Input (B, 3, 224, 224)
        │
        ▼ CNN Stage
    GhostConv → Fused-IR → CoordAtt
        │
        ▼ Transition Stage  
    PatchEmbed → PositionalEncoding
        │
        ▼ Transformer Stage
    LDA → ResLN → FFN
        │
        ▼ Classifier Stage
    GAP → ClassifierHead
        │
        ▼
    Output (B, num_classes)
    ```
    
    Args:
        config: MobilePlantViTConfig object or None for default config.
        **kwargs: Override config parameters.
    
    Examples:
        >>> # Default configuration
        >>> model = MobilePlantViT()
        >>> x = torch.randn(2, 3, 224, 224)
        >>> probs = model(x)
        >>> probs.shape
        torch.Size([2, 38])
        
        >>> # Custom configuration
        >>> config = MobilePlantViTConfig(num_classes=10, embed_dim=128)
        >>> model = MobilePlantViT(config)
        
        >>> # Override specific parameters
        >>> model = MobilePlantViT(num_classes=100, embed_dim=512)
    """
    
    def __init__(
        self,
        config: Optional[MobilePlantViTConfig] = None,
        **kwargs
    ):
        super().__init__()
        
        # Handle configuration
        if config is None:
            config = MobilePlantViTConfig(**kwargs)
        elif kwargs:
            # Override config with kwargs
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = MobilePlantViTConfig.from_dict(config_dict)
        
        self.config = config
        
        # Calculate intermediate dimensions
        self._calculate_dimensions()
        
        # Build model stages
        self._build_cnn_stage()
        self._build_transition_stage()
        self._build_transformer_stage()
        self._build_classifier_stage()
        
        # Initialize weights
        self._init_weights()
    
    def _calculate_dimensions(self):
        """Calculate intermediate tensor dimensions."""
        cfg = self.config
        
        # After GhostConv: same spatial size
        self.after_ghost_size = cfg.img_size
        
        # After Fused-IR: downsampled by stride
        self.after_fused_ir_size = cfg.img_size // cfg.fused_ir_stride
        
        # After CoordAtt: same as Fused-IR
        self.after_coord_att_size = self.after_fused_ir_size
        
        # After PatchEmbed: number of patches
        self.num_patches = (self.after_coord_att_size // cfg.patch_size) ** 2
        
        # Store for reference
        self.cnn_out_channels = cfg.fused_ir_out_channels
    
    def _build_cnn_stage(self):
        """Build CNN backbone stage."""
        cfg = self.config
        
        # GhostConv: 3 → ghost_out_channels
        self.ghost_conv = GhostConv(
            inp=cfg.in_channels,
            oup=cfg.ghost_out_channels,
            kernel_size=1,
            ratio=cfg.ghost_ratio,
            dw_size=cfg.ghost_dw_size,
            stride=1,
            relu=True
        )
        
        # Fused Inverted Residual: ghost_out_channels → fused_ir_out_channels
        self.fused_ir = FusedInvertedResidualBlock(
            inp=cfg.ghost_out_channels,
            oup=cfg.fused_ir_out_channels,
            stride=cfg.fused_ir_stride,
            expand_ratio=cfg.fused_ir_expand_ratio
        )
        
        # Coordinate Attention: same channels
        self.coord_att = CoordAtt(
            inp=cfg.fused_ir_out_channels,
            oup=cfg.fused_ir_out_channels,
            reduction=cfg.coord_att_reduction
        )
    
    def _build_transition_stage(self):
        """Build CNN-to-Transformer transition stage."""
        cfg = self.config
        
        # Patch Embedding: (B, C, H, W) → (B, N, D)
        self.patch_embed = PatchEmbedding(
            in_channels=cfg.fused_ir_out_channels,
            embed_dim=cfg.embed_dim,
            patch_size=cfg.patch_size
        )
        
        # Positional Encoding: adds position information
        self.pos_enc = PositionalEncoding(
            embed_dim=cfg.embed_dim,
            max_len=cfg.max_seq_len
        )
    
    def _build_transformer_stage(self):
        """Build transformer processing stage."""
        cfg = self.config
        
        # Linear Differential Attention
        self.lda = LinearDifferentialAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.lda_dropout,
            init=cfg.lda_init
        )
        
        # Residual LayerNorm after attention
        self.res_ln = ResidualLayerNormBlock(
            embed_dim=cfg.embed_dim
        )
        
        # Bottleneck FFN
        self.ffn = BottleneckFFN(
            inp=cfg.embed_dim,
            oup=cfg.embed_dim,
            bottleneck_ratio=cfg.ffn_bottleneck_ratio,
            dropout=cfg.ffn_dropout
        )
    
    def _build_classifier_stage(self):
        """Build classification stage."""
        cfg = self.config
        
        # Global Average Pooling
        self.gap = GlobalAveragePooling()
        
        # Classification Head
        self.classifier = ClassifierHead(
            embed_dim=cfg.embed_dim,
            num_classes=cfg.num_classes,
            dropout=cfg.classifier_dropout
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MobilePlantViT.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Class probabilities of shape (B, num_classes)
        """
        # CNN Stage
        x = self.ghost_conv(x)
        x = self.fused_ir(x)
        x = self.coord_att(x)
        
        # Transition Stage
        x = self.patch_embed(x)
        x = self.pos_enc(x)
        
        # Transformer Stage
        residual = x
        x = self.lda(x)
        x = self.res_ln(x, residual=residual)
        x = self.ffn(x)
        
        # Classifier Stage
        x = self.gap(x)
        x = self.classifier(x)
        
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning features before classifier.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Feature vector of shape (B, embed_dim)
        """
        # CNN Stage
        x = self.ghost_conv(x)
        x = self.fused_ir(x)
        x = self.coord_att(x)
        
        # Transition Stage
        x = self.patch_embed(x)
        x = self.pos_enc(x)
        
        # Transformer Stage
        residual = x
        x = self.lda(x)
        x = self.res_ln(x, residual=residual)
        x = self.ffn(x)
        
        # Global pooling only
        x = self.gap(x)
        
        return x
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits (for CrossEntropyLoss).
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logit tensor of shape (B, num_classes)
        """
        features = self.forward_features(x)
        return self.classifier.get_logits(features)
    
    def get_intermediate_outputs(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning all intermediate outputs for debugging.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Dictionary with intermediate tensors at each stage.
        """
        outputs = {'input': x}
        
        # CNN Stage
        x = self.ghost_conv(x)
        outputs['after_ghost_conv'] = x
        
        x = self.fused_ir(x)
        outputs['after_fused_ir'] = x
        
        x = self.coord_att(x)
        outputs['after_coord_att'] = x
        
        # Transition Stage
        x = self.patch_embed(x)
        outputs['after_patch_embed'] = x
        
        x = self.pos_enc(x)
        outputs['after_pos_enc'] = x
        
        # Transformer Stage
        residual = x
        x = self.lda(x)
        outputs['after_lda'] = x
        
        x = self.res_ln(x, residual=residual)
        outputs['after_res_ln'] = x
        
        x = self.ffn(x)
        outputs['after_ffn'] = x
        
        # Classifier Stage
        x = self.gap(x)
        outputs['after_gap'] = x
        
        x = self.classifier(x)
        outputs['output'] = x
        
        return outputs
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters.
        
        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_parameter_breakdown(self) -> Dict[str, int]:
        """
        Get parameter count breakdown by component.
        
        Returns:
            Dictionary mapping component names to parameter counts.
        """
        breakdown = {}
        
        # CNN Stage
        breakdown['ghost_conv'] = sum(p.numel() for p in self.ghost_conv.parameters())
        breakdown['fused_ir'] = sum(p.numel() for p in self.fused_ir.parameters())
        breakdown['coord_att'] = sum(p.numel() for p in self.coord_att.parameters())
        breakdown['cnn_total'] = breakdown['ghost_conv'] + breakdown['fused_ir'] + breakdown['coord_att']
        
        # Transition Stage
        breakdown['patch_embed'] = sum(p.numel() for p in self.patch_embed.parameters())
        breakdown['pos_enc'] = sum(p.numel() for p in self.pos_enc.parameters())
        breakdown['transition_total'] = breakdown['patch_embed'] + breakdown['pos_enc']
        
        # Transformer Stage
        breakdown['lda'] = sum(p.numel() for p in self.lda.parameters())
        breakdown['res_ln'] = sum(p.numel() for p in self.res_ln.parameters())
        breakdown['ffn'] = sum(p.numel() for p in self.ffn.parameters())
        breakdown['transformer_total'] = breakdown['lda'] + breakdown['res_ln'] + breakdown['ffn']
        
        # Classifier Stage
        breakdown['gap'] = sum(p.numel() for p in self.gap.parameters())
        breakdown['classifier'] = sum(p.numel() for p in self.classifier.parameters())
        breakdown['classifier_total'] = breakdown['gap'] + breakdown['classifier']
        
        # Total
        breakdown['total'] = self.count_parameters(trainable_only=False)
        
        return breakdown
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration as dictionary."""
        return self.config.to_dict()
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'MobilePlantViT':
        """Create model from configuration dictionary."""
        config = MobilePlantViTConfig.from_dict(config_dict)
        return cls(config)
    
    def extra_repr(self) -> str:
        """String representation with key configuration."""
        return (
            f"img_size={self.config.img_size}, "
            f"num_classes={self.config.num_classes}, "
            f"embed_dim={self.config.embed_dim}, "
            f"num_patches={self.num_patches}, "
            f"params={self.count_parameters():,}"
        )


# ============================================================================
# Model Variants (Factory Functions)
# ============================================================================

def mobileplant_vit_tiny(num_classes: int = 38, **kwargs) -> MobilePlantViT:
    """
    MobilePlantViT-Tiny: ~250K parameters.
    
    Suitable for edge devices with very limited resources.
    
    Args:
        num_classes: Number of output classes.
        **kwargs: Additional config overrides.
    
    Returns:
        MobilePlantViT model instance.
    """
    config = MobilePlantViTConfig(
        num_classes=num_classes,
        ghost_out_channels=32,
        fused_ir_out_channels=32,
        embed_dim=128,
        num_heads=4,
        ffn_bottleneck_ratio=0.25,
        **kwargs
    )
    return MobilePlantViT(config)


def mobileplant_vit_small(num_classes: int = 38, **kwargs) -> MobilePlantViT:
    """
    MobilePlantViT-Small: ~500K parameters.
    
    Good balance for mobile deployment.
    
    Args:
        num_classes: Number of output classes.
        **kwargs: Additional config overrides.
    
    Returns:
        MobilePlantViT model instance.
    """
    config = MobilePlantViTConfig(
        num_classes=num_classes,
        ghost_out_channels=48,
        fused_ir_out_channels=48,
        embed_dim=192,
        num_heads=6,
        ffn_bottleneck_ratio=0.25,
        **kwargs
    )
    return MobilePlantViT(config)


def mobileplant_vit_base(num_classes: int = 38, **kwargs) -> MobilePlantViT:
    """
    MobilePlantViT-Base: ~867K parameters (default).
    
    Default configuration for plant disease classification.
    
    Args:
        num_classes: Number of output classes.
        **kwargs: Additional config overrides.
    
    Returns:
        MobilePlantViT model instance.
    """
    config = MobilePlantViTConfig(
        num_classes=num_classes,
        ghost_out_channels=64,
        fused_ir_out_channels=64,
        embed_dim=256,
        num_heads=8,
        ffn_bottleneck_ratio=0.25,
        **kwargs
    )
    return MobilePlantViT(config)


def mobileplant_vit_large(num_classes: int = 38, **kwargs) -> MobilePlantViT:
    """
    MobilePlantViT-Large: ~2M parameters.
    
    Higher capacity for more complex datasets or server deployment.
    
    Args:
        num_classes: Number of output classes.
        **kwargs: Additional config overrides.
    
    Returns:
        MobilePlantViT model instance.
    """
    config = MobilePlantViTConfig(
        num_classes=num_classes,
        ghost_out_channels=96,
        fused_ir_out_channels=96,
        embed_dim=384,
        num_heads=12,
        ffn_bottleneck_ratio=0.25,
        **kwargs
    )
    return MobilePlantViT(config)