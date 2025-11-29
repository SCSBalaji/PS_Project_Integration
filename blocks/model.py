"""
Main MobilePlantViT Model.
"""

import torch
import torch.nn as nn

from .ghost_conv import GhostConv
from .fused_ir import FusedInvertedResidualBlock
from .coord_att import CoordAtt
from .patch_embed import PatchEmbedding, PositionalEncoding
from .lda import LinearDifferentialAttention
from .res_norm import ResidualLayerNormBlock
from .bottleneck_ffn import BottleneckFFN
from .classifier import ClassifierHead, GlobalAveragePooling


class MobilePlantViTModel(nn.Module):
    """
    MobilePlantViT: A hybrid CNN-Transformer model for plant disease classification.
    
    Args:
        img_size (int): Input image size.
        num_classes (int): Number of output classes.
        ghost_conv_params (dict): Parameters for GhostConv.
        fused_ir_params (dict): Parameters for FusedInvertedResidualBlock.
        coord_att_params (dict): Parameters for CoordAtt.
        patch_embed_params (dict): Parameters for PatchEmbedding.
        pos_encoding_params (dict): Parameters for PositionalEncoding.
        lda_params (dict): Parameters for LinearDifferentialAttention.
        res_ln_params (dict): Parameters for ResidualLayerNormBlock.
        bottleneck_ffn_params (dict): Parameters for BottleneckFFN.
    """
    def __init__(self, 
                 img_size: int = 224,
                 num_classes: int = 38,
                 ghost_conv_params: dict = None,
                 fused_ir_params: dict = None,
                 coord_att_params: dict = None,
                 patch_embed_params: dict = None,
                 pos_encoding_params: dict = None,
                 lda_params: dict = None,
                 res_ln_params: dict = None,
                 bottleneck_ffn_params: dict = None):
        super(MobilePlantViTModel, self).__init__()
        
        # Default parameters
        ghost_conv_params = ghost_conv_params or {'inp': 3, 'oup': 64}
        fused_ir_params = fused_ir_params or {'inp': 64, 'oup': 64}
        coord_att_params = coord_att_params or {'inp': 64, 'oup': 64}
        patch_embed_params = patch_embed_params or {'in_channels': 64, 'embed_dim': 256, 'patch_size': 14}
        pos_encoding_params = pos_encoding_params or {'embed_dim': 256}
        lda_params = lda_params or {'embed_dim': 256, 'num_heads': 8}
        res_ln_params = res_ln_params or {'embed_dim': 256}
        bottleneck_ffn_params = bottleneck_ffn_params or {'inp': 256, 'oup': 256}
        
        # Build layers
        self.ghost_conv = GhostConv(**ghost_conv_params)
        self.fused_inverted_residual = FusedInvertedResidualBlock(**fused_ir_params)
        self.coord_attention = CoordAtt(**coord_att_params)
        self.patch_embedding = PatchEmbedding(**patch_embed_params)
        self.positional_encoding = PositionalEncoding(**pos_encoding_params)
        self.lda = LinearDifferentialAttention(**lda_params)
        self.res_layer_norm = ResidualLayerNormBlock(**res_ln_params)
        self.bottleneck_ffn = BottleneckFFN(**bottleneck_ffn_params)
        self.global_avg_pool = GlobalAveragePooling()
        self.classifier = ClassifierHead(
            embed_dim=bottleneck_ffn_params.get('oup', 256), 
            num_classes=num_classes
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ghost_conv(x)
        x = self.fused_inverted_residual(x)
        x = self.coord_attention(x)
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.lda(x)
        x = self.res_layer_norm(x)
        x = self.bottleneck_ffn(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x