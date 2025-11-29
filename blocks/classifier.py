"""
Classifier Head and Global Average Pooling modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAveragePooling(nn.Module):
    """
    Global Average Pooling over the sequence length dimension.
    Input shape: (batch, seq_len, embed_dim)
    Output shape: (batch, embed_dim)
    """
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class ClassifierHead(nn.Module):
    """
    Final classification block with linear layer and softmax activation.
    
    Args:
        embed_dim (int): Input feature dimension.
        num_classes (int): Number of output classes.
    """
    def __init__(self, embed_dim: int, num_classes: int):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        return probs