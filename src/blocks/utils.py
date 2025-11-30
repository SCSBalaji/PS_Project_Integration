"""
Utility classes and functions for neural network blocks.

This module provides:
- BaseBlock class with common functionality
- Shape assertion helpers
- Debug mode support
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


class BaseBlock(nn.Module):
    """
    Base class for all neural network blocks with common functionality.
    
    Features:
    - Debug mode for printing intermediate shapes
    - Consistent initialization patterns
    - Parameter counting method
    - Shape assertion helpers
    
    Example:
        class MyBlock(BaseBlock):
            def __init__(self, in_channels, out_channels, debug=False):
                super().__init__(debug=debug)
                self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            
            def forward(self, x):
                self._debug_print("Input", x)
                x = self.conv(x)
                self._debug_print("After conv", x)
                return x
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the base block.
        
        Args:
            debug: If True, print intermediate tensor shapes during forward pass
        """
        super().__init__()
        self._debug = debug
        self._forward_count = 0
    
    def _debug_print(self, name: str, tensor: torch.Tensor) -> None:
        """
        Print tensor shape if debug mode is enabled.
        
        Args:
            name: Name/description of the tensor
            tensor: The tensor to print info about
        """
        if self._debug:
            print(f"  [{self.__class__.__name__}] {name}: {tuple(tensor.shape)}")
    
    def set_debug(self, debug: bool) -> None:
        """
        Enable or disable debug mode.
        
        Args:
            debug: New debug mode setting
        """
        self._debug = debug
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and total parameters.
        
        Returns:
            Dictionary with 'total', 'trainable', and 'non_trainable' counts
        """
        total = 0
        trainable = 0
        
        for param in self.parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get block configuration. Override in subclasses.
        
        Returns:
            Dictionary with block configuration
        """
        return {'class': self.__class__.__name__}
    
    @staticmethod
    def assert_shape(
        tensor: torch.Tensor,
        expected_shape: Tuple[Optional[int], ...],
        name: str = "tensor"
    ) -> None:
        """
        Assert tensor matches expected shape. Use None for dynamic dimensions.
        
        Args:
            tensor: Tensor to check
            expected_shape: Expected shape (None for any value)
            name: Name for error message
        
        Raises:
            AssertionError: If shape doesn't match
        
        Example:
            # Check that tensor is (batch, 64, any, any)
            BaseBlock.assert_shape(x, (None, 64, None, None), "feature_map")
        """
        actual_shape = tuple(tensor.shape)
        
        if len(actual_shape) != len(expected_shape):
            raise AssertionError(
                f"{name}: Expected {len(expected_shape)} dimensions, "
                f"got {len(actual_shape)} with shape {actual_shape}"
            )
        
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected is not None and actual != expected:
                raise AssertionError(
                    f"{name}: Dimension {i} expected {expected}, got {actual}. "
                    f"Full shape: {actual_shape}, expected: {expected_shape}"
                )
    
    @staticmethod
    def assert_channels(
        tensor: torch.Tensor,
        expected_channels: int,
        dim: int = 1,
        name: str = "tensor"
    ) -> None:
        """
        Assert tensor has expected number of channels.
        
        Args:
            tensor: Tensor to check
            expected_channels: Expected number of channels
            dim: Dimension where channels are (default 1 for BCHW format)
            name: Name for error message
        """
        actual_channels = tensor.shape[dim]
        if actual_channels != expected_channels:
            raise AssertionError(
                f"{name}: Expected {expected_channels} channels, got {actual_channels}"
            )


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Make a value divisible by a given number.
    
    This is commonly used for channel counts in mobile architectures to ensure
    efficient memory access patterns.
    
    Args:
        v: Original value
        divisor: Number to make divisible by
        min_value: Minimum value to return
    
    Returns:
        Adjusted value that is divisible by divisor
    
    Example:
        >>> make_divisible(70, 8)
        72
        >>> make_divisible(10, 8)
        16
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_activation(activation: str) -> nn.Module:
    """
    Get activation module by name.
    
    Args:
        activation: Name of activation ('relu', 'relu6', 'gelu', 'silu', 'none')
    
    Returns:
        Activation module
    
    Raises:
        ValueError: If activation name is not recognized
    """
    activations = {
        'relu': nn.ReLU(inplace=True),
        'relu6': nn.ReLU6(inplace=True),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(inplace=True),
        'swish': nn.SiLU(inplace=True),
        'none': nn.Identity(),
        'identity': nn.Identity(),
    }
    
    activation_lower = activation.lower()
    if activation_lower not in activations:
        raise ValueError(f"Unknown activation: {activation}. "
                        f"Available: {list(activations.keys())}")
    
    return activations[activation_lower]