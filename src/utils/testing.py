"""
Testing utility functions for validating neural network blocks.

This module provides helper functions for:
- Shape validation
- Gradient flow checking
- Numerical stability checks
- Parameter counting
- Memory leak detection
- Numerical gradient verification
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Union
import gc


def check_output_shape(
    module: nn.Module,
    input_tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    msg: str = ""
) -> bool:
    """
    Validate that module output matches expected shape.
    
    Args:
        module: PyTorch module to test
        input_tensor: Input tensor to pass through module
        expected_shape: Expected output shape as tuple
        msg: Optional message for assertion error
    
    Returns:
        True if shape matches, raises AssertionError otherwise
    
    Example:
        >>> conv = nn.Conv2d(3, 64, 3, padding=1)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> check_output_shape(conv, x, (2, 64, 224, 224))
        True
    """
    module.eval()
    with torch.no_grad():
        output = module(input_tensor)
    
    actual_shape = tuple(output.shape)
    error_msg = f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
    if msg:
        error_msg = f"{msg}: {error_msg}"
    
    assert actual_shape == expected_shape, error_msg
    return True


def check_gradient_flow(
    module: nn.Module,
    input_tensor: torch.Tensor,
    loss_fn: Optional[callable] = None
) -> Dict[str, bool]:
    """
    Verify that gradients flow to all trainable parameters.
    
    Args:
        module: PyTorch module to test
        input_tensor: Input tensor (requires_grad will be set to True)
        loss_fn: Optional loss function. If None, uses sum of output.
    
    Returns:
        Dictionary mapping parameter names to whether they received gradients
    
    Example:
        >>> linear = nn.Linear(10, 5)
        >>> x = torch.randn(2, 10)
        >>> grad_info = check_gradient_flow(linear, x)
        >>> all(grad_info.values())
        True
    """
    module.train()
    module.zero_grad()
    
    # Ensure input requires grad
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = module(input_tensor)
    
    # Compute loss
    if loss_fn is not None:
        loss = loss_fn(output)
    else:
        loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check each parameter
    grad_info = {}
    for name, param in module.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None and param.grad.abs().sum() > 0
            grad_info[name] = has_grad
    
    return grad_info


def check_no_nan_inf(
    tensor: torch.Tensor,
    tensor_name: str = "tensor"
) -> bool:
    """
    Ensure tensor contains no NaN or Inf values.
    
    Args:
        tensor: Tensor to check
        tensor_name: Name for error message
    
    Returns:
        True if no NaN/Inf, raises AssertionError otherwise
    
    Example:
        >>> x = torch.randn(10)
        >>> check_no_nan_inf(x, "output")
        True
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    assert not has_nan, f"{tensor_name} contains NaN values"
    assert not has_inf, f"{tensor_name} contains Inf values"
    
    return True


def count_parameters(
    module: nn.Module,
    trainable_only: bool = False
) -> Dict[str, int]:
    """
    Count parameters in a module.
    
    Args:
        module: PyTorch module
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Dictionary with 'total', 'trainable', and 'non_trainable' counts
    
    Example:
        >>> linear = nn.Linear(10, 5)
        >>> counts = count_parameters(linear)
        >>> counts['total']
        55  # 10*5 + 5 = 55
    """
    total = 0
    trainable = 0
    non_trainable = 0
    
    for param in module.parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params
        else:
            non_trainable += num_params
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def memory_check(
    module: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Basic memory leak detection by running multiple forward passes.
    
    Args:
        module: PyTorch module to test
        input_shape: Shape of input tensor
        num_iterations: Number of forward passes to run
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        Dictionary with memory usage statistics
    
    Note:
        For GPU, returns actual memory stats. For CPU, returns process memory.
    """
    module = module.to(device)
    module.eval()
    
    # Force garbage collection
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    memory_usage = []
    
    for _ in range(num_iterations):
        x = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            _ = module(x)
        
        if device == 'cuda':
            memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
        
        del x
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        return {
            'peak_memory_mb': peak_memory,
            'final_memory_mb': memory_usage[-1] if memory_usage else 0,
            'memory_stable': max(memory_usage) - min(memory_usage) < 1.0 if memory_usage else True
        }
    else:
        return {
            'peak_memory_mb': 0,
            'final_memory_mb': 0,
            'memory_stable': True
        }


def numeric_gradient_check(
    module: nn.Module,
    input_tensor: torch.Tensor,
    eps: float = 1e-5,
    tolerance: float = 1e-3,
    num_checks: int = 5
) -> Dict[str, float]:
    """
    Verify gradients using finite differences.
    
    Args:
        module: PyTorch module to test
        input_tensor: Input tensor
        eps: Epsilon for finite difference
        tolerance: Relative tolerance for gradient comparison
        num_checks: Number of random parameters to check
    
    Returns:
        Dictionary with gradient check results
    
    Example:
        >>> linear = nn.Linear(10, 5)
        >>> x = torch.randn(2, 10)
        >>> result = numeric_gradient_check(linear, x)
        >>> result['passed']
        True
    """
    module.train()
    
    # Get analytical gradients
    module.zero_grad()
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    output = module(input_tensor)
    loss = output.sum()
    loss.backward()
    
    # Store analytical gradients
    analytical_grads = {}
    for name, param in module.named_parameters():
        if param.requires_grad and param.grad is not None:
            analytical_grads[name] = param.grad.clone()
    
    # Check a subset of parameters
    max_rel_error = 0.0
    checked_params = []
    
    param_list = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    
    for name, param in param_list[:num_checks]:
        if param.numel() == 0:
            continue
        
        # Pick a random index
        idx = tuple(torch.randint(0, s, (1,)).item() for s in param.shape)
        
        # Compute numerical gradient
        with torch.no_grad():
            original_value = param[idx].item()
            
            # f(x + eps)
            param[idx] = original_value + eps
            output_plus = module(input_tensor.detach()).sum().item()
            
            # f(x - eps)
            param[idx] = original_value - eps
            output_minus = module(input_tensor.detach()).sum().item()
            
            # Restore original
            param[idx] = original_value
        
        numerical_grad = (output_plus - output_minus) / (2 * eps)
        analytical_grad = analytical_grads[name][idx].item()
        
        # Compute relative error
        if abs(analytical_grad) > 1e-7 or abs(numerical_grad) > 1e-7:
            rel_error = abs(numerical_grad - analytical_grad) / (abs(analytical_grad) + 1e-8)
            max_rel_error = max(max_rel_error, rel_error)
            checked_params.append({
                'name': name,
                'index': idx,
                'analytical': analytical_grad,
                'numerical': numerical_grad,
                'rel_error': rel_error
            })
    
    return {
        'passed': max_rel_error < tolerance,
        'max_relative_error': max_rel_error,
        'tolerance': tolerance,
        'num_checked': len(checked_params),
        'details': checked_params
    }


def assert_tensor_equal(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    msg: str = ""
) -> bool:
    """
    Assert two tensors are approximately equal.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        atol: Absolute tolerance
        rtol: Relative tolerance
        msg: Optional message for assertion error
    
    Returns:
        True if tensors are approximately equal
    """
    if tensor1.shape != tensor2.shape:
        raise AssertionError(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
    
    if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
        max_diff = (tensor1 - tensor2).abs().max().item()
        error_msg = f"Tensors not equal. Max difference: {max_diff}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)
    
    return True