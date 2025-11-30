"""
Benchmarking utility functions for measuring neural network performance.

This module provides helper functions for:
- Forward/backward pass timing
- FLOP estimation
- Memory usage measurement
- Proper GPU benchmarking with warmup
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Callable
import time
import gc


def time_forward_backward(
    module: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure forward and backward pass times.
    
    Args:
        module: PyTorch module to benchmark
        input_tensor: Input tensor
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs (not timed)
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        Dictionary with timing statistics in milliseconds
    
    Example:
        >>> linear = nn.Linear(1000, 1000)
        >>> x = torch.randn(32, 1000)
        >>> times = time_forward_backward(linear, x, num_runs=50)
        >>> print(f"Forward: {times['forward_mean_ms']:.2f} ms")
    """
    module = module.to(device)
    module.train()
    input_tensor = input_tensor.to(device)
    
    forward_times = []
    backward_times = []
    
    # Warmup
    for _ in range(warmup_runs):
        module.zero_grad()
        x = input_tensor.clone().requires_grad_(True)
        y = module(x)
        loss = y.sum()
        loss.backward()
        
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # Timed runs
    for _ in range(num_runs):
        module.zero_grad()
        x = input_tensor.clone().requires_grad_(True)
        
        # Forward timing
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        y = module(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)  # ms
        
        # Backward timing
        loss = y.sum()
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)  # ms
    
    return {
        'forward_mean_ms': sum(forward_times) / len(forward_times),
        'forward_std_ms': (sum((t - sum(forward_times)/len(forward_times))**2 for t in forward_times) / len(forward_times)) ** 0.5,
        'forward_min_ms': min(forward_times),
        'forward_max_ms': max(forward_times),
        'backward_mean_ms': sum(backward_times) / len(backward_times),
        'backward_std_ms': (sum((t - sum(backward_times)/len(backward_times))**2 for t in backward_times) / len(backward_times)) ** 0.5,
        'backward_min_ms': min(backward_times),
        'backward_max_ms': max(backward_times),
        'total_mean_ms': sum(forward_times) / len(forward_times) + sum(backward_times) / len(backward_times),
        'num_runs': num_runs,
    }


def estimate_flops(
    module: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
) -> Dict[str, int]:
    """
    Estimate FLOPs for a module using a simple analysis.
    
    Note: This is a rough estimate. For accurate counts, use tools like fvcore or thop.
    
    Args:
        module: PyTorch module
        input_shape: Shape of input tensor (including batch dimension)
        device: Device to run on
    
    Returns:
        Dictionary with FLOP estimates
    
    Example:
        >>> linear = nn.Linear(1000, 1000)
        >>> flops = estimate_flops(linear, (1, 1000))
        >>> print(f"Estimated FLOPs: {flops['total']:,}")
    """
    total_flops = 0
    flops_by_type = {}
    
    def count_conv2d(m, x, y):
        nonlocal total_flops
        # FLOPs = 2 * Cout * Hout * Wout * Cin * Kh * Kw / groups
        batch_size = x[0].shape[0]
        out_h, out_w = y.shape[2], y.shape[3]
        kernel_ops = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        flops = 2 * batch_size * m.out_channels * out_h * out_w * kernel_ops
        total_flops += flops
        flops_by_type['Conv2d'] = flops_by_type.get('Conv2d', 0) + flops
    
    def count_linear(m, x, y):
        nonlocal total_flops
        # FLOPs = 2 * batch * in_features * out_features
        batch_size = x[0].shape[0]
        flops = 2 * batch_size * m.in_features * m.out_features
        total_flops += flops
        flops_by_type['Linear'] = flops_by_type.get('Linear', 0) + flops
    
    def count_bn(m, x, y):
        nonlocal total_flops
        # BN: 4 ops per element (mean, var, normalize, scale+shift)
        flops = 4 * x[0].numel()
        total_flops += flops
        flops_by_type['BatchNorm'] = flops_by_type.get('BatchNorm', 0) + flops
    
    def count_ln(m, x, y):
        nonlocal total_flops
        # LN: 4 ops per element
        flops = 4 * x[0].numel()
        total_flops += flops
        flops_by_type['LayerNorm'] = flops_by_type.get('LayerNorm', 0) + flops
    
    hooks = []
    
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(count_conv2d))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(count_linear))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(count_bn))
        elif isinstance(m, nn.LayerNorm):
            hooks.append(m.register_forward_hook(count_ln))
    
    module = module.to(device)
    module.eval()
    x = torch.randn(*input_shape, device=device)
    
    with torch.no_grad():
        module(x)
    
    for hook in hooks:
        hook.remove()
    
    return {
        'total': total_flops,
        'by_type': flops_by_type,
        'total_gflops': total_flops / 1e9,
        'total_mflops': total_flops / 1e6,
    }


def measure_memory_usage(
    module: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Measure GPU memory usage during forward and backward passes.
    
    Args:
        module: PyTorch module
        input_shape: Shape of input tensor
        device: Device to run on (should be 'cuda' for meaningful results)
    
    Returns:
        Dictionary with memory usage in MB
    
    Note:
        Returns zeros for CPU device.
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return {
            'forward_memory_mb': 0,
            'backward_memory_mb': 0,
            'peak_memory_mb': 0,
            'model_memory_mb': 0,
        }
    
    module = module.to(device)
    module.train()
    
    # Clear cache and reset stats
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure model memory
    model_memory = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Create input
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    
    # Forward pass
    torch.cuda.reset_peak_memory_stats()
    y = module(x)
    forward_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Backward pass
    torch.cuda.reset_peak_memory_stats()
    loss = y.sum()
    loss.backward()
    backward_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Peak memory
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    y = module(x)
    loss = y.sum()
    loss.backward()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'forward_memory_mb': forward_memory,
        'backward_memory_mb': backward_memory,
        'peak_memory_mb': peak_memory,
        'model_memory_mb': model_memory,
    }


def warmup_and_benchmark(
    module: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 20,
    device: str = 'cpu',
    include_backward: bool = True
) -> Dict[str, float]:
    """
    Properly benchmark a module with warmup for GPU operations.
    
    Args:
        module: PyTorch module to benchmark
        input_tensor: Input tensor
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Device to run on
        include_backward: Whether to include backward pass timing
    
    Returns:
        Dictionary with comprehensive benchmark results
    """
    module = module.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup phase
    module.train()
    for _ in range(warmup_runs):
        module.zero_grad()
        x = input_tensor.clone().requires_grad_(True)
        y = module(x)
        if include_backward:
            y.sum().backward()
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark forward only
    module.eval()
    forward_only_times = []
    for _ in range(num_runs):
        x = input_tensor.clone()
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            y = module(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        forward_only_times.append((time.perf_counter() - start) * 1000)
    
    result = {
        'forward_only_mean_ms': sum(forward_only_times) / len(forward_only_times),
        'forward_only_std_ms': (sum((t - sum(forward_only_times)/len(forward_only_times))**2 for t in forward_only_times) / len(forward_only_times)) ** 0.5,
        'device': device,
        'num_runs': num_runs,
        'warmup_runs': warmup_runs,
    }
    
    # Benchmark with backward if requested
    if include_backward:
        timing_result = time_forward_backward(
            module, input_tensor, num_runs=num_runs, warmup_runs=0, device=device
        )
        result.update({
            'forward_with_grad_mean_ms': timing_result['forward_mean_ms'],
            'backward_mean_ms': timing_result['backward_mean_ms'],
            'total_mean_ms': timing_result['total_mean_ms'],
        })
    
    # Add throughput calculation
    batch_size = input_tensor.shape[0]
    result['throughput_samples_per_sec'] = batch_size / (result['forward_only_mean_ms'] / 1000)
    
    return result