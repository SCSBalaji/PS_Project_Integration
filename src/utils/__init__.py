"""
Utility functions for testing and benchmarking.
"""

from .testing import (
    check_output_shape,
    check_gradient_flow,
    check_no_nan_inf,
    count_parameters,
    memory_check,
    numeric_gradient_check,
)

from .benchmarking import (
    time_forward_backward,
    estimate_flops,
    measure_memory_usage,
    warmup_and_benchmark,
)

__all__ = [
    # Testing utilities
    'check_output_shape',
    'check_gradient_flow',
    'check_no_nan_inf',
    'count_parameters',
    'memory_check',
    'numeric_gradient_check',
    # Benchmarking utilities
    'time_forward_backward',
    'estimate_flops',
    'measure_memory_usage',
    'warmup_and_benchmark',
]