"""
Standalone test runner for GhostConv tests.
Run this file directly to execute all GhostConv tests with detailed output.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import math

def run_all_tests():
    """Run all GhostConv tests manually with detailed output."""
    
    print("=" * 70)
    print("GhostConv Comprehensive Test Suite")
    print("=" * 70)
    
    from src.blocks import GhostConv
    from src.utils.testing import (
        check_output_shape,
        check_gradient_flow,
        check_no_nan_inf,
        count_parameters,
    )
    
    all_passed = True
    test_count = 0
    pass_count = 0
    
    # ========== Shape Tests ==========
    print("\n" + "=" * 50)
    print("SHAPE TESTS")
    print("=" * 50)
    
    shape_tests = [
        ("Basic (3→64)", {"inp": 3, "oup": 64}, (2, 3, 224, 224), (2, 64, 224, 224)),
        ("Expansion (3→128)", {"inp": 3, "oup": 128}, (2, 3, 224, 224), (2, 128, 224, 224)),
        ("Reduction (256→64)", {"inp": 256, "oup": 64}, (2, 256, 56, 56), (2, 64, 56, 56)),
        ("Same (64→64)", {"inp": 64, "oup": 64}, (2, 64, 56, 56), (2, 64, 56, 56)),
        ("Small spatial", {"inp": 64, "oup": 64}, (2, 64, 7, 7), (2, 64, 7, 7)),
        ("Rectangular", {"inp": 64, "oup": 128}, (2, 64, 56, 112), (2, 128, 56, 112)),
    ]
    
    for name, params, input_shape, expected_shape in shape_tests:
        test_count += 1
        try:
            ghost = GhostConv(**params)
            x = torch.randn(*input_shape)
            check_output_shape(ghost, x, expected_shape)
            print(f"  ✅ {name}: {input_shape} → {expected_shape}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Stride Tests ==========
    print("\n" + "=" * 50)
    print("STRIDE TESTS")
    print("=" * 50)
    
    stride_tests = [
        ("Stride=1", {"inp": 64, "oup": 64, "stride": 1}, (2, 64, 56, 56), (2, 64, 56, 56)),
        ("Stride=2", {"inp": 64, "oup": 64, "stride": 2}, (2, 64, 56, 56), (2, 64, 28, 28)),
        ("Stride=4", {"inp": 64, "oup": 128, "stride": 4}, (2, 64, 224, 224), (2, 128, 56, 56)),
    ]
    
    for name, params, input_shape, expected_shape in stride_tests:
        test_count += 1
        try:
            ghost = GhostConv(**params)
            x = torch.randn(*input_shape)
            check_output_shape(ghost, x, expected_shape)
            print(f"  ✅ {name}: {input_shape} → {expected_shape}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Ratio Tests ==========
    print("\n" + "=" * 50)
    print("RATIO TESTS")
    print("=" * 50)

    ratio_tests = [
        ("Ratio=1 (no ghost)", 1, 64, 0, True),   # has_cheap_op = False
        ("Ratio=2", 2, 32, 32, False),
        ("Ratio=4", 4, 16, 48, False),
    ]

    for name, ratio, expected_init, expected_new, no_cheap_op in ratio_tests:
        test_count += 1
        try:
            ghost = GhostConv(inp=3, oup=64, ratio=ratio)
            x = torch.randn(2, 3, 56, 56)
            y = ghost(x)
            
            assert ghost.init_channels == expected_init, \
                f"init_channels={ghost.init_channels}, expected {expected_init}"
            assert ghost.new_channels == expected_new, \
                f"new_channels={ghost.new_channels}, expected {expected_new}"
            
            if no_cheap_op:
                assert ghost.cheap_op is None, "cheap_op should be None for ratio=1"
            else:
                assert ghost.cheap_op is not None, "cheap_op should exist"
            
            assert y.shape == (2, 64, 56, 56), f"Output shape {y.shape}"
            
            print(f"  ✅ {name}: init={ghost.init_channels}, new={ghost.new_channels}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Gradient Tests ==========
    print("\n" + "=" * 50)
    print("GRADIENT TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        ghost = GhostConv(inp=3, oup=64)
        x = torch.randn(2, 3, 56, 56)
        grad_info = check_gradient_flow(ghost, x)
        
        all_have_grad = all(grad_info.values())
        if all_have_grad:
            print(f"  ✅ Gradient flow: All {len(grad_info)} parameters receive gradients")
            pass_count += 1
        else:
            missing = [k for k, v in grad_info.items() if not v]
            print(f"  ❌ Gradient flow: Missing gradients for {missing}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Gradient flow: {e}")
        all_passed = False
    
    # Backward pass test
    test_count += 1
    try:
        ghost = GhostConv(inp=3, oup=64)
        x = torch.randn(2, 3, 56, 56, requires_grad=True)
        y = ghost(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None, "Input gradient is None"
        print(f"  ✅ Backward pass: Completes without error")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Backward pass: {e}")
        all_passed = False
    
    # ========== Stability Tests ==========
    print("\n" + "=" * 50)
    print("STABILITY TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        ghost = GhostConv(inp=3, oup=64)
        ghost.eval()
        x = torch.randn(2, 3, 56, 56)
        
        with torch.no_grad():
            y = ghost(x)
        
        check_no_nan_inf(y, "output")
        print(f"  ✅ No NaN/Inf: Output is numerically stable")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ No NaN/Inf: {e}")
        all_passed = False
    
    # Deterministic output test
    test_count += 1
    try:
        ghost = GhostConv(inp=3, oup=64)
        ghost.eval()
        x = torch.randn(2, 3, 56, 56)
        
        with torch.no_grad():
            y1 = ghost(x).clone()
            y2 = ghost(x).clone()
        
        assert torch.allclose(y1, y2), "Outputs differ"
        print(f"  ✅ Deterministic: Same input → same output (eval mode)")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Deterministic: {e}")
        all_passed = False
    
    # ========== Parameter Count ==========
    print("\n" + "=" * 50)
    print("PARAMETER COUNT")
    print("=" * 50)
    
    configs = [
        ("3→64, ratio=2", {"inp": 3, "oup": 64, "ratio": 2}),
        ("64→128, ratio=2", {"inp": 64, "oup": 128, "ratio": 2}),
        ("64→128, ratio=4", {"inp": 64, "oup": 128, "ratio": 4}),
    ]
    
    for name, params in configs:
        ghost = GhostConv(**params)
        p = count_parameters(ghost)
        print(f"  {name}: {p['total']:,} parameters")
    
    # ========== Benchmark ==========
    print("\n" + "=" * 50)
    print("BENCHMARK (batch=2, 3→64, 224×224)")
    print("=" * 50)
    
    import time
    
    ghost = GhostConv(inp=3, oup=64)
    ghost.eval()
    x = torch.randn(2, 3, 224, 224)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = ghost(x)
    
    # Forward timing
    forward_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = ghost(x)
        forward_times.append((time.perf_counter() - start) * 1000)
    
    forward_mean = sum(forward_times) / len(forward_times)
    print(f"  Forward time:  {forward_mean:.3f} ms")
    
    # Backward timing
    ghost.train()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    
    backward_times = []
    for _ in range(20):
        ghost.zero_grad()
        y = ghost(x)
        start = time.perf_counter()
        y.sum().backward()
        backward_times.append((time.perf_counter() - start) * 1000)
    
    backward_mean = sum(backward_times) / len(backward_times)
    print(f"  Backward time: {backward_mean:.3f} ms")
    print(f"  Total time:    {forward_mean + backward_mean:.3f} ms")
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {pass_count}/{test_count} tests passed")
    print("=" * 70)
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)