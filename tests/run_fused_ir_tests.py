"""
Standalone test runner for Fused Inverted Residual tests.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch


def run_all_tests():
    """Run all FusedIR tests manually with detailed output."""
    
    print("=" * 70)
    print("Fused Inverted Residual Comprehensive Test Suite")
    print("=" * 70)
    
    from src.blocks import FusedInvertedResidualBlock
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
        ("Same channels, stride=1", {"inp": 64, "oup": 64, "stride": 1}, (2, 64, 56, 56), (2, 64, 56, 56)),
        ("Same channels, stride=2", {"inp": 64, "oup": 64, "stride": 2}, (2, 64, 56, 56), (2, 64, 28, 28)),
        ("Channel increase, stride=1", {"inp": 64, "oup": 128, "stride": 1}, (2, 64, 56, 56), (2, 128, 56, 56)),
        ("Channel increase, stride=2", {"inp": 64, "oup": 128, "stride": 2}, (2, 64, 56, 56), (2, 128, 28, 28)),
        ("Channel decrease", {"inp": 128, "oup": 64, "stride": 1}, (2, 128, 56, 56), (2, 64, 56, 56)),
        ("Stride=4 downsample", {"inp": 64, "oup": 128, "stride": 4}, (2, 64, 224, 224), (2, 128, 56, 56)),
    ]
    
    for name, params, input_shape, expected_shape in shape_tests:
        test_count += 1
        try:
            fused_ir = FusedInvertedResidualBlock(**params)
            x = torch.randn(*input_shape)
            check_output_shape(fused_ir, x, expected_shape)
            print(f"  ✅ {name}: {input_shape} → {expected_shape}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Residual Tests ==========
    print("\n" + "=" * 50)
    print("RESIDUAL CONNECTION TESTS")
    print("=" * 50)
    
    residual_tests = [
        ("inp=64, oup=64, stride=1", 64, 64, 1, True),
        ("inp=64, oup=64, stride=2", 64, 64, 2, False),
        ("inp=64, oup=128, stride=1", 64, 128, 1, False),
        ("inp=64, oup=128, stride=2", 64, 128, 2, False),
    ]
    
    for name, inp, oup, stride, expected_res in residual_tests:
        test_count += 1
        try:
            fused_ir = FusedInvertedResidualBlock(inp=inp, oup=oup, stride=stride)
            assert fused_ir.use_res_connect == expected_res, \
                f"Expected {expected_res}, got {fused_ir.use_res_connect}"
            
            res_info = fused_ir.get_residual_info()
            print(f"  ✅ {name}: use_res={fused_ir.use_res_connect} ({res_info['reason']})")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Expansion Tests ==========
    print("\n" + "=" * 50)
    print("EXPANSION RATIO TESTS")
    print("=" * 50)
    
    expansion_tests = [
        ("expand_ratio=1", 1, 64),
        ("expand_ratio=2", 2, 128),
        ("expand_ratio=4", 4, 256),
        ("expand_ratio=6", 6, 384),
    ]
    
    for name, expand_ratio, expected_hidden in expansion_tests:
        test_count += 1
        try:
            fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, expand_ratio=expand_ratio)
            x = torch.randn(2, 64, 28, 28)
            y = fused_ir(x)
            
            assert fused_ir.hidden_dim == expected_hidden, \
                f"Expected hidden_dim={expected_hidden}, got {fused_ir.hidden_dim}"
            assert y.shape == (2, 64, 28, 28), f"Shape mismatch: {y.shape}"
            
            print(f"  ✅ {name}: hidden_dim={fused_ir.hidden_dim}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Ghost Option Tests ==========
    print("\n" + "=" * 50)
    print("GHOST OPTION TESTS")
    print("=" * 50)
    
    ghost_tests = [
        ("Standard conv", False, (2, 64, 56, 56), (2, 64, 56, 56)),
        ("GhostConv", True, (2, 64, 56, 56), (2, 64, 56, 56)),
        ("GhostConv stride=2", True, (2, 64, 56, 56), (2, 128, 28, 28)),
    ]
    
    for name, use_ghost, input_shape, expected_shape in ghost_tests:
        test_count += 1
        try:
            if "stride=2" in name:
                fused_ir = FusedInvertedResidualBlock(
                    inp=64, oup=128, stride=2, use_ghost=use_ghost
                )
            else:
                fused_ir = FusedInvertedResidualBlock(
                    inp=64, oup=64, stride=1, use_ghost=use_ghost
                )
            
            x = torch.randn(*input_shape)
            y = fused_ir(x)
            
            assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
            print(f"  ✅ {name}: {input_shape} → {y.shape}")
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
        fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, stride=1)
        x = torch.randn(2, 64, 28, 28)
        grad_info = check_gradient_flow(fused_ir, x)
        
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
    
    # Residual gradient test
    test_count += 1
    try:
        fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, stride=1)
        x = torch.randn(2, 64, 28, 28, requires_grad=True)
        y = fused_ir(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None, "Input gradient is None"
        assert x.grad.abs().sum() > 0, "Input gradient is zero"
        print(f"  ✅ Residual gradient: Input receives non-zero gradient")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Residual gradient: {e}")
        all_passed = False
    
    # ========== Stability Tests ==========
    print("\n" + "=" * 50)
    print("STABILITY TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, stride=1)
        fused_ir.eval()
        x = torch.randn(2, 64, 28, 28)
        
        with torch.no_grad():
            y = fused_ir(x)
        
        check_no_nan_inf(y, "output")
        print(f"  ✅ No NaN/Inf: Output is numerically stable")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ No NaN/Inf: {e}")
        all_passed = False
    
    # Deterministic test
    test_count += 1
    try:
        fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, stride=1)
        fused_ir.eval()
        x = torch.randn(2, 64, 28, 28)
        
        with torch.no_grad():
            y1 = fused_ir(x).clone()
            y2 = fused_ir(x).clone()
        
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
        ("64→64, expand=4, standard", {"inp": 64, "oup": 64, "expand_ratio": 4}),
        ("64→128, expand=4, standard", {"inp": 64, "oup": 128, "expand_ratio": 4}),
        ("64→64, expand=4, ghost", {"inp": 64, "oup": 64, "expand_ratio": 4, "use_ghost": True}),
        ("64→64, expand=1", {"inp": 64, "oup": 64, "expand_ratio": 1}),
    ]
    
    for name, params in configs:
        fused_ir = FusedInvertedResidualBlock(**params)
        p = count_parameters(fused_ir)
        print(f"  {name}: {p['total']:,} parameters")
    
    # ========== Benchmark ==========
    print("\n" + "=" * 50)
    print("BENCHMARK (batch=2, 64→64, 56×56, expand=4)")
    print("=" * 50)
    
    import time
    
    fused_ir = FusedInvertedResidualBlock(inp=64, oup=64, stride=1, expand_ratio=4)
    fused_ir.eval()
    x = torch.randn(2, 64, 56, 56)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = fused_ir(x)
    
    # Forward timing
    forward_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = fused_ir(x)
        forward_times.append((time.perf_counter() - start) * 1000)
    
    forward_mean = sum(forward_times) / len(forward_times)
    print(f"  Forward time:  {forward_mean:.3f} ms")
    
    # Backward timing
    fused_ir.train()
    x = torch.randn(2, 64, 56, 56, requires_grad=True)
    
    backward_times = []
    for _ in range(20):
        fused_ir.zero_grad()
        y = fused_ir(x)
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