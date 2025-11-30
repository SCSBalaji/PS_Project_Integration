"""
Standalone test runner for CoordAtt tests.
Run this file directly to execute all CoordAtt tests with detailed output.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch


def run_all_tests():
    """Run all CoordAtt tests manually with detailed output."""
    
    print("=" * 70)
    print("Coordinate Attention Comprehensive Test Suite")
    print("=" * 70)
    
    from src.blocks import CoordAtt, HSigmoid, HSwish
    from src.utils.testing import (
        check_output_shape,
        check_gradient_flow,
        check_no_nan_inf,
        count_parameters,
    )
    
    all_passed = True
    test_count = 0
    pass_count = 0
    
    # ========== Activation Tests ==========
    print("\n" + "=" * 50)
    print("ACTIVATION TESTS (HSigmoid, HSwish)")
    print("=" * 50)
    
    test_count += 1
    try:
        hsig = HSigmoid()
        x = torch.randn(100, 100) * 10
        y = hsig(x)
        assert y.min() >= 0 and y.max() <= 1
        print(f"  ✅ HSigmoid output in [0, 1]: min={y.min():.4f}, max={y.max():.4f}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ HSigmoid: {e}")
        all_passed = False
    
    test_count += 1
    try:
        hswish = HSwish()
        x = torch.randn(100, 100)
        y = hswish(x)
        assert not torch.isnan(y).any() and not torch.isinf(y).any()
        print(f"  ✅ HSwish no NaN/Inf: min={y.min():.4f}, max={y.max():.4f}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ HSwish: {e}")
        all_passed = False
    
    # ========== Shape Tests ==========
    print("\n" + "=" * 50)
    print("SHAPE TESTS")
    print("=" * 50)

    shape_tests = [
        ("Basic (64→64, 56x56)", 64, 64, 32, (2, 64, 56, 56)),
        ("Different channels (128→128)", 128, 128, 32, (2, 128, 28, 28)),
        ("inp!=oup (64→128)", 64, 128, 32, (2, 64, 28, 28)),  # Add this test
        ("Rectangular H>W", 64, 64, 32, (2, 64, 56, 28)),
        ("Rectangular W>H", 64, 64, 32, (2, 64, 28, 56)),
        ("Small spatial 7x7", 64, 64, 32, (2, 64, 7, 7)),
        ("Large spatial 112x112", 64, 64, 32, (1, 64, 112, 112)),
    ]

    for name, inp, oup, reduction, input_shape in shape_tests:
        test_count += 1
        try:
            coord_att = CoordAtt(inp=inp, oup=oup, reduction=reduction)
            x = torch.randn(*input_shape)
            y = coord_att(x)
            
            # Expected output shape has oup channels
            expected_shape = (input_shape[0], oup, input_shape[2], input_shape[3])
            assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
            print(f"  ✅ {name}: {input_shape} → {y.shape}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Reduction Tests ==========
    print("\n" + "=" * 50)
    print("REDUCTION TESTS")
    print("=" * 50)
    
    reduction_tests = [
        ("reduction=32, C=64", 64, 32, 8),   # max(8, 64//32) = max(8, 2) = 8
        ("reduction=16, C=64", 64, 16, 8),   # max(8, 64//16) = max(8, 4) = 8
        ("reduction=8, C=64", 64, 8, 8),     # max(8, 64//8) = max(8, 8) = 8
        ("reduction=4, C=64", 64, 4, 16),    # max(8, 64//4) = max(8, 16) = 16
        ("reduction=4, C=256", 256, 4, 64),  # max(8, 256//4) = max(8, 64) = 64
    ]
    
    for name, channels, reduction, expected_mip in reduction_tests:
        test_count += 1
        try:
            coord_att = CoordAtt(inp=channels, oup=channels, reduction=reduction)
            assert coord_att.mip == expected_mip, f"mip={coord_att.mip}, expected {expected_mip}"
            
            x = torch.randn(2, channels, 28, 28)
            y = coord_att(x)
            assert y.shape == (2, channels, 28, 28)
            
            print(f"  ✅ {name}: mip={coord_att.mip}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Attention Tests ==========
    print("\n" + "=" * 50)
    print("ATTENTION MECHANISM TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        coord_att = CoordAtt(inp=64, oup=64)
        coord_att.eval()
        x = torch.randn(2, 64, 28, 28)
        
        with torch.no_grad():
            a_h, a_w = coord_att.get_attention_maps(x)
        
        assert a_h.min() >= 0 and a_h.max() <= 1, "a_h not in [0, 1]"
        assert a_w.min() >= 0 and a_w.max() <= 1, "a_w not in [0, 1]"
        assert a_h.shape == (2, 64, 28, 1), f"a_h shape {a_h.shape}"
        assert a_w.shape == (2, 64, 1, 28), f"a_w shape {a_w.shape}"
        
        print(f"  ✅ Attention maps: a_h in [0,1], a_w in [0,1]")
        print(f"     a_h shape: {a_h.shape}, a_w shape: {a_w.shape}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Attention maps: {e}")
        all_passed = False
    
    test_count += 1
    try:
        coord_att = CoordAtt(inp=64, oup=64)
        coord_att.eval()
        x = torch.ones(2, 64, 28, 28)
        
        with torch.no_grad():
            y = coord_att(x)
        
        assert y.abs().sum() > 0, "Output is all zeros"
        print(f"  ✅ Identity preservation: output has non-zero values")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Identity preservation: {e}")
        all_passed = False
    
    # ========== Gradient Tests ==========
    print("\n" + "=" * 50)
    print("GRADIENT TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        coord_att = CoordAtt(inp=64, oup=64)
        x = torch.randn(2, 64, 28, 28)
        grad_info = check_gradient_flow(coord_att, x)
        
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
    
    test_count += 1
    try:
        coord_att = CoordAtt(inp=64, oup=64)
        x = torch.randn(2, 64, 28, 28, requires_grad=True)
        y = coord_att(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None, "Input gradient is None"
        print(f"  ✅ Backward pass: Completes without error")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Backward pass: {e}")
        all_passed = False
    
    # ========== Edge Case Tests ==========
    print("\n" + "=" * 50)
    print("EDGE CASE TESTS")
    print("=" * 50)
    
    edge_cases = [
        ("Odd height H=7", (2, 64, 7, 28)),
        ("Odd width W=13", (2, 64, 28, 13)),
        ("Minimum spatial 2x2", (2, 64, 2, 2)),
        ("H=1 (degenerate)", (2, 64, 1, 28)),
        ("W=1 (degenerate)", (2, 64, 28, 1)),
    ]
    
    for name, input_shape in edge_cases:
        test_count += 1
        try:
            coord_att = CoordAtt(inp=64, oup=64)
            x = torch.randn(*input_shape)
            y = coord_att(x)
            assert y.shape == input_shape
            print(f"  ✅ {name}: {input_shape} → {y.shape}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Stability Tests ==========
    print("\n" + "=" * 50)
    print("STABILITY TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        coord_att = CoordAtt(inp=64, oup=64)
        coord_att.eval()
        x = torch.randn(2, 64, 28, 28)
        
        with torch.no_grad():
            y = coord_att(x)
        
        check_no_nan_inf(y, "output")
        print(f"  ✅ No NaN/Inf: Output is numerically stable")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ No NaN/Inf: {e}")
        all_passed = False
    
    test_count += 1
    try:
        coord_att = CoordAtt(inp=64, oup=64)
        coord_att.eval()
        x = torch.randn(2, 64, 28, 28)
        
        with torch.no_grad():
            y1 = coord_att(x).clone()
            y2 = coord_att(x).clone()
        
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
        ("inp=64, oup=64, reduction=32", {"inp": 64, "oup": 64, "reduction": 32}),
        ("inp=128, oup=128, reduction=32", {"inp": 128, "oup": 128, "reduction": 32}),
        ("inp=64, oup=64, reduction=4", {"inp": 64, "oup": 64, "reduction": 4}),
    ]
    
    for name, params in configs:
        coord_att = CoordAtt(**params)
        p = count_parameters(coord_att)
        print(f"  {name}: {p['total']:,} parameters (mip={coord_att.mip})")
    
    # ========== Benchmark ==========
    print("\n" + "=" * 50)
    print("BENCHMARK (batch=2, C=64, 56×56)")
    print("=" * 50)
    
    import time
    
    coord_att = CoordAtt(inp=64, oup=64)
    coord_att.eval()
    x = torch.randn(2, 64, 56, 56)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = coord_att(x)
    
    # Forward timing
    forward_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = coord_att(x)
        forward_times.append((time.perf_counter() - start) * 1000)
    
    forward_mean = sum(forward_times) / len(forward_times)
    print(f"  Forward time:  {forward_mean:.3f} ms")
    
    # Backward timing
    coord_att.train()
    x = torch.randn(2, 64, 56, 56, requires_grad=True)
    
    backward_times = []
    for _ in range(20):
        coord_att.zero_grad()
        y = coord_att(x)
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