"""
Standalone test runner for Linear Differential Attention tests.
"""

import sys
import os
import math

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch


def run_all_tests():
    """Run all LDA tests manually with detailed output."""
    
    print("=" * 70)
    print("Linear Differential Attention Comprehensive Test Suite")
    print("=" * 70)
    
    from src.blocks import LinearDifferentialAttention, NaiveFullAttention
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
        ("Basic (B=2, N=196, D=256)", 2, 196, 256, 8),
        ("Small sequence (N=16)", 2, 16, 256, 8),
        ("Large sequence (N=400)", 2, 400, 256, 8),
        ("Small embed_dim (D=64)", 2, 49, 64, 4),
        ("Large embed_dim (D=512)", 2, 49, 512, 8),
        ("Single head", 2, 49, 256, 1),
        ("Many heads (16)", 2, 49, 256, 16),
    ]
    
    for name, B, N, D, heads in shape_tests:
        test_count += 1
        try:
            lda = LinearDifferentialAttention(embed_dim=D, num_heads=heads)
            x = torch.randn(B, N, D)
            y = lda(x)
            
            expected_shape = (B, N, D)
            assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
            print(f"  ✅ {name}: {x.shape} → {y.shape}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Differential Mechanism Tests ==========
    print("\n" + "=" * 50)
    print("DIFFERENTIAL MECHANISM TESTS")
    print("=" * 50)
    
    # Alpha learnable
    test_count += 1
    try:
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        assert lda.alpha.requires_grad, "Alpha should require gradients"
        print(f"  ✅ Alpha is learnable parameter")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Alpha learnable: {e}")
        all_passed = False
    
    # Alpha positive
    test_count += 1
    try:
        for init in [-1.0, 0.0, 0.8, 1.5]:
            lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, init=init)
            assert lda.alpha.item() > 0, f"Alpha should be positive for init={init}"
        print(f"  ✅ Alpha always positive (exp initialization)")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Alpha positive: {e}")
        all_passed = False
    
    # Alpha initialization
    test_count += 1
    try:
        init = 0.8
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, init=init)
        expected_alpha = math.exp(init)
        assert abs(lda.alpha.item() - expected_alpha) < 1e-5
        print(f"  ✅ Alpha = exp({init}) = {expected_alpha:.4f}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Alpha initialization: {e}")
        all_passed = False
    
    # Differential computation
    test_count += 1
    try:
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            A1, A2, A_diff = lda.get_attention_maps(x)
            expected_diff = lda.alpha * (A1 - A2)
            assert torch.allclose(A_diff, expected_diff, atol=1e-6)
        print(f"  ✅ A_diff = α × (A1 - A2) verified")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Differential computation: {e}")
        all_passed = False
    
    # ========== Attention Properties Tests ==========
    print("\n" + "=" * 50)
    print("ATTENTION PROPERTIES TESTS")
    print("=" * 50)
    
    # Softmax rows sum to 1
    test_count += 1
    try:
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            A1, A2, _ = lda.get_attention_maps(x)
            A1_sums = A1.sum(dim=-1)
            A2_sums = A2.sum(dim=-1)
            
            assert torch.allclose(A1_sums, torch.ones_like(A1_sums), atol=1e-5)
            assert torch.allclose(A2_sums, torch.ones_like(A2_sums), atol=1e-5)
        print(f"  ✅ Attention rows sum to 1.0")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Softmax sum: {e}")
        all_passed = False
    
    # Attention values in [0, 1]
    test_count += 1
    try:
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            A1, A2, _ = lda.get_attention_maps(x)
            assert (A1 >= 0).all() and (A1 <= 1).all()
            assert (A2 >= 0).all() and (A2 <= 1).all()
        print(f"  ✅ A1, A2 values in [0, 1]")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Attention range: {e}")
        all_passed = False
    
    # ========== Gradient Tests ==========
    print("\n" + "=" * 50)
    print("GRADIENT TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 49, 256)
        grad_info = check_gradient_flow(lda, x)
        
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
    
    # Alpha gradient
    test_count += 1
    try:
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 49, 256)
        y = lda(x)
        loss = y.sum()
        loss.backward()
        
        assert lda.alpha.grad is not None, "Alpha should receive gradient"
        assert lda.alpha.grad.abs() > 0, "Alpha gradient should be non-zero"
        print(f"  ✅ Alpha receives gradient: {lda.alpha.grad.item():.6f}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Alpha gradient: {e}")
        all_passed = False
    
    # ========== Dropout Tests ==========
    print("\n" + "=" * 50)
    print("DROPOUT TESTS")
    print("=" * 50)
    
    # Eval mode deterministic
    test_count += 1
    try:
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8, dropout=0.5)
        lda.eval()
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y1 = lda(x).clone()
            y2 = lda(x).clone()
        
        assert torch.allclose(y1, y2, atol=1e-6)
        print(f"  ✅ Eval mode: Deterministic output")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Eval mode deterministic: {e}")
        all_passed = False
    
    # ========== Stability Tests ==========
    print("\n" + "=" * 50)
    print("STABILITY TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y = lda(x)
        
        check_no_nan_inf(y, "output")
        print(f"  ✅ No NaN/Inf: Output is numerically stable")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ No NaN/Inf: {e}")
        all_passed = False
    
    # Large values
    test_count += 1
    try:
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        lda.eval()
        x = torch.randn(2, 49, 256) * 10
        
        with torch.no_grad():
            y = lda(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
        print(f"  ✅ Large values: Stable with input × 10")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Large values: {e}")
        all_passed = False
    
    # ========== Parameter Count ==========
    print("\n" + "=" * 50)
    print("PARAMETER COUNT")
    print("=" * 50)
    
    configs = [
        ("D=256, heads=8", {"embed_dim": 256, "num_heads": 8}),
        ("D=128, heads=4", {"embed_dim": 128, "num_heads": 4}),
        ("D=512, heads=16", {"embed_dim": 512, "num_heads": 16}),
    ]
    
    for name, params in configs:
        lda = LinearDifferentialAttention(**params)
        p = count_parameters(lda)
        print(f"  {name}: {p['total']:,} parameters")
    
    # ========== Benchmark ==========
    print("\n" + "=" * 50)
    print("BENCHMARK (batch=2, N=196, D=256, heads=8)")
    print("=" * 50)
    
    import time
    
    lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
    lda.eval()
    x = torch.randn(2, 196, 256)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = lda(x)
    
    # Forward timing
    forward_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = lda(x)
        forward_times.append((time.perf_counter() - start) * 1000)
    
    forward_mean = sum(forward_times) / len(forward_times)
    print(f"  Forward time:  {forward_mean:.3f} ms")
    
    # Backward timing
    lda.train()
    x = torch.randn(2, 196, 256, requires_grad=True)
    
    backward_times = []
    for _ in range(20):
        lda.zero_grad()
        y = lda(x)
        start = time.perf_counter()
        y.sum().backward()
        backward_times.append((time.perf_counter() - start) * 1000)
    
    backward_mean = sum(backward_times) / len(backward_times)
    print(f"  Backward time: {backward_mean:.3f} ms")
    print(f"  Total time:    {forward_mean + backward_mean:.3f} ms")
    
    # Scaling test
    print("\n  Scaling with sequence length:")
    lda.eval()
    for N in [49, 100, 196, 400]:
        x = torch.randn(2, N, 256)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _ = lda(x)
            times.append((time.perf_counter() - start) * 1000)
        mean_time = sum(times) / len(times)
        print(f"    N={N}: {mean_time:.3f} ms")
    
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