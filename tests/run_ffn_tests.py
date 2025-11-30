"""
Standalone test runner for Bottleneck FFN and Residual LayerNorm tests.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch


def run_all_tests():
    """Run all FFN tests manually with detailed output."""
    
    print("=" * 70)
    print("Bottleneck FFN & Residual LayerNorm Test Suite")
    print("=" * 70)
    
    from src.blocks import BottleneckFFN, ResidualLayerNormBlock
    from src.utils.testing import (
        check_output_shape,
        check_gradient_flow,
        check_no_nan_inf,
        count_parameters,
    )
    
    all_passed = True
    test_count = 0
    pass_count = 0
    
    # ========== BottleneckFFN Shape Tests ==========
    print("\n" + "=" * 50)
    print("BOTTLENECK FFN SHAPE TESTS")
    print("=" * 50)
    
    shape_tests = [
        ("Basic (256→256)", 256, 256, 0.25, (2, 196, 256)),
        ("Different dims (256→128)", 256, 128, 0.25, (2, 196, 256)),
        ("Increase dims (128→256)", 128, 256, 0.25, (2, 196, 128)),
        ("Small input (64→64)", 64, 64, 0.25, (2, 49, 64)),
        ("Large ratio (256→256, r=0.5)", 256, 256, 0.5, (2, 196, 256)),
    ]
    
    for name, inp, oup, ratio, input_shape in shape_tests:
        test_count += 1
        try:
            ffn = BottleneckFFN(inp=inp, oup=oup, bottleneck_ratio=ratio)
            x = torch.randn(*input_shape)
            y = ffn(x)
            
            expected_shape = (input_shape[0], input_shape[1], oup)
            assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
            
            print(f"  ✅ {name}: {input_shape} → {y.shape}")
            print(f"     Bottleneck channels: {ffn.bottleneck_channels}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Bottleneck Ratio Tests ==========
    print("\n" + "=" * 50)
    print("BOTTLENECK RATIO TESTS")
    print("=" * 50)
    
    ratio_tests = [
        (0.1, 25),
        (0.25, 64),
        (0.5, 128),
        (1.0, 256),
    ]
    
    for ratio, expected_channels in ratio_tests:
        test_count += 1
        try:
            ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=ratio)
            assert ffn.bottleneck_channels == expected_channels, \
                f"Expected {expected_channels}, got {ffn.bottleneck_channels}"
            
            # Verify forward pass works
            x = torch.randn(2, 49, 256)
            y = ffn(x)
            assert y.shape == (2, 49, 256)
            
            print(f"  ✅ ratio={ratio}: bottleneck_channels={ffn.bottleneck_channels}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ ratio={ratio}: {e}")
            all_passed = False
    
    # ========== ResidualLayerNormBlock Tests ==========
    print("\n" + "=" * 50)
    print("RESIDUAL LAYERNORM TESTS")
    print("=" * 50)
    
    # Basic shape test
    test_count += 1
    try:
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 196, 256)
        y = res_ln(x)
        
        assert y.shape == (2, 196, 256), f"Shape mismatch: {y.shape}"
        print(f"  ✅ Basic shape: (2, 196, 256) → {y.shape}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Basic shape: {e}")
        all_passed = False
    
    # Default residual test
    test_count += 1
    try:
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        
        y = res_ln(x)
        y_manual = res_ln.norm(x) + x
        
        assert torch.allclose(y, y_manual, atol=1e-6), "Default residual not working"
        print(f"  ✅ Default residual (y = LN(x) + x): Verified")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Default residual: {e}")
        all_passed = False
    
    # Explicit residual test
    test_count += 1
    try:
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        residual = torch.randn(2, 49, 256)
        
        y = res_ln(x, residual=residual)
        y_manual = res_ln.norm(x) + residual
        
        assert torch.allclose(y, y_manual, atol=1e-6), "Explicit residual not working"
        print(f"  ✅ Explicit residual (y = LN(x) + r): Verified")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Explicit residual: {e}")
        all_passed = False
    
    # Stateless test
    test_count += 1
    try:
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        
        res_ln.train()
        y_train = res_ln(x).clone()
        
        res_ln.eval()
        with torch.no_grad():
            y_eval = res_ln(x).clone()
        
        assert torch.allclose(y_train, y_eval, atol=1e-6), "LayerNorm not stateless"
        print(f"  ✅ Stateless (train == eval): Verified")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Stateless: {e}")
        all_passed = False
    
    # ========== Dropout Tests ==========
    print("\n" + "=" * 50)
    print("DROPOUT TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        ffn = BottleneckFFN(inp=256, oup=256, dropout=0.5)
        ffn.train()
        x = torch.randn(2, 49, 256)
        
        outputs = [ffn(x).clone() for _ in range(5)]
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        
        assert not all_same, "Dropout should cause different outputs in train mode"
        print(f"  ✅ Dropout train mode: Different outputs verified")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Dropout train mode: {e}")
        all_passed = False
    
    test_count += 1
    try:
        ffn = BottleneckFFN(inp=256, oup=256, dropout=0.5)
        ffn.eval()
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y1 = ffn(x).clone()
            y2 = ffn(x).clone()
        
        assert torch.allclose(y1, y2), "Outputs should be identical in eval mode"
        print(f"  ✅ Dropout eval mode: Identical outputs verified")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Dropout eval mode: {e}")
        all_passed = False
    
    # ========== Gradient Tests ==========
    print("\n" + "=" * 50)
    print("GRADIENT TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        ffn = BottleneckFFN(inp=256, oup=256)
        x = torch.randn(2, 49, 256)
        
        grad_info = check_gradient_flow(ffn, x)
        all_have_grad = all(grad_info.values())
        
        if all_have_grad:
            print(f"  ✅ FFN gradient flow: All {len(grad_info)} parameters receive gradients")
            pass_count += 1
        else:
            missing = [k for k, v in grad_info.items() if not v]
            print(f"  ❌ FFN gradient flow: Missing gradients for {missing}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FFN gradient flow: {e}")
        all_passed = False
    
    test_count += 1
    try:
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        
        grad_info = check_gradient_flow(res_ln, x)
        all_have_grad = all(grad_info.values())
        
        if all_have_grad:
            print(f"  ✅ ResLN gradient flow: All {len(grad_info)} parameters receive gradients")
            pass_count += 1
        else:
            missing = [k for k, v in grad_info.items() if not v]
            print(f"  ❌ ResLN gradient flow: Missing gradients for {missing}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ ResLN gradient flow: {e}")
        all_passed = False
    
    # ========== Stability Tests ==========
    print("\n" + "=" * 50)
    print("STABILITY TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        ffn = BottleneckFFN(inp=256, oup=256)
        ffn.eval()
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y = ffn(x)
        
        check_no_nan_inf(y, "FFN output")
        print(f"  ✅ FFN stability: No NaN/Inf in output")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ FFN stability: {e}")
        all_passed = False
    
    test_count += 1
    try:
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = torch.randn(2, 49, 256)
        
        with torch.no_grad():
            y = res_ln(x)
        
        check_no_nan_inf(y, "ResLN output")
        print(f"  ✅ ResLN stability: No NaN/Inf in output")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ ResLN stability: {e}")
        all_passed = False
    
    # ========== Parameter Count ==========
    print("\n" + "=" * 50)
    print("PARAMETER COUNT")
    print("=" * 50)
    
    configs = [
        ("FFN (256→256, r=0.25)", BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.25)),
        ("FFN (256→256, r=0.5)", BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.5)),
        ("FFN (256→256, r=1.0)", BottleneckFFN(inp=256, oup=256, bottleneck_ratio=1.0)),
        ("ResLN (256)", ResidualLayerNormBlock(embed_dim=256)),
    ]
    
    for name, module in configs:
        params = count_parameters(module)
        print(f"  {name}: {params['total']:,} parameters")
    
    # Verify ResLN has 2*embed_dim params
    res_ln = ResidualLayerNormBlock(embed_dim=256)
    resln_params = count_parameters(res_ln)['total']
    expected = 2 * 256
    print(f"\n  ResLN expected: {expected} (2 × embed_dim)")
    print(f"  ResLN actual:   {resln_params}")
    if resln_params == expected:
        print(f"  ✅ Parameter count correct")
    else:
        print(f"  ❌ Parameter count mismatch")
    
    # ========== Benchmark ==========
    print("\n" + "=" * 50)
    print("BENCHMARK (batch=2, N=196, D=256)")
    print("=" * 50)
    
    import time
    
    ffn = BottleneckFFN(inp=256, oup=256, bottleneck_ratio=0.25)
    res_ln = ResidualLayerNormBlock(embed_dim=256)
    ffn.eval()
    x = torch.randn(2, 196, 256)
    
    # FFN timing
    for _ in range(5):
        with torch.no_grad():
            _ = ffn(x)
    
    ffn_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = ffn(x)
        ffn_times.append((time.perf_counter() - start) * 1000)
    
    # ResLN timing
    resln_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = res_ln(x)
        resln_times.append((time.perf_counter() - start) * 1000)
    
    print(f"  FFN forward:   {sum(ffn_times)/len(ffn_times):.3f} ms")
    print(f"  ResLN forward: {sum(resln_times)/len(resln_times):.3f} ms")
    
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