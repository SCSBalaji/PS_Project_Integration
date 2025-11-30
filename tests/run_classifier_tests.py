"""
Standalone test runner for Global Average Pooling and Classifier Head tests.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn


def run_all_tests():
    """Run all classifier tests manually with detailed output."""
    
    print("=" * 70)
    print("Global Average Pooling & Classifier Head Test Suite")
    print("=" * 70)
    
    from src.blocks import GlobalAveragePooling, ClassifierHead, CombinedClassifier
    from src.utils.testing import (
        check_output_shape,
        check_gradient_flow,
        check_no_nan_inf,
        count_parameters,
    )
    
    all_passed = True
    test_count = 0
    pass_count = 0
    
    # ========== GAP Shape Tests ==========
    print("\n" + "=" * 50)
    print("GAP SHAPE TESTS")
    print("=" * 50)
    
    shape_tests = [
        ("Basic (196→pooled)", 2, 196, 256),
        ("Short sequence (16)", 2, 16, 256),
        ("Long sequence (400)", 2, 400, 256),
        ("Small embed (D=64)", 2, 49, 64),
        ("Large embed (D=512)", 2, 49, 512),
        ("Single token", 2, 1, 256),
    ]
    
    for name, B, N, D in shape_tests:
        test_count += 1
        try:
            gap = GlobalAveragePooling()
            x = torch.randn(B, N, D)
            y = gap(x)
            
            assert y.shape == (B, D), f"Expected ({B}, {D}), got {y.shape}"
            print(f"  ✅ {name}: ({B}, {N}, {D}) → {y.shape}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== GAP Averaging Tests ==========
    print("\n" + "=" * 50)
    print("GAP AVERAGING TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        gap = GlobalAveragePooling()
        x = torch.randn(2, 49, 256)
        y = gap(x)
        manual = x.mean(dim=1)
        
        assert torch.allclose(y, manual, atol=1e-6), "GAP does not match manual mean"
        print(f"  ✅ Averaging correct: max diff = {(y - manual).abs().max():.2e}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Averaging test: {e}")
        all_passed = False
    
    test_count += 1
    try:
        gap = GlobalAveragePooling()
        params = count_parameters(gap)
        assert params['total'] == 0, f"GAP should have 0 params, got {params['total']}"
        print(f"  ✅ Parameter-free: {params['total']} parameters")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Parameter count: {e}")
        all_passed = False
    
    # ========== ClassifierHead Shape Tests ==========
    print("\n" + "=" * 50)
    print("CLASSIFIER HEAD SHAPE TESTS")
    print("=" * 50)
    
    classifier_tests = [
        ("Basic (256→38)", 256, 38, (2, 256)),
        ("Small classes (10)", 256, 10, (2, 256)),
        ("Large classes (1000)", 256, 1000, (2, 256)),
        ("Small embed (64→38)", 64, 38, (2, 64)),
        ("Large embed (512→38)", 512, 38, (2, 512)),
    ]
    
    for name, embed_dim, num_classes, input_shape in classifier_tests:
        test_count += 1
        try:
            classifier = ClassifierHead(embed_dim=embed_dim, num_classes=num_classes)
            classifier.eval()
            x = torch.randn(*input_shape)
            
            with torch.no_grad():
                y = classifier(x)
            
            expected_shape = (input_shape[0], num_classes)
            assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
            print(f"  ✅ {name}: {input_shape} → {y.shape}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Probability Tests ==========
    print("\n" + "=" * 50)
    print("PROBABILITY TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        classifier.eval()
        x = torch.randn(4, 256)
        
        with torch.no_grad():
            probs = classifier(x)
        
        # Check sum to 1
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5), "Probs don't sum to 1"
        print(f"  ✅ Probabilities sum to 1: {prob_sums.tolist()}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Sum to 1 test: {e}")
        all_passed = False
    
    test_count += 1
    try:
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        classifier.eval()
        x = torch.randn(4, 256)
        
        with torch.no_grad():
            probs = classifier(x)
        
        assert (probs >= 0).all(), "Found negative probabilities"
        assert (probs <= 1).all(), "Found probabilities > 1"
        print(f"  ✅ Probabilities in [0, 1]: range [{probs.min():.4f}, {probs.max():.4f}]")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Range test: {e}")
        all_passed = False
    
    # ========== Logits Test ==========
    test_count += 1
    try:
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        classifier.eval()
        x = torch.randn(2, 256)
        
        with torch.no_grad():
            logits = classifier.get_logits(x)
            probs = classifier(x)
        
        expected = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs, expected, atol=1e-5), "Logits-probs mismatch"
        print(f"  ✅ get_logits() method works correctly")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Logits test: {e}")
        all_passed = False
    
    # ========== Combined Classifier Tests ==========
    print("\n" + "=" * 50)
    print("COMBINED CLASSIFIER TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        combined = CombinedClassifier(embed_dim=256, num_classes=38)
        combined.eval()
        x = torch.randn(2, 196, 256)
        
        with torch.no_grad():
            y = combined(x)
        
        assert y.shape == (2, 38), f"Expected (2, 38), got {y.shape}"
        assert torch.allclose(y.sum(dim=-1), torch.ones(2), atol=1e-5), "Probs don't sum to 1"
        print(f"  ✅ Combined (2, 196, 256) → {y.shape}, probs valid")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Combined test: {e}")
        all_passed = False
    
    # ========== Gradient Tests ==========
    print("\n" + "=" * 50)
    print("GRADIENT TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        
        grad_info = check_gradient_flow(classifier, x)
        all_have_grad = all(grad_info.values())
        
        if all_have_grad:
            print(f"  ✅ ClassifierHead gradient flow: {len(grad_info)} params")
            pass_count += 1
        else:
            missing = [k for k, v in grad_info.items() if not v]
            print(f"  ❌ Missing gradients: {missing}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Gradient test: {e}")
        all_passed = False
    
    test_count += 1
    try:
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        target = torch.randint(0, 38, (2,))
        
        logits = classifier.get_logits(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        assert classifier.fc.weight.grad is not None
        print(f"  ✅ CrossEntropyLoss backward works")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ CE loss test: {e}")
        all_passed = False
    
    test_count += 1
    try:
        gap = GlobalAveragePooling()
        x = torch.randn(2, 49, 256, requires_grad=True)
        
        y = gap(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        print(f"  ✅ GAP gradient passthrough works")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ GAP gradient: {e}")
        all_passed = False
    
    # ========== Dropout Tests ==========
    print("\n" + "=" * 50)
    print("DROPOUT TESTS")
    print("=" * 50)
    
    test_count += 1
    try:
        classifier = ClassifierHead(embed_dim=256, num_classes=38, dropout=0.5)
        classifier.train()
        x = torch.randn(2, 256)
        
        outputs = [classifier(x).clone() for _ in range(5)]
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        
        assert not all_same, "Dropout should cause different outputs"
        print(f"  ✅ Dropout train mode: outputs differ")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Dropout train: {e}")
        all_passed = False
    
    test_count += 1
    try:
        classifier = ClassifierHead(embed_dim=256, num_classes=38, dropout=0.5)
        classifier.eval()
        x = torch.randn(2, 256)
        
        with torch.no_grad():
            y1 = classifier(x).clone()
            y2 = classifier(x).clone()
        
        assert torch.allclose(y1, y2), "Eval mode should be deterministic"
        print(f"  ✅ Dropout eval mode: outputs identical")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Dropout eval: {e}")
        all_passed = False
    
    # ========== Parameter Count ==========
    print("\n" + "=" * 50)
    print("PARAMETER COUNT")
    print("=" * 50)
    
    gap = GlobalAveragePooling()
    classifier = ClassifierHead(embed_dim=256, num_classes=38)
    combined = CombinedClassifier(embed_dim=256, num_classes=38)
    
    gap_params = count_parameters(gap)['total']
    cls_params = count_parameters(classifier)['total']
    comb_params = count_parameters(combined)['total']
    
    expected_cls = 256 * 38 + 38
    
    print(f"  GAP:        {gap_params:,} (expected: 0)")
    print(f"  Classifier: {cls_params:,} (expected: {expected_cls:,})")
    print(f"  Combined:   {comb_params:,} (expected: {expected_cls:,})")
    
    if gap_params == 0 and cls_params == expected_cls:
        print(f"  ✅ Parameter counts correct")
    else:
        print(f"  ❌ Parameter count mismatch")
    
    # ========== Benchmark ==========
    print("\n" + "=" * 50)
    print("BENCHMARK")
    print("=" * 50)
    
    import time
    
    # GAP timing
    gap = GlobalAveragePooling()
    x = torch.randn(2, 196, 256)
    
    for _ in range(5):
        with torch.no_grad():
            _ = gap(x)
    
    gap_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = gap(x)
        gap_times.append((time.perf_counter() - start) * 1000)
    
    # Classifier timing
    classifier = ClassifierHead(embed_dim=256, num_classes=38)
    classifier.eval()
    x_cls = torch.randn(2, 256)
    
    for _ in range(5):
        with torch.no_grad():
            _ = classifier(x_cls)
    
    cls_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = classifier(x_cls)
        cls_times.append((time.perf_counter() - start) * 1000)
    
    # Combined timing
    combined = CombinedClassifier(embed_dim=256, num_classes=38)
    combined.eval()
    x_comb = torch.randn(2, 196, 256)
    
    comb_times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = combined(x_comb)
        comb_times.append((time.perf_counter() - start) * 1000)
    
    print(f"  GAP forward:        {sum(gap_times)/len(gap_times):.3f} ms")
    print(f"  Classifier forward: {sum(cls_times)/len(cls_times):.3f} ms")
    print(f"  Combined forward:   {sum(comb_times)/len(comb_times):.3f} ms")
    
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