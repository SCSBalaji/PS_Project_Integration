#!/usr/bin/env python
"""
Standalone smoke test runner.

Run this script to quickly verify that the entire pipeline is working:
    python run_smoke_test.py

This performs the same checks as the CI smoke tests but can be run locally
without pytest.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_smoke_test():
    """Run comprehensive smoke test."""
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  MOBILEPLANT-VIT SMOKE TEST".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    start_time = time.time()
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Imports
    print("─" * 70)
    print("TEST 1: Import Check")
    print("─" * 70)
    try:
        from utils import set_seed, load_config, get_device, get_runtime_info
        from utils import ExperimentLogger, ExperimentManager
        from blocks import (
            GhostConv, FusedInvertedResidualBlock, CoordAtt,
            PatchEmbedding, PositionalEncoding, LinearDifferentialAttention,
            ResidualLayerNormBlock, BottleneckFFN, ClassifierHead, GlobalAveragePooling
        )
        import torch
        import numpy as np
        
        print("  ✅ All imports successful")
        tests_passed += 1
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        tests_failed += 1
        return False
    
    # Test 2: Configuration
    print("\n" + "─" * 70)
    print("TEST 2: Configuration Loading")
    print("─" * 70)
    try:
        config = load_config("config/defaults.yaml")
        assert 'reproducibility' in config
        assert 'seed' in config['reproducibility']
        print(f"  ✅ Config loaded successfully")
        print(f"     Seed: {config['reproducibility']['seed']}")
        print(f"     Sections: {list(config.keys())}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        tests_failed += 1
    
    # Test 3: Reproducibility
    print("\n" + "─" * 70)
    print("TEST 3: Reproducibility Verification")
    print("─" * 70)
    try:
        import random
        
        # First run
        set_seed(42)
        rand1 = [random.random() for _ in range(5)]
        np1 = np.random.rand(5).tolist()
        torch1 = torch.rand(5).tolist()
        
        # Second run
        set_seed(42)
        rand2 = [random.random() for _ in range(5)]
        np2 = np.random.rand(5).tolist()
        torch2 = torch.rand(5).tolist()
        
        assert rand1 == rand2, "Python random not reproducible"
        assert np1 == np2, "NumPy random not reproducible"
        assert torch1 == torch2, "PyTorch random not reproducible"
        
        print("  ✅ Reproducibility verified")
        print("     Python random: ✓")
        print("     NumPy random: ✓")
        print("     PyTorch random: ✓")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Reproducibility test failed: {e}")
        tests_failed += 1
    
    # Test 4: Device Detection
    print("\n" + "─" * 70)
    print("TEST 4: Device Detection")
    print("─" * 70)
    try:
        device = get_device()
        runtime_info = get_runtime_info()
        print(f"  ✅ Device detected: {device}")
        print(f"     PyTorch version: {runtime_info['torch_version']}")
        print(f"     CUDA available: {runtime_info['cuda_available']}")
        if runtime_info['gpu_name']:
            print(f"     GPU: {runtime_info['gpu_name']}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Device detection failed: {e}")
        tests_failed += 1
    
    # Test 5: Forward Pass Through Pipeline
    print("\n" + "─" * 70)
    print("TEST 5: Forward Pass Through Pipeline")
    print("─" * 70)
    try:
        set_seed(42)
        
        x = torch.randn(2, 3, 224, 224)
        print(f"  Input: {x.shape}")
        
        # GhostConv
        ghost = GhostConv(inp=3, oup=64)
        x = ghost(x)
        print(f"  → GhostConv: {x.shape}")
        
        # FusedIR
        fused = FusedInvertedResidualBlock(inp=64, oup=64)
        x = fused(x)
        print(f"  → FusedIR: {x.shape}")
        
        # CoordAtt
        coord = CoordAtt(inp=64, oup=64)
        x = coord(x)
        print(f"  → CoordAtt: {x.shape}")
        
        # PatchEmbedding
        patch = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=14)
        x = patch(x)
        print(f"  → PatchEmbed: {x.shape}")
        
        # PositionalEncoding
        pos = PositionalEncoding(embed_dim=256)
        x = pos(x)
        print(f"  → PosEnc: {x.shape}")
        
        # LDA
        lda = LinearDifferentialAttention(embed_dim=256, num_heads=8)
        x = lda(x)
        print(f"  → LDA: {x.shape}")
        
        # ResidualLayerNorm
        res_ln = ResidualLayerNormBlock(embed_dim=256)
        x = res_ln(x)
        print(f"  → ResLN: {x.shape}")
        
        # BottleneckFFN
        ffn = BottleneckFFN(inp=256, oup=256)
        x = ffn(x)
        print(f"  → FFN: {x.shape}")
        
        # GAP
        gap = GlobalAveragePooling()
        x = gap(x)
        print(f"  → GAP: {x.shape}")
        
        # Classifier
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = classifier(x)
        print(f"  → Classifier: {x.shape}")
        
        print("  ✅ Forward pass successful")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        tests_failed += 1
    
    # Test 6: Output Validation
    print("\n" + "─" * 70)
    print("TEST 6: Output Validation")
    print("─" * 70)
    try:
        assert x.shape == (2, 38), f"Wrong output shape: {x.shape}"
        assert torch.allclose(x.sum(dim=1), torch.ones(2), atol=1e-5), \
            "Output probabilities don't sum to 1"
        assert (x >= 0).all(), "Output contains negative values"
        assert (x <= 1).all(), "Output contains values > 1"
        
        print("  ✅ Output validation passed")
        print(f"     Shape: {x.shape} ✓")
        print(f"     Sum to 1: {x.sum(dim=1).tolist()} ✓")
        print(f"     Range [0, 1]: ✓")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Output validation failed: {e}")
        tests_failed += 1
    
    # Test 7: Gradient Flow
    print("\n" + "─" * 70)
    print("TEST 7: Gradient Flow Check")
    print("─" * 70)
    try:
        set_seed(42)
        
        x = torch.randn(1, 3, 64, 64, requires_grad=True)  # Smaller for speed
        
        model = torch.nn.Sequential(
            GhostConv(inp=3, oup=64),
            FusedInvertedResidualBlock(inp=64, oup=64),
            CoordAtt(inp=64, oup=64),
        )
        
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, "No gradients computed"
        assert not torch.isnan(x.grad).any(), "Gradients contain NaN"
        assert not torch.isinf(x.grad).any(), "Gradients contain Inf"
        
        print("  ✅ Gradient flow verified")
        print(f"     Gradient shape: {x.grad.shape}")
        print(f"     Gradient mean: {x.grad.mean().item():.6f}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Gradient flow check failed: {e}")
        tests_failed += 1
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "═" * 70)
    print("  SMOKE TEST SUMMARY")
    print("═" * 70)
    print(f"  Tests Passed: {tests_passed}")
    print(f"  Tests Failed: {tests_failed}")
    print(f"  Time Elapsed: {elapsed:.2f}s")
    print("═" * 70)
    
    if tests_failed == 0:
        print("\n  🎉 ALL SMOKE TESTS PASSED! Pipeline is ready.\n")
        return True
    else:
        print(f"\n  ⚠️  {tests_failed} test(s) failed. Please fix before proceeding.\n")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)