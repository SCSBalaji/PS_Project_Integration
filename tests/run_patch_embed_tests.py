"""
Standalone test runner for Patch Embedding and Positional Encoding tests.
"""

import sys
import os
import math

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch


def run_all_tests():
    """Run all tests manually with detailed output."""
    
    print("=" * 70)
    print("Patch Embedding & Positional Encoding Test Suite")
    print("=" * 70)
    
    from src.blocks import PatchEmbedding, PositionalEncoding
    from src.utils.testing import (
        check_output_shape,
        check_gradient_flow,
        check_no_nan_inf,
        count_parameters,
    )
    
    all_passed = True
    test_count = 0
    pass_count = 0
    
    # ========== Patch Embedding Shape Tests ==========
    print("\n" + "=" * 50)
    print("PATCH EMBEDDING SHAPE TESTS")
    print("=" * 50)
    
    shape_tests = [
        ("Basic (64, 56, 56) → (196, 256)", 64, 256, 4, (2, 64, 56, 56), (2, 196, 256)),
        ("Small spatial (64, 28, 28) → (49, 256)", 64, 256, 4, (2, 64, 28, 28), (2, 49, 256)),
        ("Large spatial (64, 112, 112) → (784, 256)", 64, 256, 4, (2, 64, 112, 112), (2, 784, 256)),
        ("Patch size 2 (64, 14, 14) → (49, 256)", 64, 256, 2, (2, 64, 14, 14), (2, 49, 256)),
        ("Rectangular (64, 28, 56) → (98, 256)", 64, 256, 4, (2, 64, 28, 56), (2, 98, 256)),
    ]
    
    for name, in_ch, embed_dim, patch_size, input_shape, expected_shape in shape_tests:
        test_count += 1
        try:
            patch_embed = PatchEmbedding(in_channels=in_ch, embed_dim=embed_dim, patch_size=patch_size)
            x = torch.randn(*input_shape)
            y = patch_embed(x)
            
            assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
            print(f"  ✅ {name}")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Positional Encoding Shape Tests ==========
    print("\n" + "=" * 50)
    print("POSITIONAL ENCODING SHAPE TESTS")
    print("=" * 50)
    
    pos_shape_tests = [
        ("Basic (196, 256)", 256, (2, 196, 256)),
        ("Short sequence (49, 256)", 256, (2, 49, 256)),
        ("Long sequence (784, 256)", 256, (2, 784, 256)),
        ("Different embed_dim (196, 128)", 128, (2, 196, 128)),
    ]
    
    for name, embed_dim, input_shape in pos_shape_tests:
        test_count += 1
        try:
            pos_enc = PositionalEncoding(embed_dim=embed_dim, max_len=5000)
            x = torch.randn(*input_shape)
            y = pos_enc(x)
            
            assert y.shape == input_shape, f"Expected {input_shape}, got {y.shape}"
            print(f"  ✅ {name}: Input shape preserved")
            pass_count += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # ========== Positional Encoding Value Tests ==========
    print("\n" + "=" * 50)
    print("POSITIONAL ENCODING VALUE TESTS")
    print("=" * 50)
    
    # Test position info added
    test_count += 1
    try:
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        x = torch.randn(2, 49, 256)
        y = pos_enc(x)
        
        diff = (y - x).abs().mean()
        assert diff > 0, "Positional encoding did not change values"
        print(f"  ✅ Position info added: mean diff = {diff:.6f}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Position info test: {e}")
        all_passed = False
    
    # Test no learnable params
    test_count += 1
    try:
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        num_params = sum(p.numel() for p in pos_enc.parameters())
        
        assert num_params == 0, f"Expected 0 params, got {num_params}"
        print(f"  ✅ No learnable parameters: {num_params}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Learnable params test: {e}")
        all_passed = False
    
    # Test sinusoidal pattern
    test_count += 1
    try:
        pos_enc = PositionalEncoding(embed_dim=256, max_len=100)
        pe = pos_enc.pe[0]
        
        # Check sin(0) = 0, cos(0) = 1
        assert abs(pe[0, 0].item() - 0.0) < 1e-5, "PE[0,0] should be 0"
        assert abs(pe[0, 1].item() - 1.0) < 1e-5, "PE[0,1] should be 1"
        print(f"  ✅ Sinusoidal pattern: PE[0,0]={pe[0,0].item():.4f}, PE[0,1]={pe[0,1].item():.4f}")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Sinusoidal pattern test: {e}")
        all_passed = False
    
    # ========== Combined Pipeline Tests ==========
    print("\n" + "=" * 50)
    print("COMBINED PIPELINE TESTS")
    print("=" * 50)
    
    # Pipeline shape test
    test_count += 1
    try:
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        x = torch.randn(2, 64, 56, 56)
        patches = patch_embed(x)
        output = pos_enc(patches)
        
        assert output.shape == (2, 196, 256), f"Expected (2, 196, 256), got {output.shape}"
        print(f"  ✅ Pipeline shape: (2, 64, 56, 56) → (2, 196, 256)")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Pipeline shape test: {e}")
        all_passed = False
    
    # Pipeline gradient test
    test_count += 1
    try:
        patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
        pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
        
        x = torch.randn(2, 64, 56, 56, requires_grad=True)
        patches = patch_embed(x)
        output = pos_enc(patches)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "Input gradient is None"
        assert x.grad.abs().sum() > 0, "Input gradient is zero"
        print(f"  ✅ Pipeline gradient flow: OK")
        pass_count += 1
    except Exception as e:
        print(f"  ❌ Pipeline gradient test: {e}")
        all_passed = False
    
    # ========== Parameter Count ==========
    print("\n" + "=" * 50)
    print("PARAMETER COUNT")
    print("=" * 50)
    
    patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
    pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
    
    pe_params = count_parameters(patch_embed)
    pos_params = count_parameters(pos_enc)
    
    print(f"  PatchEmbedding (64→256, p=4): {pe_params['total']:,} parameters")
    print(f"  PositionalEncoding (D=256): {pos_params['total']:,} parameters")
    print(f"  Total Transition Stage: {pe_params['total'] + pos_params['total']:,} parameters")
    
    # ========== Benchmark ==========
    print("\n" + "=" * 50)
    print("BENCHMARK")
    print("=" * 50)
    
    import time
    
    patch_embed = PatchEmbedding(in_channels=64, embed_dim=256, patch_size=4)
    pos_enc = PositionalEncoding(embed_dim=256, max_len=5000)
    
    patch_embed.eval()
    pos_enc.eval()
    
    x = torch.randn(2, 64, 56, 56)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = pos_enc(patch_embed(x))
    
    # Timing
    times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = pos_enc(patch_embed(x))
        times.append((time.perf_counter() - start) * 1000)
    
    mean_time = sum(times) / len(times)
    print(f"  Pipeline forward time: {mean_time:.3f} ms")
    
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