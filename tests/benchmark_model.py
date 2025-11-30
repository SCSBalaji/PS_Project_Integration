"""
Comprehensive benchmark script for MobilePlantViT.
"""

import sys
import os
import time
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models import (
    MobilePlantViT,
    mobileplant_vit_tiny,
    mobileplant_vit_small,
    mobileplant_vit_base,
    mobileplant_vit_large,
)


def benchmark_variant(name, model_fn, num_warmup=5, num_runs=20):
    """Benchmark a single model variant."""
    model = model_fn()
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    
    # Parameter count
    params = model.count_parameters()
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)
    
    # Forward benchmark
    forward_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        forward_times.append((time.perf_counter() - start) * 1000)
    
    # Backward benchmark
    model.train()
    target = torch.randint(0, 38, (2,))
    
    backward_times = []
    for _ in range(num_runs // 2):
        model.zero_grad()
        start = time.perf_counter()
        logits = model.get_logits(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        backward_times.append((time.perf_counter() - start) * 1000)
    
    # Throughput
    model.eval()
    batch_x = torch.randn(8, 3, 224, 224)
    
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(batch_x)
    elapsed = time.perf_counter() - start
    throughput = (8 * num_runs) / elapsed
    
    return {
        'name': name,
        'params': params,
        'forward_ms': sum(forward_times) / len(forward_times),
        'backward_ms': sum(backward_times) / len(backward_times),
        'throughput': throughput,
    }


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 70)
    print("MobilePlantViT Benchmark Suite")
    print("=" * 70)
    
    variants = [
        ("Tiny", mobileplant_vit_tiny),
        ("Small", mobileplant_vit_small),
        ("Base", mobileplant_vit_base),
        ("Large", mobileplant_vit_large),
    ]
    
    results = []
    for name, fn in variants:
        print(f"\nBenchmarking {name}...")
        result = benchmark_variant(name, fn)
        results.append(result)
    
    # Print results table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"\n{'Variant':<10} {'Params':>12} {'Forward (ms)':>14} {'Fwd+Bwd (ms)':>14} {'Throughput':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<10} {r['params']:>12,} {r['forward_ms']:>14.2f} {r['backward_ms']:>14.2f} {r['throughput']:>10.1f}/s")
    
    # Parameter breakdown for base model
    print("\n" + "=" * 70)
    print("PARAMETER BREAKDOWN (Base Model)")
    print("=" * 70)
    
    model = mobileplant_vit_base()
    breakdown = model.get_parameter_breakdown()
    
    print(f"\n{'Component':<20} {'Parameters':>15} {'Percentage':>12}")
    print("-" * 50)
    
    components = [
        ('GhostConv', breakdown['ghost_conv']),
        ('Fused-IR', breakdown['fused_ir']),
        ('CoordAtt', breakdown['coord_att']),
        ('PatchEmbed', breakdown['patch_embed']),
        ('PosEnc', breakdown['pos_enc']),
        ('LDA', breakdown['lda']),
        ('ResLN', breakdown['res_ln']),
        ('FFN', breakdown['ffn']),
        ('GAP', breakdown['gap']),
        ('Classifier', breakdown['classifier']),
    ]
    
    total = breakdown['total']
    for name, count in components:
        pct = count / total * 100
        print(f"{name:<20} {count:>15,} {pct:>11.1f}%")
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {total:>15,} {100.0:>11.1f}%")
    print(f"\nBudget: 5,000,000")
    print(f"Usage: {total / 5_000_000 * 100:.1f}%")
    
    # Comparison to other models
    print("\n" + "=" * 70)
    print("COMPARISON TO OTHER ARCHITECTURES")
    print("=" * 70)
    
    comparisons = [
        ("MobilePlantViT-Base", breakdown['total']),
        ("MobileNetV2 (est.)", 3_500_000),
        ("ResNet-18 (est.)", 11_700_000),
        ("ResNet-50 (est.)", 25_600_000),
        ("ViT-Base (est.)", 86_000_000),
        ("EfficientNet-B0 (est.)", 5_300_000),
    ]
    
    print(f"\n{'Model':<25} {'Parameters':>15} {'Relative':>12}")
    print("-" * 55)
    
    base_params = breakdown['total']
    for name, params in comparisons:
        relative = params / base_params
        print(f"{name:<25} {params:>15,} {relative:>11.1f}x")
    
    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()