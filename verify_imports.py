"""
Verification script for MobilePlantViT package structure.
Run this script to verify all imports work correctly before uploading to Kaggle.

Usage:
    python verify_imports.py
"""

import sys
import os

# Add project root to path (simulates what we'll do in Kaggle)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("  MobilePlantViT Import Verification")
print("=" * 70)
print(f"\nProject root: {project_root}")
print(f"Python path includes project: {project_root in sys.path}")

# Track success/failure
all_passed = True
results = []

# ============================================================================
# TEST 1: Import individual blocks from their modules
# ============================================================================
print("\n" + "-" * 70)
print("TEST 1: Importing individual blocks from their source files")
print("-" * 70)

blocks_to_test = [
    ("GhostConv", "ghost_conv"),
    ("FusedInvertedResidualBlock", "fused_ir"),
    ("CoordAtt", "coord_attention"),
    ("PatchEmbedding", "patch_embed"),
    ("PositionalEncoding", "positional_encoding"),
    ("LinearDifferentialAttention", "attention"),
    ("ResidualLayerNormBlock", "ffn"),
    ("BottleneckFFN", "ffn"),
    ("GlobalAveragePooling", "classifier"),
    ("ClassifierHead", "classifier"),
]

for class_name, module_name in blocks_to_test:
    try:
        exec(f"from src.blocks.{module_name} import {class_name}")
        print(f"  ✅ {class_name} from src.blocks.{module_name}")
        results.append((f"blocks.{module_name}.{class_name}", True, None))
    except Exception as e:
        print(f"  ❌ {class_name} from src.blocks.{module_name}: {e}")
        results.append((f"blocks.{module_name}.{class_name}", False, str(e)))
        all_passed = False

# ============================================================================
# TEST 2: Import all blocks from src.blocks (via __init__.py)
# ============================================================================
print("\n" + "-" * 70)
print("TEST 2: Importing all blocks from src.blocks (via __init__.py)")
print("-" * 70)

try:
    from src.blocks import (
        GhostConv,
        FusedInvertedResidualBlock,
        CoordAtt,
        PatchEmbedding,
        PositionalEncoding,
        LinearDifferentialAttention,
        ResidualLayerNormBlock,
        BottleneckFFN,
        GlobalAveragePooling,
        ClassifierHead,
    )
    print("  ✅ All blocks imported successfully from src.blocks")
    results.append(("src.blocks (all)", True, None))
except Exception as e:
    print(f"  ❌ Failed to import from src.blocks: {e}")
    results.append(("src.blocks (all)", False, str(e)))
    all_passed = False

# ============================================================================
# TEST 3: Import MobilePlantViT model
# ============================================================================
print("\n" + "-" * 70)
print("TEST 3: Importing MobilePlantViT from src.models")
print("-" * 70)

try:
    from src.models import MobilePlantViT, MobilePlantViTConfig
    print("  ✅ MobilePlantViT imported successfully")
    print("  ✅ MobilePlantViTConfig imported successfully")
    results.append(("MobilePlantViT", True, None))
    results.append(("MobilePlantViTConfig", True, None))
except Exception as e:
    print(f"  ❌ Failed to import MobilePlantViT: {e}")
    results.append(("MobilePlantViT", False, str(e)))
    all_passed = False

# ============================================================================
# TEST 4: Import factory functions
# ============================================================================
print("\n" + "-" * 70)
print("TEST 4: Importing factory functions")
print("-" * 70)

factory_functions = [
    "mobileplant_vit_tiny",
    "mobileplant_vit_small",
    "mobileplant_vit_base",
    "mobileplant_vit_large",
]

for func_name in factory_functions:
    try:
        exec(f"from src.models import {func_name}")
        print(f"  ✅ {func_name}")
        results.append((func_name, True, None))
    except Exception as e:
        print(f"  ❌ {func_name}: {e}")
        results.append((func_name, False, str(e)))
        all_passed = False

# ============================================================================
# TEST 5: Import from top-level src package
# ============================================================================
print("\n" + "-" * 70)
print("TEST 5: Importing from top-level src package")
print("-" * 70)

try:
    from src import MobilePlantViT, mobileplant_vit_base
    print("  ✅ Direct import from src package works")
    results.append(("src (top-level)", True, None))
except Exception as e:
    print(f"  ❌ Top-level import failed: {e}")
    results.append(("src (top-level)", False, str(e)))
    all_passed = False

# ============================================================================
# TEST 6: Model instantiation
# ============================================================================
print("\n" + "-" * 70)
print("TEST 6: Model instantiation test")
print("-" * 70)

try:
    import torch
    from src.models import MobilePlantViT, mobileplant_vit_base
    
    # Test default instantiation
    model = MobilePlantViT()
    print(f"  ✅ MobilePlantViT() instantiated")
    print(f"     Parameters: {model.count_parameters():,}")
    
    # Test factory function
    model_base = mobileplant_vit_base(num_classes=38)
    print(f"  ✅ mobileplant_vit_base(num_classes=38) instantiated")
    print(f"     Parameters: {model_base.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model_base(x)
    print(f"  ✅ Forward pass successful")
    print(f"     Input shape:  {tuple(x.shape)}")
    print(f"     Output shape: {tuple(output.shape)}")
    
    results.append(("Model instantiation", True, None))
    results.append(("Forward pass", True, None))
    
except Exception as e:
    print(f"  ❌ Model instantiation/forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    results.append(("Model instantiation", False, str(e)))
    all_passed = False

# ============================================================================
# TEST 7: Parameter breakdown
# ============================================================================
print("\n" + "-" * 70)
print("TEST 7: Parameter breakdown test")
print("-" * 70)

try:
    from src.models import mobileplant_vit_base
    model = mobileplant_vit_base(num_classes=38)
    
    breakdown = model.get_parameter_breakdown()
    print(f"  ✅ Parameter breakdown retrieved")
    print(f"     CNN Stage:         {breakdown['cnn_total']:,}")
    print(f"     Transition Stage:  {breakdown['transition_total']:,}")
    print(f"     Transformer Stage: {breakdown['transformer_total']:,}")
    print(f"     Classifier Stage:  {breakdown['classifier_total']:,}")
    print(f"     Total:             {breakdown['total']:,}")
    
    results.append(("Parameter breakdown", True, None))
except Exception as e:
    print(f"  ❌ Parameter breakdown failed: {e}")
    results.append(("Parameter breakdown", False, str(e)))
    all_passed = False

# ============================================================================
# TEST 8: Test all model variants
# ============================================================================
print("\n" + "-" * 70)
print("TEST 8: Testing all model variants")
print("-" * 70)

try:
    import torch
    from src.models import (
        mobileplant_vit_tiny,
        mobileplant_vit_small,
        mobileplant_vit_base,
        mobileplant_vit_large,
    )
    
    variants = [
        ("tiny", mobileplant_vit_tiny),
        ("small", mobileplant_vit_small),
        ("base", mobileplant_vit_base),
        ("large", mobileplant_vit_large),
    ]
    
    x = torch.randn(1, 3, 224, 224)
    
    for name, factory_fn in variants:
        model = factory_fn(num_classes=38)
        with torch.no_grad():
            output = model(x)
        params = model.count_parameters()
        print(f"  ✅ {name:6s}: {params:>10,} params, output shape: {tuple(output.shape)}")
    
    results.append(("All variants", True, None))
except Exception as e:
    print(f"  ❌ Variant testing failed: {e}")
    results.append(("All variants", False, str(e)))
    all_passed = False

# ============================================================================
# TEST 9: Test get_logits method (for CrossEntropyLoss)
# ============================================================================
print("\n" + "-" * 70)
print("TEST 9: Testing get_logits() method")
print("-" * 70)

try:
    import torch
    from src.models import mobileplant_vit_base
    
    model = mobileplant_vit_base(num_classes=38)
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        logits = model.get_logits(x)
        probs = model(x)
    
    print(f"  ✅ get_logits() returns shape: {tuple(logits.shape)}")
    print(f"  ✅ forward() returns shape:    {tuple(probs.shape)}")
    print(f"     Logits sum (not 1):  {logits[0].sum().item():.4f}")
    print(f"     Probs sum (should be 1): {probs[0].sum().item():.4f}")
    
    results.append(("get_logits method", True, None))
except Exception as e:
    print(f"  ❌ get_logits() test failed: {e}")
    results.append(("get_logits method", False, str(e)))
    all_passed = False

# ============================================================================
# TEST 10: Test intermediate outputs
# ============================================================================
print("\n" + "-" * 70)
print("TEST 10: Testing get_intermediate_outputs() method")
print("-" * 70)

try:
    import torch
    from src.models import mobileplant_vit_base
    
    model = mobileplant_vit_base(num_classes=38)
    x = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        intermediates = model.get_intermediate_outputs(x)
    
    print(f"  ✅ Intermediate outputs retrieved ({len(intermediates)} stages)")
    for stage_name, tensor in intermediates.items():
        print(f"     {stage_name:20s}: {tuple(tensor.shape)}")
    
    results.append(("Intermediate outputs", True, None))
except Exception as e:
    print(f"  ❌ Intermediate outputs test failed: {e}")
    results.append(("Intermediate outputs", False, str(e)))
    all_passed = False

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  VERIFICATION SUMMARY")
print("=" * 70)

passed = sum(1 for _, success, _ in results if success)
failed = sum(1 for _, success, _ in results if not success)

print(f"\n  Total tests: {len(results)}")
print(f"  ✅ Passed:   {passed}")
print(f"  ❌ Failed:   {failed}")

if all_passed:
    print("\n  🎉 ALL TESTS PASSED! Package is ready for Kaggle upload.")
else:
    print("\n  ⚠️  SOME TESTS FAILED! Please fix the issues above.")
    print("\n  Failed tests:")
    for name, success, error in results:
        if not success:
            print(f"    - {name}: {error}")

print("\n" + "=" * 70)

# Exit with appropriate code
sys.exit(0 if all_passed else 1)