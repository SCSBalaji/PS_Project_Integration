"""
Quick test script for the inference pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    from src.inference import (
        PlantDiseasePredictor,
        PredictionResult,
        InferenceOutput,
        load_predictor,
        PlantImagePreprocessor,
        get_default_preprocessor,
        visualize_prediction,
        visualize_batch_predictions,
        create_prediction_report,
    )
    
    print("✅ All imports successful!")
    return True


def test_preprocessor():
    """Test preprocessor functionality."""
    print("\nTesting preprocessor...")
    
    from src.inference import get_default_preprocessor
    import numpy as np
    from PIL import Image
    
    preprocessor = get_default_preprocessor()
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (256, 256), color='green')
    
    # Test preprocessing
    tensor = preprocessor.preprocess(dummy_image)
    
    assert tensor.shape == (1, 3, 224, 224), f"Unexpected shape: {tensor.shape}"
    print(f"✅ Preprocessor works! Output shape: {tensor.shape}")
    
    # Test denormalization
    denorm = preprocessor.denormalize(tensor[0])
    assert denorm.min() >= 0 and denorm.max() <= 1, "Denormalization failed"
    print("✅ Denormalization works!")
    
    return True


def test_predictor_loading():
    """Test predictor loading (if checkpoint exists)."""
    print("\nTesting predictor loading...")
    
    from src.inference import PlantDiseasePredictor
    
    checkpoint_paths = [
        project_root / "outputs" / "checkpoints" / "MobilePlantViT-Base_best.pth",
        project_root / "outputs" / "exports" / "mobileplant_vit_full_checkpoint.pth",
    ]
    
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            predictor = PlantDiseasePredictor(checkpoint_path)
            print(f"✅ Predictor loaded from: {checkpoint_path}")
            print(f"   Model: {predictor.model_name}")
            print(f"   Classes: {predictor.num_classes}")
            return True
    
    print("⚠️  No checkpoint found - skipping predictor test")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("  MobilePlantViT Inference Pipeline Tests")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Preprocessor Test", test_preprocessor),
        ("Predictor Loading Test", test_predictor_loading),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'✅ All tests passed!' if all_passed else '❌ Some tests failed'}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())