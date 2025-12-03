"""
MobilePlantViT Inference Demo

This script demonstrates various ways to use the inference pipeline.
Run this after training is complete and you have a checkpoint file.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import (
    PlantDiseasePredictor,
    visualize_prediction,
    visualize_batch_predictions,
    visualize_preprocessing,
    get_default_preprocessor,
)


def demo_single_image_prediction():
    """Demonstrate single image prediction."""
    print("\n" + "="*60)
    print("  Demo 1: Single Image Prediction")
    print("="*60)
    
    # Try multiple checkpoint locations
    checkpoint_paths = [
        project_root / "outputs" / "checkpoints" / "MobilePlantViT-Base_best.pth",
        project_root / "outputs" / "exports" / "mobileplant_vit_full_checkpoint.pth",
        project_root / "outputs" / "checkpoints" / "best_model.pth",
    ]
    
    checkpoint_path = None
    for cp in checkpoint_paths:
        if cp.exists():
            checkpoint_path = cp
            break
    
    if checkpoint_path is None:
        print("⚠️  No checkpoint found. Checked locations:")
        for cp in checkpoint_paths:
            print(f"   - {cp}")
        return None
    
    # Load predictor
    print(f"\n📂 Loading model from: {checkpoint_path}")
    predictor = PlantDiseasePredictor(checkpoint_path)
    
    # Show model info
    print("\n📊 Model Information:")
    info = predictor.get_model_info()
    for key, value in info.items():
        if key != 'class_names':
            print(f"   {key}: {value}")
    
    print(f"\n📋 Sample Classes ({min(10, len(predictor.class_names))} of {len(predictor.class_names)}):")
    for i, name in enumerate(predictor.class_names[:10]):
        print(f"   {i}: {name}")
    if len(predictor.class_names) > 10:
        print(f"   ... and {len(predictor.class_names) - 10} more")
    
    return predictor


def demo_with_test_image(predictor, image_path):
    """Run prediction on a test image."""
    if predictor is None:
        print("⚠️  No predictor loaded.")
        return
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"⚠️  Image not found: {image_path}")
        return
    
    # Make prediction
    print(f"\n🔍 Predicting on: {image_path}")
    result = predictor.predict(image_path)
    
    # Display results
    print(f"\n📊 Prediction Results:")
    print(f"   Top prediction: {result.top_prediction.class_name}")
    print(f"   Confidence: {result.top_prediction.confidence:.2%}")
    print(f"   Plant: {result.top_prediction.plant_name}")
    print(f"   Condition: {result.top_prediction.condition}")
    print(f"   Is Healthy: {'Yes ✅' if result.top_prediction.is_healthy else 'No 🔴'}")
    print(f"   Inference Time: {result.inference_time_ms:.2f} ms")
    
    print(f"\n   Top 5 Predictions:")
    for i, pred in enumerate(result.top_k_predictions, 1):
        status = "✅" if pred.is_healthy else "🔴"
        print(f"   {i}. {status} {pred.class_name}: {pred.confidence:.2%}")
    
    return result


def demo_preprocessing():
    """Demonstrate preprocessing pipeline."""
    print("\n" + "="*60)
    print("  Demo: Preprocessing Pipeline")
    print("="*60)
    
    preprocessor = get_default_preprocessor()
    
    print(f"\n📊 Preprocessor Configuration:")
    print(f"   Input Size: {preprocessor.input_size}")
    print(f"   Mean: {preprocessor.mean}")
    print(f"   Std: {preprocessor.std}")
    print(f"   Device: {preprocessor.device}")
    
    return preprocessor


def main():
    """Main demo function."""
    print("\n" + "="*60)
    print("  🌿 MobilePlantViT Inference Demo")
    print("="*60)
    
    # Demo 1: Load model and show info
    predictor = demo_single_image_prediction()
    
    # Demo 2: Preprocessing info
    preprocessor = demo_preprocessing()
    
    # Interactive mode
    if predictor is not None:
        print("\n" + "="*60)
        print("  Interactive Mode")
        print("="*60)
        print("\nYou can now test with your own images.")
        print("Example usage in Python:")
        print("  result = predictor.predict('path/to/your/image.jpg')")
        print("  visualize_prediction('path/to/image.jpg', result)")
    
    return predictor, preprocessor


if __name__ == '__main__':
    predictor, preprocessor = main()