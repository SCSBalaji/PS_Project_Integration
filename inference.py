#!/usr/bin/env python
"""
MobilePlantViT Inference CLI

Standalone command-line tool for plant disease classification.

Usage:
    python inference.py --image path/to/image.jpg --checkpoint path/to/model.pth
    python inference.py --image-dir path/to/images/ --checkpoint path/to/model.pth --output results/
    python inference.py --image image.jpg --checkpoint model.pth --visualize
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference import (
    PlantDiseasePredictor,
    visualize_prediction,
    visualize_batch_predictions,
    create_prediction_report,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MobilePlantViT Plant Disease Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python inference.py --image leaf.jpg --checkpoint model.pth
  
  # Batch prediction on directory
  python inference.py --image-dir ./test_images/ --checkpoint model.pth
  
  # With visualization
  python inference.py --image leaf.jpg --checkpoint model.pth --visualize
  
  # Save results to JSON
  python inference.py --image-dir ./images/ --checkpoint model.pth --output results.json
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to a single image file'
    )
    input_group.add_argument(
        '--image-dir', '-d',
        type=str,
        help='Path to directory containing images'
    )
    
    # Model options
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for inference (default: auto)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output path for results (JSON file or directory)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )
    
    # Visualization options
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Show visualization of predictions'
    )
    parser.add_argument(
        '--save-viz',
        type=str,
        help='Save visualization to file'
    )
    
    # Other options
    parser.add_argument(
        '--report',
        type=str,
        help='Generate HTML report at specified path'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except errors'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON to stdout'
    )
    
    return parser.parse_args()


def find_images(directory: Path) -> list:
    """Find all image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    images = []
    
    for ext in image_extensions:
        images.extend(directory.glob(f'*{ext}'))
        images.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(images)


def print_prediction(result, top_k: int = 5, quiet: bool = False):
    """Print prediction result to console."""
    if quiet:
        return
    
    top = result.top_prediction
    status = "✅ HEALTHY" if top.is_healthy else "🔴 DISEASED"
    
    print(f"\n{'='*60}")
    print(f"  {status}")
    print(f"{'='*60}")
    print(f"  Plant:      {top.plant_name}")
    print(f"  Condition:  {top.condition}")
    print(f"  Confidence: {top.confidence:.2%}")
    print(f"  Time:       {result.inference_time_ms:.2f} ms")
    
    if result.image_path:
        print(f"  Image:      {result.image_path}")
    
    print(f"\n  Top {top_k} Predictions:")
    print(f"  {'-'*50}")
    
    for i, pred in enumerate(result.top_k_predictions[:top_k], 1):
        health = "✓" if pred.is_healthy else "✗"
        print(f"  {i}. [{health}] {pred.class_name}: {pred.confidence:.2%}")
    
    print()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set device
    device = None if args.device == 'auto' else args.device
    
    # Load predictor
    if not args.quiet:
        print(f"\n🌿 MobilePlantViT Plant Disease Classifier")
        print(f"{'='*60}")
    
    try:
        predictor = PlantDiseasePredictor(
            checkpoint_path=args.checkpoint,
            device=device
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Collect images
    if args.image:
        images = [Path(args.image)]
        if not images[0].exists():
            print(f"Error: Image not found: {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            print(f"Error: Directory not found: {args.image_dir}", file=sys.stderr)
            sys.exit(1)
        images = find_images(image_dir)
        if not images:
            print(f"Error: No images found in {args.image_dir}", file=sys.stderr)
            sys.exit(1)
        if not args.quiet:
            print(f"\nFound {len(images)} images to process")
    
    # Run inference
    results = []
    
    if len(images) == 1:
        # Single image
        result = predictor.predict(images[0], top_k=args.top_k)
        results.append(result)
        
        if not args.json:
            print_prediction(result, top_k=args.top_k, quiet=args.quiet)
        
        # Visualization
        if args.visualize or args.save_viz:
            from src.inference import visualize_prediction
            visualize_prediction(
                images[0],
                result,
                show_top_k=args.top_k,
                save_path=args.save_viz,
                show=args.visualize
            )
    else:
        # Batch processing
        if not args.quiet:
            print(f"\nProcessing {len(images)} images...")
        
        # Use batch inference for efficiency
        batch_size = 32
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_results = predictor.predict_batch(
                batch_images,
                top_k=args.top_k
            )
            results.extend(batch_results)
            
            if not args.quiet:
                print(f"  Processed {min(i+batch_size, len(images))}/{len(images)} images")
        
        # Print summary
        if not args.json and not args.quiet:
            healthy_count = sum(1 for r in results if r.top_prediction.is_healthy)
            diseased_count = len(results) - healthy_count
            avg_conf = sum(r.top_prediction.confidence for r in results) / len(results)
            avg_time = sum(r.inference_time_ms for r in results) / len(results)
            
            print(f"\n{'='*60}")
            print(f"  BATCH RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"  Total Images:      {len(results)}")
            print(f"  Healthy:           {healthy_count} ({healthy_count/len(results)*100:.1f}%)")
            print(f"  Diseased:          {diseased_count} ({diseased_count/len(results)*100:.1f}%)")
            print(f"  Avg Confidence:    {avg_conf:.2%}")
            print(f"  Avg Inference Time: {avg_time:.2f} ms/image")
            print()
        
        # Batch visualization
        if args.visualize or args.save_viz:
            from src.inference import visualize_batch_predictions
            visualize_batch_predictions(
                images,
                results,
                save_path=args.save_viz,
                show=args.visualize
            )
    
    # JSON output
    if args.json:
        output = {
            'timestamp': datetime.now().isoformat(),
            'model': predictor.model_name,
            'results': [r.to_dict() for r in results]
        }
        print(json.dumps(output, indent=2))
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'model': predictor.model_name,
            'checkpoint': str(args.checkpoint),
            'total_images': len(results),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if not args.quiet:
            print(f"Results saved to: {output_path}")
    
    # Generate HTML report
    if args.report:
        from src.inference import create_prediction_report
        create_prediction_report(
            results,
            args.report,
            title=f"Plant Disease Classification Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())