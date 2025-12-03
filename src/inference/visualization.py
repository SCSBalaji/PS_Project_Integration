"""
Visualization utilities for MobilePlantViT inference results.

This module provides functions to visualize predictions, including:
- Single image prediction display
- Batch prediction grids
- Confidence bar charts
- Preprocessing visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .predictor import InferenceOutput, PredictionResult


# Color scheme
HEALTHY_COLOR = '#27ae60'  # Green
DISEASED_COLOR = '#e74c3c'  # Red
UNCERTAIN_COLOR = '#f39c12'  # Orange
BACKGROUND_COLOR = '#ecf0f1'  # Light gray


def visualize_prediction(
    image: Union[str, Path, Image.Image, np.ndarray],
    prediction: 'InferenceOutput',
    figsize: Tuple[int, int] = (12, 5),
    show_top_k: int = 5,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize a single prediction with image and confidence bars.
    
    Args:
        image: Original image (path or PIL Image)
        prediction: InferenceOutput from predictor
        figsize: Figure size (width, height)
        show_top_k: Number of top predictions to show
        save_path: Path to save figure (optional)
        show: Whether to display the figure
    
    Returns:
        matplotlib Figure if show=False, else None
    """
    # Load image if needed
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot image
    ax1.imshow(image)
    ax1.axis('off')
    
    # Add prediction label on image
    top_pred = prediction.top_prediction
    color = HEALTHY_COLOR if top_pred.is_healthy else DISEASED_COLOR
    
    ax1.set_title(
        f"{top_pred.plant_name}\n{top_pred.condition}\n({top_pred.confidence:.1%})",
        fontsize=12,
        fontweight='bold',
        color=color
    )
    
    # Plot confidence bars
    top_k = prediction.top_k_predictions[:show_top_k]
    
    class_names = [p.class_name.replace('___', '\n').replace('_', ' ') for p in top_k]
    confidences = [p.confidence for p in top_k]
    colors = [HEALTHY_COLOR if p.is_healthy else DISEASED_COLOR for p in top_k]
    
    y_pos = np.arange(len(class_names))
    
    bars = ax2.barh(y_pos, confidences, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names, fontsize=9)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Top Predictions', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    
    # Add confidence values on bars
    for bar, conf in zip(bars, confidences):
        width = bar.get_width()
        ax2.text(
            width + 0.01, bar.get_y() + bar.get_height()/2,
            f'{conf:.1%}',
            va='center', fontsize=9
        )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=HEALTHY_COLOR, lw=4, label='Healthy'),
        Line2D([0], [0], color=DISEASED_COLOR, lw=4, label='Diseased')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_batch_predictions(
    images: List[Union[str, Path, Image.Image]],
    predictions: List['InferenceOutput'],
    cols: int = 4,
    figsize_per_image: Tuple[float, float] = (3, 3.5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize multiple predictions in a grid.
    
    Args:
        images: List of images (paths or PIL Images)
        predictions: List of InferenceOutput from predictor
        cols: Number of columns in grid
        figsize_per_image: Size per image in the grid
        save_path: Path to save figure (optional)
        show: Whether to display the figure
    
    Returns:
        matplotlib Figure if show=False, else None
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    figsize = (figsize_per_image[0] * cols, figsize_per_image[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes for easy iteration
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        if idx < n_images:
            # Load image
            img = images[idx]
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            ax.imshow(img)
            
            # Get prediction
            pred = predictions[idx].top_prediction
            color = HEALTHY_COLOR if pred.is_healthy else DISEASED_COLOR
            
            # Truncate long names
            plant = pred.plant_name[:15] + '...' if len(pred.plant_name) > 15 else pred.plant_name
            cond = pred.condition[:15] + '...' if len(pred.condition) > 15 else pred.condition
            
            ax.set_title(
                f"{plant}\n{cond}\n{pred.confidence:.1%}",
                fontsize=8,
                color=color,
                fontweight='bold'
            )
        
        ax.axis('off')
    
    plt.suptitle('Batch Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved batch visualization to: {save_path}")
    
    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_preprocessing(
    image: Union[str, Path, Image.Image],
    preprocessor: 'PlantImagePreprocessor',
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize the preprocessing pipeline.
    
    Shows: Original → Resized → Normalized → Ready for Model
    
    Args:
        image: Original image
        preprocessor: PlantImagePreprocessor instance
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        matplotlib Figure if show=False, else None
    """
    # Load original
    if isinstance(image, (str, Path)):
        original = Image.open(image).convert('RGB')
    else:
        original = image
    
    # Preprocess
    tensor, _ = preprocessor.preprocess(image, return_original=True)
    
    # Get intermediate steps
    resized = original.resize(preprocessor.input_size, Image.BILINEAR)
    
    # Denormalize for visualization
    denorm_tensor = preprocessor.denormalize(tensor[0])
    denorm_image = preprocessor.tensor_to_image(denorm_tensor, denormalize=False)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    titles = ['Original', 'Resized', 'Normalized\n(visualized)', 'Tensor Stats']
    images_to_show = [original, resized, denorm_image, None]
    
    for ax, title, img in zip(axes[:3], titles[:3], images_to_show[:3]):
        ax.imshow(img)
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
        
        # Add size annotation
        if img is not None:
            w, h = img.size
            ax.text(
                0.5, -0.1, f'{w}×{h}',
                transform=ax.transAxes,
                ha='center', fontsize=9
            )
    
    # Stats panel
    ax = axes[3]
    ax.axis('off')
    
    tensor_np = tensor[0].cpu().numpy()
    stats_text = f"""Tensor Statistics:
    
Shape: {tensor_np.shape}
Dtype: float32

Per Channel:
  R: μ={tensor_np[0].mean():.3f}, σ={tensor_np[0].std():.3f}
  G: μ={tensor_np[1].mean():.3f}, σ={tensor_np[1].std():.3f}
  B: μ={tensor_np[2].mean():.3f}, σ={tensor_np[2].std():.3f}

Range: [{tensor_np.min():.3f}, {tensor_np.max():.3f}]
"""
    
    ax.text(
        0.1, 0.5, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily='monospace',
        verticalalignment='center'
    )
    ax.set_title('Tensor Stats', fontweight='bold')
    
    plt.suptitle('Preprocessing Pipeline', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    else:
        return fig


def create_prediction_report(
    predictions: List['InferenceOutput'],
    output_path: Union[str, Path],
    title: str = "Plant Disease Classification Report"
):
    """
    Create an HTML report of predictions.
    
    Args:
        predictions: List of predictions
        output_path: Path to save HTML report
        title: Report title
    """
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        .summary {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .prediction {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; 
                      display: flex; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .prediction img {{ width: 100px; height: 100px; object-fit: cover; border-radius: 5px; }}
        .details {{ margin-left: 20px; flex: 1; }}
        .class-name {{ font-size: 18px; font-weight: bold; color: #2c3e50; }}
        .confidence {{ font-size: 24px; font-weight: bold; }}
        .healthy {{ color: #27ae60; }}
        .diseased {{ color: #e74c3c; }}
        .bar {{ height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }}
        .bar-fill {{ height: 100%; transition: width 0.3s; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="summary">
            <h3>Summary</h3>
            <p>Total Images: {total}</p>
            <p>Healthy: {healthy} ({healthy_pct:.1f}%)</p>
            <p>Diseased: {diseased} ({diseased_pct:.1f}%)</p>
            <p>Average Confidence: {avg_conf:.1f}%</p>
        </div>
        <h2>Predictions</h2>
        {predictions_html}
    </div>
</body>
</html>
"""
    
    # Calculate summary stats
    total = len(predictions)
    healthy = sum(1 for p in predictions if p.top_prediction.is_healthy)
    diseased = total - healthy
    avg_conf = np.mean([p.top_prediction.confidence for p in predictions]) * 100
    
    # Generate predictions HTML
    predictions_html = ""
    for i, pred in enumerate(predictions):
        top = pred.top_prediction
        status_class = "healthy" if top.is_healthy else "diseased"
        bar_color = HEALTHY_COLOR if top.is_healthy else DISEASED_COLOR
        
        predictions_html += f"""
        <div class="prediction">
            <div class="details">
                <div class="class-name">{top.plant_name} - {top.condition}</div>
                <div class="confidence {status_class}">{top.confidence:.1%}</div>
                <div class="bar">
                    <div class="bar-fill" style="width: {top.confidence*100}%; background: {bar_color};"></div>
                </div>
                <small>Image: {pred.image_path or f'Image {i+1}'}</small>
            </div>
        </div>
        """
    
    # Generate final HTML
    html = html_template.format(
        title=title,
        total=total,
        healthy=healthy,
        healthy_pct=healthy/total*100 if total > 0 else 0,
        diseased=diseased,
        diseased_pct=diseased/total*100 if total > 0 else 0,
        avg_conf=avg_conf,
        predictions_html=predictions_html
    )
    
    # Save
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Report saved to: {output_path}")