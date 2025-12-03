"""
MobilePlantViT Inference Module

This module provides easy-to-use utilities for plant disease classification
using trained MobilePlantViT models.

Quick Start:
    >>> from src.inference import PlantDiseasePredictor
    >>> predictor = PlantDiseasePredictor("path/to/checkpoint.pth")
    >>> result = predictor.predict("path/to/leaf.jpg")
    >>> print(result.top_prediction)

Modules:
    - predictor: Main inference class
    - preprocessing: Image preprocessing utilities
    - visualization: Result visualization functions
"""

from .predictor import (
    PlantDiseasePredictor,
    PredictionResult,
    InferenceOutput,
    load_predictor,
)

from .preprocessing import (
    PlantImagePreprocessor,
    get_default_preprocessor,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_INPUT_SIZE,
)

from .visualization import (
    visualize_prediction,
    visualize_batch_predictions,
    visualize_preprocessing,
    create_prediction_report,
)

__all__ = [
    # Predictor
    'PlantDiseasePredictor',
    'PredictionResult',
    'InferenceOutput',
    'load_predictor',
    
    # Preprocessing
    'PlantImagePreprocessor',
    'get_default_preprocessor',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'DEFAULT_INPUT_SIZE',
    
    # Visualization
    'visualize_prediction',
    'visualize_batch_predictions',
    'visualize_preprocessing',
    'create_prediction_report',
]

__version__ = '1.0.0'