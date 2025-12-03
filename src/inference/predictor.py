"""
Main inference predictor for MobilePlantViT plant disease classification.

This module provides a high-level interface for loading trained models
and making predictions on plant leaf images.
"""

import torch
import torch.nn.functional as F
import json
import os
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

from .preprocessing import PlantImagePreprocessor, get_default_preprocessor


@dataclass
class PredictionResult:
    """
    Container for a single prediction result.
    
    Attributes:
        class_name: Predicted class name (e.g., "Tomato___Late_blight")
        class_index: Predicted class index (0-37)
        confidence: Confidence score (0-1)
        plant_name: Extracted plant name (e.g., "Tomato")
        condition: Extracted condition (e.g., "Late_blight" or "healthy")
        is_healthy: Whether the prediction is a healthy class
    """
    class_name: str
    class_index: int
    confidence: float
    plant_name: str
    condition: str
    is_healthy: bool
    
    def __repr__(self):
        status = "✅ Healthy" if self.is_healthy else "🔴 Diseased"
        return f"PredictionResult({self.class_name}, {self.confidence:.2%}, {status})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'class_name': self.class_name,
            'class_index': self.class_index,
            'confidence': self.confidence,
            'plant_name': self.plant_name,
            'condition': self.condition,
            'is_healthy': self.is_healthy,
        }


@dataclass
class InferenceOutput:
    """
    Container for complete inference output.
    
    Attributes:
        top_prediction: The highest confidence prediction
        top_k_predictions: List of top-k predictions
        probabilities: Full probability distribution (numpy array)
        inference_time_ms: Time taken for inference in milliseconds
        image_path: Source image path (if available)
    """
    top_prediction: PredictionResult
    top_k_predictions: List[PredictionResult]
    probabilities: Any  # numpy array
    inference_time_ms: float
    image_path: Optional[str] = None
    
    def __repr__(self):
        return f"InferenceOutput(top={self.top_prediction.class_name}, conf={self.top_prediction.confidence:.2%})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'top_prediction': self.top_prediction.to_dict(),
            'top_k_predictions': [p.to_dict() for p in self.top_k_predictions],
            'probabilities': self.probabilities.tolist() if hasattr(self.probabilities, 'tolist') else self.probabilities,
            'inference_time_ms': self.inference_time_ms,
            'image_path': self.image_path,
        }


class PlantDiseasePredictor:
    """
    High-level predictor for plant disease classification.
    
    This class provides an easy-to-use interface for loading MobilePlantViT
    models and making predictions on plant leaf images.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Target device ('cuda', 'cpu', or torch.device). Default: auto-detect
        use_torchscript: If True, try to load TorchScript model for faster inference
    
    Example:
        >>> predictor = PlantDiseasePredictor("path/to/checkpoint.pth")
        >>> result = predictor.predict("path/to/leaf_image.jpg")
        >>> print(f"Prediction: {result.top_prediction.class_name}")
        >>> print(f"Confidence: {result.top_prediction.confidence:.2%}")
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device, None] = None,
        use_torchscript: bool = False
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.use_torchscript = use_torchscript
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Initialize model and metadata
        self.model = None
        self.class_names = None
        self.num_classes = None
        self.model_name = None
        self.preprocessor = None
        
        # Load model
        self._load_model()
        
        # Initialize preprocessor
        self.preprocessor = get_default_preprocessor(device=self.device)
    
    def _load_model(self):
        """Load model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        print(f"Loading model from: {self.checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Check for TorchScript model first
        if self.use_torchscript:
            torchscript_path = self.checkpoint_path.parent / 'mobileplant_vit_traced.pt'
            if torchscript_path.exists():
                self._load_torchscript(torchscript_path)
                return
            else:
                warnings.warn(f"TorchScript model not found at {torchscript_path}, falling back to checkpoint")
        
        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path, 
            map_location=self.device,
            weights_only=False  # Allow loading full checkpoint with config
        )
        
        # Extract metadata
        self.class_names = checkpoint.get('class_names', [])
        self.num_classes = checkpoint.get('num_classes', len(self.class_names))
        self.model_name = checkpoint.get('model_name', 'MobilePlantViT')
        model_type = checkpoint.get('model_type', 'mobileplant_vit')
        
        # Reconstruct model
        if model_type == 'mobileplant_vit':
            self._load_mobileplant_vit(checkpoint)
        elif model_type == 'mobilenet_v2':
            self._load_mobilenet_v2(checkpoint)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.eval()
        print(f"✅ Model loaded: {self.model_name}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_mobileplant_vit(self, checkpoint: Dict):
        """Load MobilePlantViT model."""
        try:
            # Try to import MobilePlantViT
            from src.models import MobilePlantViT, MobilePlantViTConfig
            
            model_config = checkpoint.get('model_config', None)
            
            if model_config is not None:
                # Reconstruct from saved config
                config = MobilePlantViTConfig(**model_config)
                self.model = MobilePlantViT(config)
            else:
                # Fallback to base variant
                from src.models import mobileplant_vit_base
                self.model = mobileplant_vit_base(num_classes=self.num_classes)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self._model_type = 'mobileplant_vit'
            
        except ImportError as e:
            raise ImportError(
                f"Could not import MobilePlantViT. Make sure src/models is in your Python path.\n"
                f"Error: {e}"
            )
    
    def _load_mobilenet_v2(self, checkpoint: Dict):
        """Load MobileNetV2 model."""
        from torchvision import models
        import torch.nn as nn
        
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            self.num_classes
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self._model_type = 'mobilenet_v2'
    
    def _load_torchscript(self, path: Path):
        """Load TorchScript model."""
        print(f"Loading TorchScript model from: {path}")
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()
        self._model_type = 'torchscript'
        
        # Try to load metadata from deployment_metadata.json
        metadata_path = path.parent / 'deployment_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.class_names = metadata.get('class_names', [])
            self.num_classes = metadata.get('num_classes', len(self.class_names))
            self.model_name = metadata.get('model_name', 'MobilePlantViT (TorchScript)')
    
    def _parse_class_name(self, class_name: str) -> Tuple[str, str, bool]:
        """
        Parse class name into plant, condition, and health status.
        
        Args:
            class_name: Full class name (e.g., "Tomato___Late_blight")
        
        Returns:
            Tuple of (plant_name, condition, is_healthy)
        """
        # Split by triple underscore (PlantVillage format)
        if '___' in class_name:
            parts = class_name.split('___', 1)
            plant_name = parts[0].replace('_', ' ')
            condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
        else:
            plant_name = class_name.replace('_', ' ')
            condition = 'Unknown'
        
        # Check if healthy
        is_healthy = 'healthy' in condition.lower()
        
        return plant_name, condition, is_healthy
    
    def _create_prediction_result(
        self, 
        class_index: int, 
        confidence: float
    ) -> PredictionResult:
        """Create a PredictionResult from class index and confidence."""
        class_name = self.class_names[class_index] if class_index < len(self.class_names) else f"class_{class_index}"
        plant_name, condition, is_healthy = self._parse_class_name(class_name)
        
        return PredictionResult(
            class_name=class_name,
            class_index=class_index,
            confidence=confidence,
            plant_name=plant_name,
            condition=condition,
            is_healthy=is_healthy
        )
    
    @torch.no_grad()
    def predict(
        self,
        image_source: Union[str, Path, 'Image.Image', 'np.ndarray'],
        top_k: int = 5,
        return_probabilities: bool = True
    ) -> InferenceOutput:
        """
        Make prediction on a single image.
        
        Args:
            image_source: Image to classify (path, PIL Image, or numpy array)
            top_k: Number of top predictions to return
            return_probabilities: Whether to include full probability distribution
        
        Returns:
            InferenceOutput with predictions and metadata
        """
        import time
        import numpy as np
        
        # Preprocess image
        tensor = self.preprocessor.preprocess(image_source)
        
        # Track image path if available
        image_path = str(image_source) if isinstance(image_source, (str, Path)) else None
        
        # Run inference
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Get model output
        if self._model_type == 'mobileplant_vit':
            # MobilePlantViT forward() returns probabilities
            probabilities = self.model(tensor)
        else:
            # Other models return logits
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Convert to numpy
        probs_np = probabilities.cpu().numpy()[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probs_np)[-top_k:][::-1]
        
        top_k_predictions = [
            self._create_prediction_result(int(idx), float(probs_np[idx]))
            for idx in top_k_indices
        ]
        
        return InferenceOutput(
            top_prediction=top_k_predictions[0],
            top_k_predictions=top_k_predictions,
            probabilities=probs_np if return_probabilities else None,
            inference_time_ms=inference_time_ms,
            image_path=image_path
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        image_sources: List[Union[str, Path, 'Image.Image', 'np.ndarray']],
        top_k: int = 5,
        return_probabilities: bool = False
    ) -> List[InferenceOutput]:
        """
        Make predictions on multiple images.
        
        Args:
            image_sources: List of images to classify
            top_k: Number of top predictions per image
            return_probabilities: Whether to include probability distributions
        
        Returns:
            List of InferenceOutput, one per image
        """
        import time
        import numpy as np
        
        # Preprocess all images
        batch_tensor = self.preprocessor.preprocess_batch(image_sources)
        
        # Run batch inference
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        if self._model_type == 'mobileplant_vit':
            probabilities = self.model(batch_tensor)
        else:
            logits = self.model(batch_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time_ms = (time.time() - start_time) * 1000
        per_image_time_ms = total_time_ms / len(image_sources)
        
        # Convert to numpy
        probs_np = probabilities.cpu().numpy()
        
        # Create results for each image
        results = []
        for i, image_source in enumerate(image_sources):
            image_probs = probs_np[i]
            top_k_indices = np.argsort(image_probs)[-top_k:][::-1]
            
            top_k_predictions = [
                self._create_prediction_result(int(idx), float(image_probs[idx]))
                for idx in top_k_indices
            ]
            
            image_path = str(image_source) if isinstance(image_source, (str, Path)) else None
            
            results.append(InferenceOutput(
                top_prediction=top_k_predictions[0],
                top_k_predictions=top_k_predictions,
                probabilities=image_probs if return_probabilities else None,
                inference_time_ms=per_image_time_ms,
                image_path=image_path
            ))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'model_type': self._model_type,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'checkpoint_path': str(self.checkpoint_path),
        }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.class_names.copy()
    
    def get_healthy_classes(self) -> List[str]:
        """Get list of healthy class names."""
        return [c for c in self.class_names if 'healthy' in c.lower()]
    
    def get_diseased_classes(self) -> List[str]:
        """Get list of diseased class names."""
        return [c for c in self.class_names if 'healthy' not in c.lower()]


def load_predictor(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None
) -> PlantDiseasePredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Target device ('cuda', 'cpu', or None for auto)
    
    Returns:
        Configured PlantDiseasePredictor instance
    """
    return PlantDiseasePredictor(
        checkpoint_path=checkpoint_path,
        device=device
    )