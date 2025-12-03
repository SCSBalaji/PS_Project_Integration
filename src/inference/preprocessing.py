"""
Image preprocessing utilities for MobilePlantViT inference.

This module provides preprocessing functions that match the training pipeline,
ensuring consistent results between training and inference.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Union, List, Tuple, Optional
from pathlib import Path


# Default ImageNet normalization (used during training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default input size
DEFAULT_INPUT_SIZE = (224, 224)


class PlantImagePreprocessor:
    """
    Preprocessor for plant leaf images.
    
    Handles image loading, resizing, normalization, and tensor conversion
    to match the training pipeline exactly.
    
    Args:
        input_size: Target image size (height, width). Default: (224, 224)
        mean: Normalization mean values. Default: ImageNet mean
        std: Normalization std values. Default: ImageNet std
        device: Target device for tensors. Default: auto-detect
    
    Example:
        >>> preprocessor = PlantImagePreprocessor()
        >>> tensor = preprocessor.preprocess("path/to/leaf.jpg")
        >>> batch = preprocessor.preprocess_batch(["img1.jpg", "img2.jpg"])
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        mean: List[float] = None,
        std: List[float] = None,
        device: Optional[torch.device] = None
    ):
        self.input_size = input_size
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build transform pipeline (matches val/test transforms from training)
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def load_image(self, image_source: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load image from various sources.
        
        Args:
            image_source: Can be:
                - str or Path: File path to image
                - PIL.Image: Already loaded PIL image
                - np.ndarray: NumPy array (RGB, HWC format)
        
        Returns:
            PIL.Image in RGB format
        
        Raises:
            ValueError: If image source type is not supported
            FileNotFoundError: If image file doesn't exist
        """
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            image = Image.open(path)
        elif isinstance(image_source, Image.Image):
            image = image_source
        elif isinstance(image_source, np.ndarray):
            # Assume RGB, HWC format
            if image_source.ndim == 2:
                # Grayscale, convert to RGB
                image = Image.fromarray(image_source).convert('RGB')
            elif image_source.ndim == 3:
                if image_source.shape[2] == 4:
                    # RGBA, convert to RGB
                    image = Image.fromarray(image_source[:, :, :3])
                else:
                    image = Image.fromarray(image_source)
            else:
                raise ValueError(f"Unsupported array shape: {image_source.shape}")
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def preprocess(
        self, 
        image_source: Union[str, Path, Image.Image, np.ndarray],
        return_original: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Image.Image]]:
        """
        Preprocess a single image for inference.
        
        Args:
            image_source: Image to preprocess (path, PIL Image, or numpy array)
            return_original: If True, also return the original PIL image
        
        Returns:
            If return_original=False: Tensor of shape (1, 3, H, W)
            If return_original=True: Tuple of (tensor, original_image)
        """
        # Load image
        image = self.load_image(image_source)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)
        
        if return_original:
            return tensor, image
        return tensor
    
    def preprocess_batch(
        self,
        image_sources: List[Union[str, Path, Image.Image, np.ndarray]],
        return_originals: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Image.Image]]]:
        """
        Preprocess multiple images for batch inference.
        
        Args:
            image_sources: List of images to preprocess
            return_originals: If True, also return original PIL images
        
        Returns:
            If return_originals=False: Tensor of shape (N, 3, H, W)
            If return_originals=True: Tuple of (tensor, list of original images)
        """
        tensors = []
        originals = []
        
        for source in image_sources:
            if return_originals:
                tensor, original = self.preprocess(source, return_original=True)
                originals.append(original)
            else:
                tensor = self.preprocess(source, return_original=False)
            tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.cat(tensors, dim=0)
        
        if return_originals:
            return batch_tensor, originals
        return batch_tensor
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization for visualization.
        
        Args:
            tensor: Normalized tensor of shape (C, H, W) or (N, C, H, W)
        
        Returns:
            Denormalized tensor with values in [0, 1]
        """
        tensor = tensor.clone()
        
        # Handle batch dimension
        if tensor.dim() == 4:
            for i in range(tensor.shape[0]):
                for c, (m, s) in enumerate(zip(self.mean, self.std)):
                    tensor[i, c] = tensor[i, c] * s + m
        else:
            for c, (m, s) in enumerate(zip(self.mean, self.std)):
                tensor[c] = tensor[c] * s + m
        
        return torch.clamp(tensor, 0, 1)
    
    def tensor_to_image(self, tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
        """
        Convert tensor back to PIL Image.
        
        Args:
            tensor: Tensor of shape (C, H, W) or (1, C, H, W)
            denormalize: Whether to reverse normalization
        
        Returns:
            PIL Image
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # Remove batch dimension
        
        if denormalize:
            tensor = self.denormalize(tensor)
        
        # Move to CPU and convert to numpy
        array = tensor.cpu().permute(1, 2, 0).numpy()
        array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array)


def get_default_preprocessor(device: Optional[torch.device] = None) -> PlantImagePreprocessor:
    """
    Get a preprocessor with default settings matching training.
    
    Args:
        device: Target device. Default: auto-detect
    
    Returns:
        Configured PlantImagePreprocessor instance
    """
    return PlantImagePreprocessor(
        input_size=DEFAULT_INPUT_SIZE,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        device=device
    )