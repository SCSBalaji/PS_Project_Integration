"""
Preprocessing transforms for training and inference.
"""

import torchvision.transforms as transforms


class Preprocessing:
    """
    Image preprocessing pipeline.
    
    Args:
        img_size (int): Target image size.
        training (bool): Whether to apply training augmentations.
    """
    def __init__(self, img_size: int = 224, training: bool = True):
        base_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        if training:
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ]
            self.transform = transforms.Compose(augmentations + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __call__(self, img):
        return self.transform(img)