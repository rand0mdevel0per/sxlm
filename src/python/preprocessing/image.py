"""Image preprocessing for VQ-GAN encoding"""

import torch
from typing import Tuple

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    def preprocess(self, image_path: str) -> torch.Tensor:
        """Preprocess image for VQ-GAN encoding"""
        # Simplified: return random tensor
        return torch.randn(3, *self.target_size)

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image to [-1, 1]"""
        return (image - 0.5) * 2
