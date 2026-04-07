"""
All image transformations and augmentations.

Contains:
    - get_clip_transforms(): Standard CLIP preprocessing
    - compute_dct_map(): RGB -> DCT spectral map conversion
    - get_train_transforms(): Training augmentations
    - get_eval_transforms(): Evaluation transforms (no augmentation)
"""

import io
import random
from typing import Optional

import numpy as np
import scipy.fft as fft
import torch
from PIL import Image, ImageFilter
from torchvision import transforms


def compute_dct_map(image_pil: Image.Image) -> np.ndarray:
    """Convert PIL image to 2D DCT spectral map.

    Steps:
        1. Convert to grayscale via luminance formula
        2. Apply 2D DCT (type-II, orthonormalized)
        3. Take abs() + log1p() for dynamic range compression
        4. Min-max normalize to [0, 1]

    The resulting map highlights frequency-domain artifacts that
    differ between real photographs and AI-generated images.

    Args:
        image_pil: PIL Image (any mode, will be converted to grayscale).

    Returns:
        numpy array of shape (H, W) with values in [0, 1].
    """
    # Convert to grayscale
    gray = np.array(image_pil.convert("L"), dtype=np.float32)

    # Apply 2D DCT (type-II, orthonormalized)
    dct_coeffs = fft.dctn(gray, type=2, norm="ortho")

    # Dynamic range compression
    dct_map = np.log1p(np.abs(dct_coeffs))

    # Min-max normalize to [0, 1]
    dct_min = dct_map.min()
    dct_max = dct_map.max()
    if dct_max - dct_min > 0:
        dct_map = (dct_map - dct_min) / (dct_max - dct_min)
    else:
        dct_map = np.zeros_like(dct_map)

    return dct_map


def get_clip_transforms(image_size: int = 224) -> transforms.Compose:
    """Get standard CLIP-compatible image transforms.

    Args:
        image_size: Target image size (default 224).

    Returns:
        torchvision Compose transform.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Get training transforms with light augmentation.

    Args:
        image_size: Target image size.

    Returns:
        torchvision Compose transform.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Get evaluation transforms (no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        torchvision Compose transform.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


class RobustnessAugmentation:
    """Apply random degradations to simulate real-world image processing.

    Simulates:
        - Social media JPEG recompression
        - Messaging app blur
        - Screenshot/resize artifacts

    Args:
        jpeg_q_range: Min/max JPEG quality for random compression.
        blur_sigma_range: Min/max sigma for Gaussian blur.
        downscale_size: Size to downscale to before upscaling back.
        prob: Base probability of applying each augmentation.
    """

    def __init__(
        self,
        jpeg_q_range: tuple[int, int] = (50, 100),
        blur_sigma_range: tuple[float, float] = (0.1, 2.0),
        downscale_size: int = 112,
        prob: float = 0.5,
    ) -> None:
        self.jpeg_q_range = jpeg_q_range
        self.blur_sigma_range = blur_sigma_range
        self.downscale_size = downscale_size
        self.prob = prob

    def __call__(self, image_pil: Image.Image) -> Image.Image:
        """Apply random degradations to a PIL image.

        Args:
            image_pil: Input PIL Image.

        Returns:
            Degraded PIL Image.
        """
        # Random JPEG compression
        if random.random() < self.prob:
            q = random.randint(*self.jpeg_q_range)
            buffer = io.BytesIO()
            image_pil.save(buffer, format="JPEG", quality=q)
            buffer.seek(0)
            image_pil = Image.open(buffer).copy()

        # Random Gaussian blur
        if random.random() < self.prob * 0.6:
            sigma = random.uniform(*self.blur_sigma_range)
            image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random downscale + upscale (social media simulation)
        if random.random() < self.prob * 0.6:
            w, h = image_pil.size
            small = image_pil.resize(
                (self.downscale_size, self.downscale_size), Image.LANCZOS
            )
            image_pil = small.resize((w, h), Image.LANCZOS)

        return image_pil
