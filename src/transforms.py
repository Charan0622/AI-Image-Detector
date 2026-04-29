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
        - Social media JPEG recompression (single + double pass)
        - Messaging app blur
        - Screenshot/resize artifacts

    Args:
        jpeg_q_range: Min/max JPEG quality for random compression.
            Defaults to (35, 100) — WhatsApp / Discord typically recompress
            around Q40-60, so the lower bound matters for deployment.
        blur_sigma_range: Min/max sigma for Gaussian blur.
        downscale_size: Size to downscale to before upscaling back.
        prob: Base probability of applying each augmentation.
        double_jpeg_prob: Probability of a second JPEG re-encode after the
            first. Models common social-media pipelines that recompress
            already-recompressed uploads.
    """

    def __init__(
        self,
        jpeg_q_range: tuple[int, int] = (35, 100),
        blur_sigma_range: tuple[float, float] = (0.1, 2.0),
        downscale_size: int = 112,
        prob: float = 0.5,
        double_jpeg_prob: float = 0.3,
    ) -> None:
        self.jpeg_q_range = jpeg_q_range
        self.blur_sigma_range = blur_sigma_range
        self.downscale_size = downscale_size
        self.prob = prob
        self.double_jpeg_prob = double_jpeg_prob

    @staticmethod
    def _jpeg_recompress(image_pil: Image.Image, q: int) -> Image.Image:
        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return Image.open(buffer).copy()

    def __call__(self, image_pil: Image.Image) -> Image.Image:
        """Apply random degradations to a PIL image."""
        # Random JPEG compression (with optional double-JPEG path)
        if random.random() < self.prob:
            q = random.randint(*self.jpeg_q_range)
            image_pil = self._jpeg_recompress(image_pil, q)
            # Independent second pass — models social-media recompression
            if random.random() < self.double_jpeg_prob:
                q2 = random.randint(*self.jpeg_q_range)
                image_pil = self._jpeg_recompress(image_pil, q2)

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


class SmartphoneAesthetic:
    """Augment images to simulate smartphone / social-media aesthetics.

    The GenImage "real" class is dominated by ImageNet-style natural photos.
    Smartphone uploads have very different statistics: aggressive HDR
    tonemapping, Instagram-like colour grading, sensor read noise, mild
    chromatic aberration. Without exposure to these signals during training,
    smartphone real photos look out-of-distribution and get mistakenly
    flagged as AI-generated.

    This augmentation injects those signals so the trained model sees
    "smartphone-looking" pixels as a normal variant of *both* real and AI
    classes (we apply with the same probability regardless of label).

    Operations (each independently sampled):
        - PIL ColorJitter on brightness/contrast/saturation/hue
        - Random gamma in [0.7, 1.4]
        - Per-channel Gaussian read-noise σ ∈ [0, 4/255]
        - ±1 px chromatic aberration on R/B channels

    Args:
        prob: probability of applying the whole bundle to one sample.
        jitter_strengths: (brightness, contrast, saturation, hue) used by the
            internal ColorJitter.
        gamma_range: low/high gamma multipliers.
        noise_std: max σ for the per-channel additive Gaussian noise (in
            normalised [0,1] image units).
        chroma_max_shift: maximum integer pixel shift for R/B channels.
    """

    def __init__(
        self,
        prob: float = 0.6,
        jitter_strengths: tuple[float, float, float, float] = (0.3, 0.3, 0.4, 0.05),
        gamma_range: tuple[float, float] = (0.7, 1.4),
        noise_std: float = 4.0 / 255.0,
        chroma_max_shift: int = 1,
    ) -> None:
        self.prob = prob
        self.jitter = transforms.ColorJitter(*jitter_strengths)
        self.gamma_low, self.gamma_high = gamma_range
        self.noise_std = noise_std
        self.chroma_max_shift = chroma_max_shift

    @staticmethod
    def _apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
        return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))

    def _add_sensor_noise(self, img: Image.Image) -> Image.Image:
        if self.noise_std <= 0:
            return img
        sigma = random.uniform(0, self.noise_std)
        if sigma == 0:
            return img
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = arr + np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))

    def _chromatic_aberration(self, img: Image.Image) -> Image.Image:
        if self.chroma_max_shift <= 0:
            return img
        dx = random.randint(-self.chroma_max_shift, self.chroma_max_shift)
        dy = random.randint(-self.chroma_max_shift, self.chroma_max_shift)
        if dx == 0 and dy == 0:
            return img
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
        # Shift R channel by (dx, dy) and B channel by (-dx, -dy)
        arr[..., 0] = np.roll(arr[..., 0], (dy, dx), axis=(0, 1))
        arr[..., 2] = np.roll(arr[..., 2], (-dy, -dx), axis=(0, 1))
        return Image.fromarray(arr)

    def __call__(self, image_pil: Image.Image) -> Image.Image:
        if random.random() >= self.prob:
            return image_pil
        out = image_pil.convert("RGB")
        out = self.jitter(out)
        if random.random() < 0.7:
            out = self._apply_gamma(out, random.uniform(self.gamma_low, self.gamma_high))
        if random.random() < 0.7:
            out = self._add_sensor_noise(out)
        if random.random() < 0.5:
            out = self._chromatic_aberration(out)
        return out
