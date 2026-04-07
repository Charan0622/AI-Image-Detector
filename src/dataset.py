"""
PyTorch Dataset classes for AI-generated image detection.

Classes:
    - AIDetectDataset: Standard RGB dataset with labels
    - AIDetectDCTDataset: Returns both RGB and DCT frequency map

All datasets return:
    - image: Tensor (3, 224, 224)
    - label: int (0=real, 1=fake)
    - metadata: dict with generator name, filename, etc.

For DCT variant:
    - dct_map: Tensor (1, 224, 224) — DCT frequency representation
"""

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.transforms import compute_dct_map, get_clip_transforms, get_eval_transforms


class AIDetectDataset(Dataset):
    """Standard RGB dataset for AI image detection.

    Loads images from directory structure:
        split/real/*.jpg  (label=0)
        split/fake/*.jpg  (label=1)

    Args:
        data_dir: Path to the processed data directory.
        split: One of 'train', 'val'.
        transform: Optional torchvision transform to apply.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or get_clip_transforms()

        # Collect all image paths and labels
        self.samples: list[tuple[Path, int]] = []

        split_dir = self.data_dir / split
        real_dir = split_dir / "real"
        fake_dir = split_dir / "fake"

        if real_dir.exists():
            for img_path in sorted(real_dir.glob("*.jpg")):
                self.samples.append((img_path, 0))  # 0 = real

        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob("*.jpg")):
                self.samples.append((img_path, 1))  # 1 = fake

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # Extract generator name from filename (e.g., "adm_000001.jpg")
        generator = img_path.stem.rsplit("_", 1)[0]

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": label,
            "metadata": {
                "generator": generator,
                "filename": img_path.name,
                "split": self.split,
            },
        }


class AIDetectTestDataset(Dataset):
    """Test dataset organized by generator for cross-gen evaluation.

    Loads images from directory structure:
        test/generator_name/real/*.jpg  (label=0)
        test/generator_name/fake/*.jpg  (label=1)

    Args:
        data_dir: Path to the processed data directory.
        generator: Generator name (e.g., 'adm', 'midjourney').
        transform: Optional torchvision transform to apply.
    """

    def __init__(
        self,
        data_dir: Path,
        generator: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.generator = generator
        self.transform = transform or get_eval_transforms()

        self.samples: list[tuple[Path, int]] = []

        gen_dir = self.data_dir / "test" / generator

        real_dir = gen_dir / "real"
        fake_dir = gen_dir / "fake"

        if real_dir.exists():
            for img_path in sorted(real_dir.glob("*.jpg")):
                self.samples.append((img_path, 0))

        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob("*.jpg")):
                self.samples.append((img_path, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": label,
            "metadata": {
                "generator": self.generator,
                "filename": img_path.name,
            },
        }


class AIDetectDCTDataset(Dataset):
    """Dataset returning both RGB and DCT frequency map.

    Extends the base dataset by also computing a DCT spectral
    map of each image. The DCT map captures frequency-domain
    artifacts that distinguish real from AI-generated images.

    Args:
        data_dir: Path to the processed data directory.
        split: One of 'train', 'val', or a generator name for test.
        transform: Optional torchvision transform for RGB images.
        is_test: If True, uses test directory structure (per-generator).
        generator: Generator name (required if is_test=True).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        is_test: bool = False,
        generator: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or get_clip_transforms()
        self.is_test = is_test
        self.generator = generator

        self.samples: list[tuple[Path, int]] = []

        if is_test and generator:
            gen_dir = self.data_dir / "test" / generator
            for label_name, label_int in [("real", 0), ("fake", 1)]:
                label_dir = gen_dir / label_name
                if label_dir.exists():
                    for img_path in sorted(label_dir.glob("*.jpg")):
                        self.samples.append((img_path, label_int))
        else:
            split_dir = self.data_dir / split
            for label_name, label_int in [("real", 0), ("fake", 1)]:
                label_dir = split_dir / label_name
                if label_dir.exists():
                    for img_path in sorted(label_dir.glob("*.jpg")):
                        self.samples.append((img_path, label_int))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, label = self.samples[idx]
        img_pil = Image.open(img_path).convert("RGB")

        # Compute DCT map from raw PIL image (before transforms)
        dct_map = compute_dct_map(img_pil)
        dct_tensor = torch.from_numpy(dct_map).unsqueeze(0).float()  # (1, H, W)

        # Extract generator name
        if self.is_test:
            generator = self.generator or "unknown"
        else:
            generator = img_path.stem.rsplit("_", 1)[0]

        # Apply RGB transforms
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.from_numpy(
                np.array(img_pil).transpose(2, 0, 1)
            ).float() / 255.0

        return {
            "image": img_tensor,
            "dct_map": dct_tensor,
            "label": label,
            "metadata": {
                "generator": generator,
                "filename": img_path.name,
                "split": self.split,
            },
        }
