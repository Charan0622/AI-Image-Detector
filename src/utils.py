"""Miscellaneous helpers for logging, timing, and disk checks."""

import shutil
import time
from pathlib import Path
from typing import Any


def check_disk_space(min_gb: float = 1.0) -> tuple[float, bool]:
    """Check free disk space on the home partition.

    Args:
        min_gb: Minimum required free space in GB.

    Returns:
        Tuple of (free_gb, has_enough_space).
    """
    total, used, free = shutil.disk_usage(Path.home())
    free_gb = free / 1e9
    return free_gb, free_gb >= min_gb


class Timer:
    """Simple context manager for timing code blocks.

    Usage:
        with Timer("Training epoch"):
            train_one_epoch()
    """

    def __init__(self, name: str = "Block") -> None:
        self.name = name
        self.start: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.time() - self.start
        print(f"{self.name}: {self.elapsed:.1f}s")


def count_parameters(model: "torch.nn.Module") -> tuple[int, int]:
    """Count trainable and total parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (trainable_params, total_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
