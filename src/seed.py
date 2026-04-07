"""Fix ALL random seeds for reproducibility."""

import os
import random

import numpy as np
import torch


def fix_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducible results.

    Args:
        seed: The seed value to use everywhere.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't have manual_seed_all but torch.manual_seed covers it
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
