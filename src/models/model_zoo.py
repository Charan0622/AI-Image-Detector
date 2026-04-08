"""
Model registry to load any model by name.

Usage:
    model = get_model('clip_probe', config)
    model = get_model('hybrid', config)
    model = get_model('freq_guided', config)
"""

import torch.nn as nn

from src.config import Config


def get_model(name: str, config: Config) -> nn.Module:
    """Load a model by name.

    Args:
        name: Model identifier ('clip_probe', 'hybrid', 'freq_guided').
        config: Configuration object.

    Returns:
        Initialized model.

    Raises:
        ValueError: If model name is unknown.
    """
    if name == "clip_probe":
        from src.models.clip_probe import CLIPLinearProbe

        return CLIPLinearProbe(
            clip_model_name=config.clip_model_name,
            clip_pretrained=config.clip_pretrained,
        )
    elif name == "hybrid":
        from src.models.hybrid import HybridDetector

        return HybridDetector(config)
    elif name == "freq_guided":
        from src.models.freq_guided import FreqGuidedDetector

        return FreqGuidedDetector(config)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: clip_probe, hybrid, freq_guided")
