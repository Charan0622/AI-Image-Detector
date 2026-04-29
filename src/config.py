"""
Central configuration file. ALL hyperparameters live here.
Nothing is hardcoded anywhere else in the codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Central configuration for all models and training."""

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(default=None)
    checkpoint_dir: Path = field(default=None)
    results_dir: Path = field(default=None)

    # Seeds (FIXED EVERYWHERE)
    seed: int = 42

    # Data
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = False  # True for CUDA, False for MPS

    # CLIP
    clip_model_name: str = "ViT-B-16"
    clip_pretrained: str = "laion2b_s34b_b88k"
    clip_embed_dim: int = 512

    # Training — Linear Probe
    probe_batch_size: int = 32
    probe_lr: float = 1e-3
    probe_weight_decay: float = 1e-4
    probe_epochs: int = 20
    probe_scheduler: str = "cosine"

    # Training — Hybrid
    hybrid_batch_size: int = 16
    hybrid_lr: float = 5e-4
    hybrid_weight_decay: float = 1e-4
    hybrid_epochs: int = 30

    # Training — Freq-Guided (final model)
    final_batch_size: int = 16
    final_lr: float = 3e-4
    final_weight_decay: float = 1e-4
    final_epochs: int = 40

    # Frequency branch
    freq_branch_out_dim: int = 256
    freq_branch_type: str = "resnet18"

    # Fusion
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.3

    # Robustness augmentation
    # Q range extended to 35 to cover WhatsApp/Discord recompression quality.
    jpeg_q_range: tuple = (35, 100)
    blur_sigma_range: tuple = (0.1, 2.0)
    downscale_size: int = 112
    robustness_prob: float = 0.5
    double_jpeg_prob: float = 0.3
    smartphone_aesthetic_prob: float = 0.6

    # Evaluation
    test_generators: list = field(
        default_factory=lambda: [
            "adm",
            "glide",
            "midjourney",
            "sd15",
            "vqdm",
            "wukong",
        ]
    )

    def __post_init__(self) -> None:
        """Set derived paths after init."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "processed"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "checkpoints"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results"

    @property
    def device(self) -> "torch.device":
        """Get the best available device."""
        import torch

        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
