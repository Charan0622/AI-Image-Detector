"""
AIDE-Style Hybrid Detector — Baseline 2 (SOTA Candidate #2).

Architecture:
    Branch 1: Frozen CLIP ViT-B/16 -> 512-dim CLS token
    Branch 2: Small CNN on DCT frequency map -> 256-dim
    Fusion: Concatenate -> MLP(768, 256, 2)

This follows the approach of AIDE (Yan et al., ICLR 2025) which showed
that combining semantic (CLIP) and frequency (DCT) features significantly
outperforms either branch alone.

Trainable parameters: ~2M (frequency CNN + fusion MLP only)
"""

import torch
import torch.nn as nn
import open_clip


class FrequencyCNN(nn.Module):
    """Lightweight CNN for processing DCT frequency maps.

    Architecture: 4 conv blocks -> global average pool -> FC
    Input: (B, 1, 224, 224) DCT map
    Output: (B, out_dim) frequency features

    Args:
        in_channels: Number of input channels (1 for grayscale DCT).
        out_dim: Output feature dimension.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # 112x112
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 56x56
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),  # 28x28
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: DCT maps, shape (B, 1, 224, 224).

        Returns:
            Frequency features, shape (B, out_dim).
        """
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


class HybridDetector(nn.Module):
    """Two-branch hybrid detector combining CLIP and frequency features.

    Args:
        config: Configuration object with model hyperparameters.
    """

    def __init__(self, config: "Config") -> None:
        super().__init__()
        # Branch 1: Frozen CLIP
        clip_model, _, _ = open_clip.create_model_and_transforms(
            config.clip_model_name, pretrained=config.clip_pretrained
        )
        self.clip_encoder = clip_model.visual
        self.clip_encoder.eval()
        for p in self.clip_encoder.parameters():
            p.requires_grad = False

        # Branch 2: Frequency CNN (trainable)
        self.freq_encoder = FrequencyCNN(
            in_channels=1, out_dim=config.freq_branch_out_dim
        )

        # Fusion head (trainable)
        fused_dim = config.clip_embed_dim + config.freq_branch_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, 2),
        )

    def forward(self, rgb: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
        """Forward pass combining both branches.

        Args:
            rgb: RGB images, shape (B, 3, 224, 224).
            dct: DCT maps, shape (B, 1, 224, 224).

        Returns:
            Logits, shape (B, 2).
        """
        with torch.no_grad():
            clip_feat = self.clip_encoder(rgb)  # (B, 512)
        freq_feat = self.freq_encoder(dct)  # (B, 256)
        fused = torch.cat([clip_feat, freq_feat], dim=1)  # (B, 768)
        return self.classifier(fused)

    def get_branch_features(
        self, rgb: torch.Tensor, dct: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return individual branch outputs for analysis.

        Args:
            rgb: RGB images.
            dct: DCT maps.

        Returns:
            Tuple of (clip_features, freq_features).
        """
        with torch.no_grad():
            clip_feat = self.clip_encoder(rgb)
        freq_feat = self.freq_encoder(dct)
        return clip_feat, freq_feat
