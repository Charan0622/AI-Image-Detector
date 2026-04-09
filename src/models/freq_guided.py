"""
Frequency-Guided Detector — Final Model.

KEY INNOVATION: Instead of simple concatenation, frequency features
guide attention over spatial frequency bands, then fuse with CLIP.

Architecture (optimized for cached CLIP features):
    1. CLIP ViT-B/16 CLS token (512-dim, pre-extracted)
    2. DCT frequency map -> Deep FrequencyCNN -> multi-scale features
    3. Frequency-Guided Spatial Attention: learns which frequency
       bands are most discriminative per image
    4. Gated fusion of CLIP + attended frequency features
    5. Deeper classifier with residual connection

This enables:
    - Better cross-generator generalization (frequency artifacts are
      generator-agnostic)
    - Attention weights show which frequency regions matter most

Trainable parameters: ~1.5M (freq CNN + attention + gated fusion + classifier)
"""

import torch
import torch.nn as nn


class MultiScaleFreqCNN(nn.Module):
    """Multi-scale frequency feature extractor.

    Extracts features at multiple spatial scales from DCT maps,
    capturing both low-frequency structure and high-frequency artifacts.

    Args:
        in_channels: Input channels (1 for DCT).
        out_dim: Output feature dimension.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 256) -> None:
        super().__init__()
        # Scale 1: Full resolution features
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 56
        )

        # Scale 2: Mid-frequency features
        self.scale2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),  # 28
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Scale 3: High-frequency features
        self.scale3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2),  # 14
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Spatial attention over frequency bands
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with spatial attention.

        Args:
            x: DCT maps, shape (B, 1, 224, 224).

        Returns:
            Tuple of (features (B, out_dim), attention_map (B, 1, 14, 14)).
        """
        s1 = self.scale1(x)   # (B, 64, 56, 56)
        s2 = self.scale2(s1)  # (B, 128, 28, 28)
        s3 = self.scale3(s2)  # (B, 256, 14, 14)

        # Spatial attention
        attn = self.spatial_attn(s3)  # (B, 1, 14, 14)
        attended = s3 * attn  # (B, 256, 14, 14)

        pooled = self.pool(attended).flatten(1)  # (B, 256)
        features = self.fc(pooled)  # (B, out_dim)

        return features, attn


class GatedFusion(nn.Module):
    """Gated fusion of CLIP and frequency features.

    Learns to dynamically weight the contribution of each branch
    based on the input, rather than fixed concatenation.

    Args:
        clip_dim: CLIP feature dimension.
        freq_dim: Frequency feature dimension.
        hidden_dim: Hidden dimension for gate computation.
    """

    def __init__(
        self, clip_dim: int = 512, freq_dim: int = 256, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        total_dim = clip_dim + freq_dim

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )

        # Project both to same dimension
        self.clip_proj = nn.Linear(clip_dim, hidden_dim)
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)

    def forward(
        self, clip_feat: torch.Tensor, freq_feat: torch.Tensor
    ) -> torch.Tensor:
        """Gated fusion.

        Args:
            clip_feat: CLIP features, shape (B, clip_dim).
            freq_feat: Frequency features, shape (B, freq_dim).

        Returns:
            Fused features, shape (B, hidden_dim).
        """
        combined = torch.cat([clip_feat, freq_feat], dim=1)
        gates = self.gate(combined)  # (B, 2)

        clip_proj = self.clip_proj(clip_feat)   # (B, hidden)
        freq_proj = self.freq_proj(freq_feat)   # (B, hidden)

        # Weighted sum
        fused = gates[:, 0:1] * clip_proj + gates[:, 1:2] * freq_proj
        return fused


class FreqGuidedDetector(nn.Module):
    """Frequency-guided detector with gated fusion.

    For use with full CLIP model (not cached features).

    Args:
        config: Configuration object.
    """

    def __init__(self, config: "Config") -> None:
        super().__init__()
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(
            config.clip_model_name, pretrained=config.clip_pretrained
        )
        self.clip_visual = clip_model.visual
        self.clip_visual.eval()
        for p in self.clip_visual.parameters():
            p.requires_grad = False

        self.freq_encoder = MultiScaleFreqCNN(
            in_channels=1, out_dim=config.freq_branch_out_dim
        )
        self.fusion = GatedFusion(
            clip_dim=config.clip_embed_dim,
            freq_dim=config.freq_branch_out_dim,
            hidden_dim=config.fusion_hidden_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, 2),
        )

    def forward(self, rgb: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            rgb: RGB images, shape (B, 3, 224, 224).
            dct: DCT maps, shape (B, 1, 224, 224).

        Returns:
            Logits, shape (B, 2).
        """
        with torch.no_grad():
            clip_feat = self.clip_visual(rgb)
        freq_feat, _ = self.freq_encoder(dct)
        fused = self.fusion(clip_feat, freq_feat)
        return self.classifier(fused)

    def get_attention_map(self, dct: torch.Tensor) -> torch.Tensor:
        """Get frequency spatial attention map for visualization.

        Args:
            dct: DCT maps, shape (B, 1, 224, 224).

        Returns:
            Attention map, shape (B, 14, 14).
        """
        _, attn = self.freq_encoder(dct)
        return attn.squeeze(1)  # (B, 14, 14)


class FreqGuidedFromFeatures(nn.Module):
    """Freq-guided model using cached CLIP features (for fast training).

    Args:
        clip_dim: CLIP feature dimension.
        freq_out_dim: Frequency branch output dimension.
        fusion_hidden: Fusion hidden dimension.
        fusion_dropout: Dropout rate.
    """

    def __init__(
        self,
        clip_dim: int = 512,
        freq_out_dim: int = 256,
        fusion_hidden: int = 256,
        fusion_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.freq_encoder = MultiScaleFreqCNN(in_channels=1, out_dim=freq_out_dim)
        self.fusion = GatedFusion(
            clip_dim=clip_dim, freq_dim=freq_out_dim, hidden_dim=fusion_hidden
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, 2),
        )

    def forward(self, clip_feat: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            clip_feat: Pre-extracted CLIP features, shape (B, 512).
            dct: DCT maps, shape (B, 1, 224, 224).

        Returns:
            Logits, shape (B, 2).
        """
        freq_feat, _ = self.freq_encoder(dct)
        fused = self.fusion(clip_feat, freq_feat)
        return self.classifier(fused)

    def get_attention_map(self, dct: torch.Tensor) -> torch.Tensor:
        """Get frequency spatial attention map.

        Args:
            dct: DCT maps, shape (B, 1, 224, 224).

        Returns:
            Attention map, shape (B, 14, 14).
        """
        _, attn = self.freq_encoder(dct)
        return attn.squeeze(1)
