"""
CLIP Linear Probe — Baseline 1 (SOTA Candidate #1).

Architecture:
    Frozen CLIP ViT-B/16 image encoder -> 512-dim CLS token -> Linear(512, 2)

This is the simplest possible detector. It tests whether CLIP's
pre-trained features already contain enough signal to detect AI images.

Published baselines (Cozzolino et al., CVPRW 2024) show this approach
achieves ~90% in-distribution and ~75-85% cross-generator AUC.
"""

import torch
import torch.nn as nn
import open_clip


class CLIPLinearProbe(nn.Module):
    """Frozen CLIP encoder with a single linear classification head.

    Args:
        clip_model_name: OpenCLIP model architecture name.
        clip_pretrained: Pretrained weights identifier.
        num_classes: Number of output classes (2 for real/fake).
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.visual_encoder = clip_model.visual
        self.visual_encoder.eval()
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        # Get embedding dimension dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            embed_dim = self.visual_encoder(dummy).shape[-1]

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen CLIP features -> linear classifier.

        Args:
            x: Input images, shape (B, 3, 224, 224).

        Returns:
            Logits, shape (B, num_classes).
        """
        with torch.no_grad():
            features = self.visual_encoder(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for analysis).

        Args:
            x: Input images, shape (B, 3, 224, 224).

        Returns:
            Feature vectors, shape (B, embed_dim).
        """
        with torch.no_grad():
            return self.visual_encoder(x)
