"""
Generate sample Grad-CAM visualizations for all generators.

Picks 3 real + 3 fake images from each generator, generates
heatmap overlays, and saves to results/gradcam_samples/.
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import Config
from src.explain import generate_explanations
from src.gradcam_utils import (
    create_heatmap_overlay,
    get_freq_attention_map,
    gradcam_freq_branch,
)
from src.models.freq_guided import FreqGuidedFromFeatures
from src.seed import fix_seeds
from src.transforms import compute_dct_map

random.seed(42)


def main() -> None:
    """Generate Grad-CAM samples for all generators."""
    config = Config()
    fix_seeds(config.seed)
    device = config.device

    feat_dir = config.project_root / "data" / "features"
    output_dir = config.results_dir / "gradcam_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load freq-guided model
    print("Loading freq-guided model...")
    model = FreqGuidedFromFeatures(
        clip_dim=config.clip_embed_dim,
        freq_out_dim=config.freq_branch_out_dim,
        fusion_hidden=config.fusion_hidden_dim,
        fusion_dropout=config.fusion_dropout,
    ).to(device)

    ckpt_path = config.checkpoint_dir / "freq_guided_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: No freq_guided checkpoint found, using random weights")

    model.eval()

    samples_per_class = 3

    for gen in config.test_generators:
        print(f"\n=== Generating Grad-CAM for {gen} ===")

        test_feats = np.load(feat_dir / f"test_{gen}_features.npy").astype(np.float32)
        test_labels = np.load(feat_dir / f"test_{gen}_labels.npy")

        gen_dir = config.data_dir / "test" / gen
        real_paths = sorted((gen_dir / "real").glob("*.jpg"))
        fake_paths = sorted((gen_dir / "fake").glob("*.jpg"))

        real_feats = test_feats[test_labels == 0]
        fake_feats = test_feats[test_labels == 1]

        # Pick random samples
        real_indices = random.sample(range(len(real_paths)), min(samples_per_class, len(real_paths)))
        fake_indices = random.sample(range(len(fake_paths)), min(samples_per_class, len(fake_paths)))

        # Create figure: 2 rows (real, fake) x (samples_per_class * 2) cols (original + overlay)
        fig, axes = plt.subplots(2, samples_per_class * 2, figsize=(4 * samples_per_class * 2, 8))
        fig.suptitle(f"Grad-CAM Visualization — {gen}", fontsize=14, fontweight="bold")

        for row, (indices, paths, feats, label_name) in enumerate([
            (real_indices, real_paths, real_feats, "Real"),
            (fake_indices, fake_paths, fake_feats, "Fake"),
        ]):
            for j, idx in enumerate(indices):
                img_path = paths[idx]
                clip_feat = torch.from_numpy(feats[idx])
                img_pil = Image.open(img_path).convert("RGB")

                # Compute DCT
                dct = compute_dct_map(img_pil)
                dct_tensor = torch.from_numpy(dct).unsqueeze(0).unsqueeze(0).float()

                # Get heatmap
                heatmap = gradcam_freq_branch(
                    model,
                    clip_feat.unsqueeze(0),
                    dct_tensor,
                    device,
                    target_class=1,
                )

                # Create overlay
                overlay = create_heatmap_overlay(img_pil, heatmap, alpha=0.4)

                # Get prediction
                with torch.no_grad():
                    logits = model(
                        clip_feat.unsqueeze(0).to(device),
                        dct_tensor.to(device),
                    )
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_label = "Fake" if probs[1] > 0.5 else "Real"
                    confidence = probs[1].item() if pred_label == "Fake" else probs[0].item()

                # Plot original
                col = j * 2
                axes[row, col].imshow(img_pil)
                axes[row, col].set_title(f"{label_name} ({pred_label} {confidence:.0%})", fontsize=9)
                axes[row, col].axis("off")

                # Plot overlay
                axes[row, col + 1].imshow(overlay)
                axes[row, col + 1].set_title("Grad-CAM", fontsize=9)
                axes[row, col + 1].axis("off")

        plt.tight_layout()
        save_path = output_dir / f"gradcam_{gen}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # Generate a combined summary image
    print("\n=== Generating Summary Panel ===")
    fig, axes = plt.subplots(len(config.test_generators), 4, figsize=(16, 4 * len(config.test_generators)))
    fig.suptitle("Grad-CAM Summary: Real vs Fake (All Generators)", fontsize=16, y=1.01)

    for i, gen in enumerate(config.test_generators):
        test_feats = np.load(feat_dir / f"test_{gen}_features.npy").astype(np.float32)
        test_labels = np.load(feat_dir / f"test_{gen}_labels.npy")

        gen_dir = config.data_dir / "test" / gen
        real_paths = sorted((gen_dir / "real").glob("*.jpg"))
        fake_paths = sorted((gen_dir / "fake").glob("*.jpg"))

        real_feats = test_feats[test_labels == 0]
        fake_feats = test_feats[test_labels == 1]

        # One real, one fake
        for j, (paths, feats, label_name) in enumerate([
            (real_paths, real_feats, "Real"),
            (fake_paths, fake_feats, "Fake"),
        ]):
            idx = 0
            img_pil = Image.open(paths[idx]).convert("RGB")
            clip_feat = torch.from_numpy(feats[idx])
            dct = compute_dct_map(img_pil)
            dct_tensor = torch.from_numpy(dct).unsqueeze(0).unsqueeze(0).float()

            heatmap = gradcam_freq_branch(model, clip_feat.unsqueeze(0), dct_tensor, device, 1)
            overlay = create_heatmap_overlay(img_pil, heatmap, alpha=0.4)

            col = j * 2
            axes[i, col].imshow(img_pil)
            axes[i, col].set_title(f"{gen} — {label_name}", fontsize=10)
            axes[i, col].axis("off")
            axes[i, col + 1].imshow(overlay)
            axes[i, col + 1].set_title("Grad-CAM", fontsize=10)
            axes[i, col + 1].axis("off")

    plt.tight_layout()
    summary_path = output_dir / "gradcam_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary: {summary_path}")

    print("\n✅ Grad-CAM visualization complete!")


if __name__ == "__main__":
    main()
