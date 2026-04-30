"""
Warm fine-tune `hybrid_robust` and `freq_guided` heads on the expanded
training set (original GenImage + smartphone real photos + modern AI fakes
appended by ``scripts/expand_training_data.py``) with the new
``SmartphoneAesthetic`` and double-JPEG augmentations.

The CLIP backbone stays frozen — we only retrain the small heads (~1.5-2M
params). Each variant initialises from the v1 checkpoint and trains 8
epochs at LR=1e-4 (1/5 of the original).

Usage:
    python -m src.train_hybrid_robust_v2 --variant hybrid_robust
    python -m src.train_hybrid_robust_v2 --variant freq_guided
    python -m src.train_hybrid_robust_v2 --variant all   # train both
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.models.freq_guided import FreqGuidedFromFeatures
from src.seed import fix_seeds
from src.train_freq_guided import (
    AugmentedHybridDataset,
    HybridRobustFromFeatures,
    eval_epoch,
    get_image_paths,
    train_epoch,
)
from src.transforms import RobustnessAugmentation, SmartphoneAesthetic


VARIANTS = {
    "hybrid_robust": {
        "cls": HybridRobustFromFeatures,
        "load_ckpt": "hybrid_robust_best.pth",
        "save_ckpt": "hybrid_robust_v2_best.pth",
        "metrics_name": "hybrid_robust_v2",
    },
    "freq_guided": {
        "cls": FreqGuidedFromFeatures,
        "load_ckpt": "freq_guided_best.pth",
        "save_ckpt": "freq_guided_v2_best.pth",
        "metrics_name": "freq_guided_v2",
    },
}


class CombinedAug:
    """Apply SmartphoneAesthetic *before* RobustnessAugmentation.

    SmartphoneAesthetic injects camera-pipeline signals (Instagram colour
    grading, sensor noise, chromatic aberration). RobustnessAugmentation
    then applies a social-media JPEG / blur / resize hit on top.
    """

    def __init__(self, smartphone: SmartphoneAesthetic, robustness: RobustnessAugmentation) -> None:
        self.smartphone = smartphone
        self.robustness = robustness

    def __call__(self, img):
        img = self.smartphone(img)
        img = self.robustness(img)
        return img


def load_paths_with_extras(config: Config, split: str) -> list[Path]:
    """Return the original split paths followed by the appended `_extra` paths.

    `scripts/expand_training_data.py` appends features for new images at the
    end of `data/features/{split}_features.npy` and writes the corresponding
    paths in order to `data/features/{split}_extra_paths.txt`. The dataset
    aligns features↔paths positionally, so ordering matters.
    """
    base = get_image_paths(config.data_dir, split)
    feat_dir = config.project_root / "data" / "features"
    extras_file = feat_dir / f"{split}_extra_paths.txt"
    if extras_file.exists():
        with open(extras_file) as f:
            extras = [Path(line.strip()) for line in f if line.strip()]
        print(f"  appended {len(extras)} extra paths to {split} (total {len(base) + len(extras)})")
        return base + extras
    return base


def train_one_variant(variant: str, config: Config, device: torch.device, *, epochs: int, lr: float) -> None:
    info = VARIANTS[variant]
    model_name = info["metrics_name"]

    print(f"\n{'=' * 60}")
    print(f"Warm fine-tune: {model_name}")
    print(f"  Init from   : {info['load_ckpt']}")
    print(f"  Save to     : {info['save_ckpt']}")
    print(f"  Epochs      : {epochs}")
    print(f"  LR          : {lr}")
    print(f"{'=' * 60}")

    feat_dir = config.project_root / "data" / "features"
    train_feats = np.load(feat_dir / "train_features.npy")
    train_labels = np.load(feat_dir / "train_labels.npy")
    val_feats = np.load(feat_dir / "val_features.npy")
    val_labels = np.load(feat_dir / "val_labels.npy")

    train_paths = load_paths_with_extras(config, "train")
    val_paths = load_paths_with_extras(config, "val")

    # Sanity: features and paths must align by length
    if len(train_paths) != len(train_feats):
        raise RuntimeError(
            f"Train alignment broken: {len(train_paths)} paths vs {len(train_feats)} features. "
            f"Re-run scripts/expand_training_data.py."
        )
    if len(val_paths) != len(val_feats):
        raise RuntimeError(
            f"Val alignment broken: {len(val_paths)} paths vs {len(val_feats)} features."
        )

    # Combined augmentation (smartphone + robustness)
    smartphone = SmartphoneAesthetic(prob=config.smartphone_aesthetic_prob)
    robustness = RobustnessAugmentation(
        jpeg_q_range=config.jpeg_q_range,
        blur_sigma_range=config.blur_sigma_range,
        downscale_size=config.downscale_size,
        prob=config.robustness_prob,
        double_jpeg_prob=config.double_jpeg_prob,
    )
    aug = CombinedAug(smartphone, robustness)

    train_ds = AugmentedHybridDataset(train_feats, train_labels, train_paths, aug)
    val_ds = AugmentedHybridDataset(val_feats, val_labels, val_paths, None)

    train_loader = DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=config.train_batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=False)

    # Build model + load v1 weights
    model = info["cls"](
        clip_dim=config.clip_embed_dim,
        freq_out_dim=config.freq_branch_out_dim,
        fusion_hidden=config.fusion_hidden_dim,
        fusion_dropout=config.fusion_dropout,
    ).to(device)

    init_path = config.checkpoint_dir / info["load_ckpt"]
    if init_path.exists():
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"  loaded init weights from {init_path}")
    else:
        print(f"  WARNING: no init checkpoint at {init_path} — training from scratch.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.final_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = config.checkpoint_dir / info["save_ckpt"]
    best_auc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_auc": round(val_auc, 4),
            "time_s": round(elapsed, 1),
        }
        history.append(metrics)

        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, save_path)
            improved = " ★"

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} | "
            f"{elapsed:.1f}s{improved}",
            flush=True,
        )

        if device.type == "mps":
            torch.mps.empty_cache()

    metrics_out = config.results_dir / "metrics" / f"{model_name}_training.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(
            {
                "model": model_name,
                "best_val_auc": round(best_auc, 4),
                "total_epochs": len(history),
                "history": history,
                "config": {"epochs": epochs, "lr": lr, "init_from": info["load_ckpt"]},
            },
            f,
            indent=2,
        )
    print(f"\nBest val AUC: {best_auc:.4f}  →  {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="hybrid_robust",
                        choices=["hybrid_robust", "freq_guided", "all"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = Config()
    # Override DataLoader knobs from CLI so a bigger batch / more workers
    # don't require editing the script.
    config.train_batch_size = args.batch_size
    config.num_workers = args.num_workers
    fix_seeds(config.seed)
    device = config.device
    print(f"Device: {device}")

    variants = ["hybrid_robust", "freq_guided"] if args.variant == "all" else [args.variant]
    for v in variants:
        train_one_variant(v, config, device, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
