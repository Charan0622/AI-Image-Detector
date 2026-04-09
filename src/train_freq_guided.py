"""
Train freq-guided detector with optional robustness augmentation.

Supports ablation variants:
    - freq_guided: Full model (both improvements)
    - freq_guided_no_robust: Without robustness augmentation
    - hybrid_robust: Hybrid + robustness augmentation only

Usage:
    python -m src.train_freq_guided --variant full
    python -m src.train_freq_guided --variant no_robust
    python -m src.train_freq_guided --variant hybrid_robust
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import Config
from src.models.freq_guided import FreqGuidedFromFeatures
from src.models.hybrid import FrequencyCNN
from src.seed import fix_seeds
from src.transforms import RobustnessAugmentation, compute_dct_map


class AugmentedHybridDataset(Dataset):
    """Dataset with cached CLIP features + DCT with optional robustness aug.

    Args:
        clip_features: Pre-extracted CLIP features.
        labels: Labels.
        image_paths: Paths for DCT computation.
        robustness_aug: Optional robustness augmentation.
    """

    def __init__(
        self,
        clip_features: np.ndarray,
        labels: np.ndarray,
        image_paths: list[Path],
        robustness_aug: RobustnessAugmentation | None = None,
    ) -> None:
        self.clip_features = torch.from_numpy(clip_features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.image_paths = image_paths
        self.robustness_aug = robustness_aug

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Apply robustness augmentation before DCT
        if self.robustness_aug is not None:
            img = self.robustness_aug(img)

        dct_map = compute_dct_map(img)
        dct_tensor = torch.from_numpy(dct_map).unsqueeze(0).float()

        return {
            "clip_feat": self.clip_features[idx],
            "dct_map": dct_tensor,
            "label": self.labels[idx],
        }


class HybridRobustFromFeatures(nn.Module):
    """Hybrid model (simple concat) for ablation — same as hybrid but with robustness.

    Args:
        clip_dim: CLIP feature dimension.
        freq_out_dim: Frequency branch output dim.
        fusion_hidden: Fusion hidden dim.
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
        self.freq_encoder = FrequencyCNN(in_channels=1, out_dim=freq_out_dim)
        fused_dim = clip_dim + freq_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, 2),
        )

    def forward(self, clip_feat: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        freq_feat = self.freq_encoder(dct)
        fused = torch.cat([clip_feat, freq_feat], dim=1)
        return self.classifier(fused)


def get_image_paths(data_dir: Path, split: str, max_samples: int, seed: int = 42) -> list[Path]:
    """Get image paths matching feature extraction order."""
    from src.dataset import AIDetectDataset

    ds = AIDetectDataset(data_dir, split=split)
    random.seed(seed)
    real_idx = [i for i, (_, l) in enumerate(ds.samples) if l == 0]
    fake_idx = [i for i, (_, l) in enumerate(ds.samples) if l == 1]
    per_class = max_samples // 2
    random.shuffle(real_idx)
    random.shuffle(fake_idx)
    selected = real_idx[:per_class] + fake_idx[:per_class]
    random.shuffle(selected)
    return [ds.samples[i][0] for i in selected]


def get_test_paths(data_dir: Path, generator: str) -> list[Path]:
    """Get test image paths."""
    from src.dataset import AIDetectTestDataset

    ds = AIDetectTestDataset(data_dir, generator=generator)
    return [ds.samples[i][0] for i in range(len(ds))]


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        clip_feat = batch["clip_feat"].to(device)
        dct = batch["dct_map"].to(device)
        labels = batch["label"].to(device)
        logits = model(clip_feat, dct)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """Evaluate one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs, all_labels = [], []
    for batch in loader:
        clip_feat = batch["clip_feat"].to(device)
        dct = batch["dct_map"].to(device)
        labels = batch["label"].to(device)
        logits = model(clip_feat, dct)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return total_loss / total, correct / total, auc


def train_variant(variant: str, config: Config) -> dict:
    """Train a specific model variant and return cross-gen results.

    Args:
        variant: One of 'full', 'no_robust', 'hybrid_robust'.
        config: Configuration.

    Returns:
        Dictionary with training and cross-gen results.
    """
    device = config.device
    feat_dir = config.project_root / "data" / "features"

    use_robustness = variant in ("full", "hybrid_robust")
    use_freq_guided = variant in ("full", "no_robust")

    model_name = {
        "full": "freq_guided",
        "no_robust": "freq_guided_no_robust",
        "hybrid_robust": "hybrid_robust",
    }[variant]

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"  Freq-guided attention: {use_freq_guided}")
    print(f"  Robustness augmentation: {use_robustness}")
    print(f"{'='*60}")

    # Load features
    train_feats = np.load(feat_dir / "train_features.npy")
    train_labels = np.load(feat_dir / "train_labels.npy")
    val_feats = np.load(feat_dir / "val_features.npy")
    val_labels = np.load(feat_dir / "val_labels.npy")

    train_paths = get_image_paths(config.data_dir, "train", 20000)
    val_paths = get_image_paths(config.data_dir, "val", 8000)

    # Robustness augmentation (only for training)
    rob_aug = RobustnessAugmentation(
        jpeg_q_range=config.jpeg_q_range,
        blur_sigma_range=config.blur_sigma_range,
        downscale_size=config.downscale_size,
        prob=config.robustness_prob,
    ) if use_robustness else None

    train_ds = AugmentedHybridDataset(train_feats, train_labels, train_paths, rob_aug)
    val_ds = AugmentedHybridDataset(val_feats, val_labels, val_paths, None)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    # Model
    if use_freq_guided:
        model = FreqGuidedFromFeatures(
            clip_dim=config.clip_embed_dim,
            freq_out_dim=config.freq_branch_out_dim,
            fusion_hidden=config.fusion_hidden_dim,
            fusion_dropout=config.fusion_dropout,
        ).to(device)
    else:
        model = HybridRobustFromFeatures(
            clip_dim=config.clip_embed_dim,
            freq_out_dim=config.freq_branch_out_dim,
            fusion_hidden=config.fusion_hidden_dim,
            fusion_dropout=config.fusion_dropout,
        ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    epochs = min(config.final_epochs, 20)  # Cap at 20 for MPS speed
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.final_lr, weight_decay=config.final_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    patience_counter = 0
    history = []

    ckpt_path = config.checkpoint_dir / f"{model_name}_best.pth"
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        metrics = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4), "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4), "val_auc": round(val_auc, 4),
            "time_s": round(elapsed, 1),
        }
        history.append(metrics)

        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, ckpt_path)
            improved = " ★"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:2d}/{epochs} | Train: loss={train_loss:.4f} acc={train_acc:.4f} | Val: loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} | {elapsed:.1f}s{improved}")

        if patience_counter >= 5:
            print(f"Early stopping at epoch {epoch}.")
            break

        if device.type == "mps":
            torch.mps.empty_cache()

    # Save training metrics
    train_results = {"model": model_name, "best_val_auc": round(best_auc, 4), "total_epochs": len(history), "history": history}
    metrics_path = config.results_dir / "metrics" / f"{model_name}_training.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(train_results, f, indent=2)

    # Cross-gen evaluation
    print(f"\nBest val AUC: {best_auc:.4f}")
    print(f"\n--- Cross-Generator Evaluation ({model_name}) ---")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cross_gen = {}
    for gen in config.test_generators:
        test_feat_path = feat_dir / f"test_{gen}_features.npy"
        if not test_feat_path.exists():
            continue

        test_feats = np.load(test_feat_path)
        test_labels_arr = np.load(feat_dir / f"test_{gen}_labels.npy")
        test_paths = get_test_paths(config.data_dir, gen)

        test_ds = AugmentedHybridDataset(test_feats, test_labels_arr, test_paths, None)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

        _, acc, auc = eval_epoch(model, test_loader, criterion, device)

        all_probs, all_preds, all_labs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch["clip_feat"].to(device), batch["dct_map"].to(device))
                all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labs.extend(batch["label"].numpy())

        labs = np.array(all_labs)
        preds = np.array(all_preds)

        cross_gen[gen] = {
            "accuracy": round(acc, 4), "auc": round(auc, 4),
            "precision": round(precision_score(labs, preds, zero_division=0), 4),
            "recall": round(recall_score(labs, preds, zero_division=0), 4),
            "f1": round(f1_score(labs, preds, zero_division=0), 4),
        }
        print(f"  {gen:12s}: acc={acc:.4f}  auc={auc:.4f}  f1={cross_gen[gen]['f1']:.4f}")

    # Save cross-gen
    cg_path = config.results_dir / "metrics" / f"{model_name}_cross_gen.json"
    with open(cg_path, "w") as f:
        json.dump(cross_gen, f, indent=2)

    return {"model": model_name, "best_val_auc": best_auc, "cross_gen": cross_gen}


def main() -> None:
    """Train freq-guided variants and run ablation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full", choices=["full", "no_robust", "hybrid_robust", "all"])
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)

    if args.variant == "all":
        # Run all ablation variants
        results = {}
        for v in ["full", "no_robust", "hybrid_robust"]:
            fix_seeds(config.seed)
            r = train_variant(v, config)
            results[r["model"]] = r

        # Generate ablation table
        generate_ablation_table(results, config)
    else:
        train_variant(args.variant, config)


def generate_ablation_table(results: dict, config: Config) -> None:
    """Generate ablation study table combining all model results."""
    # Load probe and hybrid results too
    probe_cg = json.load(open(config.results_dir / "metrics" / "clip_probe_cross_gen.json"))
    hybrid_cg = json.load(open(config.results_dir / "metrics" / "hybrid_cross_gen.json"))

    all_models = {
        "CLIP Linear Probe": {"cross_gen": probe_cg},
        "AIDE-style Hybrid": {"cross_gen": hybrid_cg},
    }
    all_models.update({r["model"]: r for r in results.values()})

    lines = [
        "# Ablation Study Results\n",
        "| # | Model Variant | Cross-Gen Avg AUC | Cross-Gen Avg Acc | vs Probe |",
        "|---|---------------|-------------------|-------------------|----------|",
    ]

    probe_avg_auc = np.mean([m["auc"] for m in probe_cg.values()])

    for i, (name, data) in enumerate(all_models.items(), 1):
        cg = data["cross_gen"]
        avg_auc = np.mean([m["auc"] for m in cg.values()])
        avg_acc = np.mean([m["accuracy"] for m in cg.values()])
        delta = avg_auc - probe_avg_auc
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        lines.append(f"| {i} | {name} | {avg_auc:.4f} | {avg_acc:.4f} | {delta_str} |")

    table = "\n".join(lines)
    table_path = config.results_dir / "tables" / "ablation_table.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write(table)

    print(f"\n{table}")
    print(f"\nSaved to {table_path}")


if __name__ == "__main__":
    main()
