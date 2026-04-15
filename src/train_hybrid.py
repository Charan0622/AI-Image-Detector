"""
Fast hybrid detector training using cached CLIP features + live DCT.

The CLIP branch is frozen, so we reuse pre-extracted 512-dim features.
Only the DCT frequency CNN and fusion MLP are trained.

This avoids running CLIP inference every epoch, reducing training
from hours to minutes on MPS.

Usage:
    python -m src.train_hybrid
"""

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import Config
from src.models.hybrid import FrequencyCNN
from src.seed import fix_seeds
from src.transforms import compute_dct_map, get_eval_transforms
from src.utils import check_disk_space


class HybridFeatureDataset(Dataset):
    """Dataset that pairs cached CLIP features with live DCT maps.

    Loads pre-extracted CLIP features from .npy files and computes
    DCT maps on-the-fly from original images.

    Args:
        clip_features: Pre-extracted CLIP features, shape (N, 512).
        labels: Labels, shape (N,).
        image_paths: Paths to original images for DCT computation.
    """

    def __init__(
        self,
        clip_features: np.ndarray,
        labels: np.ndarray,
        image_paths: list[Path],
    ) -> None:
        self.clip_features = torch.from_numpy(clip_features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        img = Image.open(self.image_paths[idx]).convert("RGB")
        dct_map = compute_dct_map(img)
        dct_tensor = torch.from_numpy(dct_map).unsqueeze(0).float()

        return {
            "clip_feat": self.clip_features[idx],
            "dct_map": dct_tensor,
            "label": self.labels[idx],
        }


class HybridFromFeatures(nn.Module):
    """Hybrid model that takes pre-extracted CLIP features + DCT maps.

    Args:
        clip_dim: CLIP feature dimension.
        freq_out_dim: Frequency branch output dimension.
        fusion_hidden: Fusion MLP hidden dimension.
        fusion_dropout: Dropout rate in fusion.
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
        """Forward pass.

        Args:
            clip_feat: Pre-extracted CLIP features, shape (B, 512).
            dct: DCT maps, shape (B, 1, 224, 224).

        Returns:
            Logits, shape (B, 2).
        """
        freq_feat = self.freq_encoder(dct)
        fused = torch.cat([clip_feat, freq_feat], dim=1)
        return self.classifier(fused)


def get_image_paths_for_features(
    data_dir: Path, split: str, max_samples: int = 0, seed: int = 42
) -> list[Path]:
    """Get image paths matching the feature extraction order.

    If features were extracted for the full dataset (no subsampling),
    returns all paths in dataset order. If max_samples > 0 and less
    than dataset size, returns the subsampled paths.

    Args:
        data_dir: Path to processed data.
        split: 'train' or 'val'.
        max_samples: Max samples (0 = use all).
        seed: Random seed.

    Returns:
        List of image paths in the same order as extracted features.
    """
    from src.dataset import AIDetectDataset

    ds = AIDetectDataset(data_dir, split=split)

    # If no subsampling needed, return all in order
    if max_samples <= 0 or max_samples >= len(ds):
        return [ds.samples[i][0] for i in range(len(ds))]

    # Subsampled mode
    random.seed(seed)
    real_indices = [i for i, (_, label) in enumerate(ds.samples) if label == 0]
    fake_indices = [i for i, (_, label) in enumerate(ds.samples) if label == 1]

    per_class = max_samples // 2
    random.shuffle(real_indices)
    random.shuffle(fake_indices)

    selected = real_indices[:per_class] + fake_indices[:per_class]
    random.shuffle(selected)

    return [ds.samples[i][0] for i in selected]


def get_test_image_paths(data_dir: Path, generator: str) -> list[Path]:
    """Get test image paths for a generator.

    Args:
        data_dir: Path to processed data.
        generator: Generator name.

    Returns:
        List of image paths.
    """
    from src.dataset import AIDetectTestDataset

    ds = AIDetectTestDataset(data_dir, generator=generator)
    return [ds.samples[i][0] for i in range(len(ds))]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        clip_feat = batch["clip_feat"].to(device)
        dct_map = batch["dct_map"].to(device)
        labels = batch["label"].to(device)

        logits = model(clip_feat, dct_map)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate on one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        clip_feat = batch["clip_feat"].to(device)
        dct_map = batch["dct_map"].to(device)
        labels = batch["label"].to(device)

        logits = model(clip_feat, dct_map)
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


def main() -> None:
    """Train hybrid detector."""
    config = Config()
    fix_seeds(config.seed)
    device = config.device

    feat_dir = config.project_root / "data" / "features"

    print("=== AIDE-Style Hybrid Detector Training ===")
    print(f"Device: {device}")

    # Load cached CLIP features
    print("\nLoading cached CLIP features...")
    train_feats = np.load(feat_dir / "train_features.npy")
    train_labels = np.load(feat_dir / "train_labels.npy")
    val_feats = np.load(feat_dir / "val_features.npy")
    val_labels = np.load(feat_dir / "val_labels.npy")

    # Get matching image paths for DCT computation
    print("Resolving image paths...")
    train_paths = get_image_paths_for_features(config.data_dir, "train")
    val_paths = get_image_paths_for_features(config.data_dir, "val")

    print(f"Train: {len(train_feats)} samples | Val: {len(val_feats)} samples")

    # Create datasets
    train_ds = HybridFeatureDataset(train_feats, train_labels, train_paths)
    val_ds = HybridFeatureDataset(val_feats, val_labels, val_paths)

    train_loader = DataLoader(
        train_ds, batch_size=config.hybrid_batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.hybrid_batch_size, shuffle=False, num_workers=2
    )

    # Model
    model = HybridFromFeatures(
        clip_dim=config.clip_embed_dim,
        freq_out_dim=config.freq_branch_out_dim,
        fusion_hidden=config.fusion_hidden_dim,
        fusion_dropout=config.fusion_dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.hybrid_lr, weight_decay=config.hybrid_weight_decay
    )
    epochs = config.hybrid_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    history: list[dict] = []

    ckpt_dir = config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = ckpt_dir / "backups"
    backup_dir.mkdir(exist_ok=True)

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_auc": round(val_auc, 4),
            "time_s": round(elapsed, 1),
        }
        history.append(epoch_metrics)

        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            ckpt_path = ckpt_dir / "hybrid_best.pth"
            if ckpt_path.exists():
                shutil.copy2(ckpt_path, backup_dir / "hybrid_best_backup.pth")
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": epoch_metrics},
                ckpt_path,
            )
            improved = " ★"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} | "
            f"{elapsed:.1f}s{improved}"
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

        if device.type == "mps":
            torch.mps.empty_cache()

    # Save history
    results = {
        "model": "hybrid",
        "best_val_auc": round(best_auc, 4),
        "total_epochs": len(history),
        "history": history,
    }
    metrics_path = config.results_dir / "metrics" / "hybrid_training.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest val AUC: {best_auc:.4f}")

    # Cross-gen evaluation
    print("\n=== Cross-Generator Evaluation ===")
    ckpt = torch.load(ckpt_dir / "hybrid_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cross_gen: dict = {}
    for gen in config.test_generators:
        test_feat_path = feat_dir / f"test_{gen}_features.npy"
        if not test_feat_path.exists():
            continue

        test_feats = np.load(test_feat_path)
        test_labels = np.load(feat_dir / f"test_{gen}_labels.npy")
        test_paths = get_test_image_paths(config.data_dir, gen)

        test_ds = HybridFeatureDataset(test_feats, test_labels, test_paths)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

        _, acc, auc = eval_epoch(model, test_loader, criterion, device)

        # Full metrics
        all_probs, all_preds, all_labs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                clip_feat = batch["clip_feat"].to(device)
                dct_map = batch["dct_map"].to(device)
                logits = model(clip_feat, dct_map)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labs.extend(batch["label"].numpy())

        from sklearn.metrics import f1_score, precision_score, recall_score

        labs_arr = np.array(all_labs)
        preds_arr = np.array(all_preds)

        cross_gen[gen] = {
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "precision": round(precision_score(labs_arr, preds_arr, zero_division=0), 4),
            "recall": round(recall_score(labs_arr, preds_arr, zero_division=0), 4),
            "f1": round(f1_score(labs_arr, preds_arr, zero_division=0), 4),
        }
        print(f"  {gen:12s}: acc={acc:.4f}  auc={auc:.4f}  f1={cross_gen[gen]['f1']:.4f}")

    # Save cross-gen
    cross_gen_path = config.results_dir / "metrics" / "hybrid_cross_gen.json"
    with open(cross_gen_path, "w") as f:
        json.dump(cross_gen, f, indent=2)

    # Generate table
    accs = [m["accuracy"] for m in cross_gen.values()]
    aucs = [m["auc"] for m in cross_gen.values()]

    table_lines = [
        "# Hybrid Detector — Cross-Generator Results\n",
        "| Generator | Accuracy | AUC | Precision | Recall | F1 |",
        "|-----------|----------|-----|-----------|--------|-----|",
    ]
    for gen, m in sorted(cross_gen.items()):
        table_lines.append(
            f"| {gen} | {m['accuracy']:.4f} | {m['auc']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
        )
    table_lines.append(
        f"| **Average** | **{np.mean(accs):.4f}** | **{np.mean(aucs):.4f}** | | | |"
    )

    table = "\n".join(table_lines)
    table_path = config.results_dir / "tables" / "hybrid_results.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write(table)

    print(f"\n{table}")


if __name__ == "__main__":
    main()
