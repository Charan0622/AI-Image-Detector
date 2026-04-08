"""
Fast CLIP linear probe training on pre-extracted features.

Uses cached CLIP features from data/features/ to train a linear
classifier in seconds instead of hours. This is the standard
approach for linear probing.

Usage:
    python -m src.train_probe
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import Config
from src.seed import fix_seeds
from src.utils import check_disk_space


class LinearProbeHead(nn.Module):
    """Simple linear classifier for pre-extracted features.

    Args:
        input_dim: Dimension of input features.
        num_classes: Number of output classes.
    """

    def __init__(self, input_dim: int = 512, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature vectors, shape (B, input_dim).

        Returns:
            Logits, shape (B, num_classes).
        """
        return self.fc(x)


def load_features(feat_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load pre-extracted features and labels.

    Args:
        feat_dir: Directory containing feature .npy files.
        split: Split name ('train', 'val', or 'test_{gen}').

    Returns:
        Tuple of (features_tensor, labels_tensor).
    """
    features = np.load(feat_dir / f"{split}_features.npy").astype(np.float32)
    labels = np.load(feat_dir / f"{split}_labels.npy").astype(np.int64)
    return torch.from_numpy(features), torch.from_numpy(labels)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch on cached features.

    Args:
        model: Linear probe head.
        loader: Feature data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
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
    """Evaluate on cached features.

    Args:
        model: Linear probe head.
        loader: Feature data loader.
        criterion: Loss function.
        device: Device.

    Returns:
        Tuple of (average_loss, accuracy, auc).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
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
    """Train linear probe on pre-extracted CLIP features."""
    config = Config()
    fix_seeds(config.seed)
    device = config.device

    feat_dir = config.project_root / "data" / "features"

    print("=== CLIP Linear Probe (Fast Training on Cached Features) ===")
    print(f"Device: {device}")

    # Load features
    print("\nLoading features...")
    train_feats, train_labels = load_features(feat_dir, "train")
    val_feats, val_labels = load_features(feat_dir, "val")
    print(f"Train: {train_feats.shape} | Val: {val_feats.shape}")

    # Create data loaders
    train_ds = TensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # Model
    input_dim = train_feats.shape[1]
    model = LinearProbeHead(input_dim=input_dim).to(device)
    print(f"Linear head: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.probe_lr, weight_decay=config.probe_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.probe_epochs
    )

    # Training loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    history: list[dict] = []

    print(f"\nTraining for {config.probe_epochs} epochs...")
    print("-" * 80)

    for epoch in range(1, config.probe_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_auc = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_auc": round(val_auc, 4),
            "time_s": round(elapsed, 2),
        }
        history.append(epoch_metrics)

        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            # Save the linear head weights + full model compatible checkpoint
            ckpt_dir = config.checkpoint_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            backup_dir = ckpt_dir / "backups"
            backup_dir.mkdir(exist_ok=True)

            ckpt_path = ckpt_dir / "clip_probe_best.pth"
            if ckpt_path.exists():
                shutil.copy2(ckpt_path, backup_dir / "clip_probe_best_backup.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "metrics": epoch_metrics,
                },
                ckpt_path,
            )
            improved = " ★"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:2d}/{config.probe_epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} | "
            f"{elapsed:.2f}s{improved}"
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # Save training metrics
    results = {
        "model": "clip_probe",
        "best_val_auc": round(best_auc, 4),
        "total_epochs": len(history),
        "config": {
            "batch_size": 256,
            "lr": config.probe_lr,
            "weight_decay": config.probe_weight_decay,
        },
        "history": history,
    }
    metrics_path = config.results_dir / "metrics" / "clip_probe_training.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest val AUC: {best_auc:.4f}")
    print(f"Checkpoint: {config.checkpoint_dir / 'clip_probe_best.pth'}")
    print(f"Metrics: {metrics_path}")

    # Cross-generator evaluation on cached features
    print("\n=== Cross-Generator Evaluation ===")
    ckpt = torch.load(
        config.checkpoint_dir / "clip_probe_best.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cross_gen: dict = {}
    for gen in config.test_generators:
        feat_path = feat_dir / f"test_{gen}_features.npy"
        if not feat_path.exists():
            continue

        test_feats, test_labels = load_features(feat_dir, f"test_{gen}")
        test_ds = TensorDataset(test_feats, test_labels)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        _, acc, auc = eval_epoch(model, test_loader, criterion, device)

        # Get full metrics
        all_probs = []
        all_preds = []
        all_labels_list = []
        with torch.no_grad():
            for feats, labs in test_loader:
                feats = feats.to(device)
                logits = model(feats)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels_list.extend(labs.numpy())

        from sklearn.metrics import f1_score, precision_score, recall_score

        all_labels_arr = np.array(all_labels_list)
        all_preds_arr = np.array(all_preds)

        cross_gen[gen] = {
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "precision": round(precision_score(all_labels_arr, all_preds_arr, zero_division=0), 4),
            "recall": round(recall_score(all_labels_arr, all_preds_arr, zero_division=0), 4),
            "f1": round(f1_score(all_labels_arr, all_preds_arr, zero_division=0), 4),
        }
        print(
            f"  {gen:12s}: acc={acc:.4f}  auc={auc:.4f}  "
            f"f1={cross_gen[gen]['f1']:.4f}"
        )

    # Save cross-gen results
    cross_gen_path = config.results_dir / "metrics" / "clip_probe_cross_gen.json"
    with open(cross_gen_path, "w") as f:
        json.dump(cross_gen, f, indent=2)

    # Generate results table
    accs = [m["accuracy"] for m in cross_gen.values()]
    aucs = [m["auc"] for m in cross_gen.values()]

    table_lines = [
        "# CLIP Linear Probe — Cross-Generator Results\n",
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
    table_path = config.results_dir / "tables" / "clip_probe_results.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write(table)

    print(f"\n{table}")
    print(f"\nTable saved to {table_path}")


if __name__ == "__main__":
    main()
