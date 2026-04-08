"""
Generic training loop for all model variants.

Features:
    - Works with any model that takes (image) or (image, dct_map) as input
    - Early stopping on validation AUC
    - Automatic checkpoint saving (with backup of previous best)
    - Logs to console + results/metrics/
    - Disk check before saving checkpoints
    - tqdm progress bars

Usage:
    python -m src.train --model clip_probe --epochs 20
    python -m src.train --model hybrid --epochs 30
    python -m src.train --model freq_guided --epochs 40
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.dataset import AIDetectDataset, AIDetectDCTDataset
from src.models.model_zoo import get_model
from src.seed import fix_seeds
from src.transforms import get_clip_transforms, get_eval_transforms, get_train_transforms
from src.utils import check_disk_space, count_parameters


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    needs_dct: bool = False,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: The model to train.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        needs_dct: Whether model needs DCT input (hybrid/freq_guided).

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    # Keep CLIP encoder in eval mode (frozen BatchNorm/Dropout)
    if hasattr(model, "visual_encoder"):
        model.visual_encoder.eval()
    if hasattr(model, "clip_encoder"):
        model.clip_encoder.eval()
    if hasattr(model, "clip_visual"):
        model.clip_visual.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if needs_dct:
            dct_maps = batch["dct_map"].to(device)
            logits = model(images, dct_maps)
        else:
            logits = model(images)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    needs_dct: bool = False,
) -> tuple[float, float, float]:
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate.
        loader: Evaluation data loader.
        criterion: Loss function.
        device: Device.
        needs_dct: Whether model needs DCT input.

    Returns:
        Tuple of (average_loss, accuracy, auc).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if needs_dct:
            dct_maps = batch["dct_map"].to(device)
            logits = model(images, dct_maps)
        else:
            logits = model(images)

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5  # Only one class present

    return avg_loss, accuracy, auc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
    backup_dir: Path,
) -> None:
    """Save model checkpoint with backup of previous best.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        metrics: Dictionary of current metrics.
        path: Path to save checkpoint.
        backup_dir: Directory for backup of previous checkpoint.
    """
    free_gb, has_space = check_disk_space(1.0)
    if not has_space:
        print(f"WARNING: Only {free_gb:.1f}GB free. Skipping checkpoint save.")
        return

    # Backup previous checkpoint
    if path.exists():
        backup_path = backup_dir / f"{path.stem}_backup{path.suffix}"
        shutil.copy2(path, backup_path)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train AI image detector")
    parser.add_argument(
        "--model",
        type=str,
        default="clip_probe",
        choices=["clip_probe", "hybrid", "freq_guided"],
        help="Model to train",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)

    device = config.device
    print(f"Device: {device}")

    # Model-specific settings
    needs_dct = args.model in ("hybrid", "freq_guided")

    if args.model == "clip_probe":
        epochs = args.epochs or config.probe_epochs
        batch_size = args.batch_size or config.probe_batch_size
        lr = args.lr or config.probe_lr
        weight_decay = config.probe_weight_decay
    elif args.model == "hybrid":
        epochs = args.epochs or config.hybrid_epochs
        batch_size = args.batch_size or config.hybrid_batch_size
        lr = args.lr or config.hybrid_lr
        weight_decay = config.hybrid_weight_decay
    else:  # freq_guided
        epochs = args.epochs or config.final_epochs
        batch_size = args.batch_size or config.final_batch_size
        lr = args.lr or config.final_lr
        weight_decay = config.final_weight_decay

    print(f"Model: {args.model} | Epochs: {epochs} | BS: {batch_size} | LR: {lr}")

    # Create datasets
    if needs_dct:
        train_ds = AIDetectDCTDataset(
            config.data_dir, split="train", transform=get_train_transforms()
        )
        val_ds = AIDetectDCTDataset(
            config.data_dir, split="val", transform=get_eval_transforms()
        )
    else:
        train_ds = AIDetectDataset(
            config.data_dir, split="train", transform=get_train_transforms()
        )
        val_ds = AIDetectDataset(
            config.data_dir, split="val", transform=get_eval_transforms()
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Create model
    model = get_model(args.model, config).to(device)
    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Paths
    ckpt_path = config.checkpoint_dir / f"{args.model}_best.pth"
    backup_dir = config.checkpoint_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.results_dir / "metrics" / f"{args.model}_training.json"

    # Training loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    history: list[dict] = []

    print(f"\n{'='*60}")
    print(f"Training {args.model}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, needs_dct
        )
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, needs_dct
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_current = scheduler.get_last_lr()[0]

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_auc": round(val_auc, 4),
            "lr": round(lr_current, 6),
            "time_s": round(elapsed, 1),
        }
        history.append(epoch_metrics)

        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, epoch_metrics, ckpt_path, backup_dir)
            improved = " ★"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} | "
            f"LR={lr_current:.6f} | {elapsed:.0f}s{improved}"
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

        # Clear MPS cache if available
        if device.type == "mps":
            torch.mps.empty_cache()

    # Save training history
    results = {
        "model": args.model,
        "best_val_auc": round(best_auc, 4),
        "total_epochs": len(history),
        "config": {
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs_planned": epochs,
        },
        "history": history,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest val AUC: {best_auc:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
