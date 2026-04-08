"""
Pre-extract CLIP features to disk for fast linear probe training.

Uses a subsample of training data (20K images) to keep extraction
time under 30 minutes on MPS. For a linear probe on frozen CLIP
features, 20K samples is more than sufficient.

Output:
    data/features/{split}_features.npy  (N, 512) float16
    data/features/{split}_labels.npy    (N,) int64
"""

import random
import shutil
from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import Config
from src.dataset import AIDetectDataset, AIDetectTestDataset
from src.seed import fix_seeds
from src.transforms import get_eval_transforms


# Max training samples (10K real + 10K fake)
MAX_TRAIN_SAMPLES = 20000
MAX_VAL_SAMPLES = 8000


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract CLIP features from a data loader.

    Args:
        model: CLIP visual encoder.
        loader: Data loader.
        device: Compute device.

    Returns:
        Tuple of (features, labels) as numpy arrays.
    """
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            images = batch["image"].to(device)
            labels = batch["label"]

            features = model(images)
            all_features.append(features.cpu().float().numpy())
            all_labels.append(labels.numpy())

            # Periodically clear MPS cache
            if device.type == "mps" and len(all_features) % 50 == 0:
                torch.mps.empty_cache()

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def subsample_dataset(
    dataset: AIDetectDataset, max_samples: int, seed: int = 42
) -> Subset:
    """Subsample a dataset while maintaining class balance.

    Args:
        dataset: Full dataset.
        max_samples: Maximum total samples.
        seed: Random seed.

    Returns:
        Subset with balanced classes.
    """
    random.seed(seed)
    real_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
    fake_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]

    per_class = max_samples // 2
    random.shuffle(real_indices)
    random.shuffle(fake_indices)

    selected = real_indices[:per_class] + fake_indices[:per_class]
    random.shuffle(selected)
    return Subset(dataset, selected)


def main() -> None:
    """Extract and save CLIP features for all splits."""
    config = Config()
    fix_seeds(config.seed)
    device = config.device

    # Check disk space
    total, used, free = shutil.disk_usage(Path.home())
    print(f"Disk: {free / 1e9:.1f}GB free")
    print(f"Device: {device}")

    # Output directory
    feat_dir = config.project_root / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP visual encoder
    print("Loading CLIP ViT-B/16...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        config.clip_model_name, pretrained=config.clip_pretrained
    )
    visual_encoder = clip_model.visual.to(device)
    visual_encoder.eval()
    for p in visual_encoder.parameters():
        p.requires_grad = False

    transform = get_eval_transforms()
    batch_size = 32  # Conservative for MPS stability

    # Extract train features (subsampled)
    print(f"\n=== Extracting Train Features (subsampled to {MAX_TRAIN_SAMPLES}) ===")
    train_feat_path = feat_dir / "train_features.npy"
    if train_feat_path.exists():
        print("Train features already exist, skipping.")
    else:
        full_train = AIDetectDataset(config.data_dir, split="train", transform=transform)
        train_subset = subsample_dataset(full_train, MAX_TRAIN_SAMPLES)
        print(f"Subsampled: {len(train_subset)} from {len(full_train)}")

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        train_feats, train_labels = extract_features(visual_encoder, train_loader, device)
        np.save(feat_dir / "train_features.npy", train_feats.astype(np.float16))
        np.save(feat_dir / "train_labels.npy", train_labels)
        print(f"Saved: {train_feats.shape}, labels dist: real={sum(train_labels==0)}, fake={sum(train_labels==1)}")
        torch.mps.empty_cache() if device.type == "mps" else None

    # Extract val features (subsampled)
    print(f"\n=== Extracting Val Features (subsampled to {MAX_VAL_SAMPLES}) ===")
    val_feat_path = feat_dir / "val_features.npy"
    if val_feat_path.exists():
        print("Val features already exist, skipping.")
    else:
        full_val = AIDetectDataset(config.data_dir, split="val", transform=transform)
        val_subset = subsample_dataset(full_val, MAX_VAL_SAMPLES)
        print(f"Subsampled: {len(val_subset)} from {len(full_val)}")

        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        val_feats, val_labels = extract_features(visual_encoder, val_loader, device)
        np.save(feat_dir / "val_features.npy", val_feats.astype(np.float16))
        np.save(feat_dir / "val_labels.npy", val_labels)
        print(f"Saved: {val_feats.shape}")
        torch.mps.empty_cache() if device.type == "mps" else None

    # Extract test features (per generator — small, keep all)
    print("\n=== Extracting Test Features ===")
    for gen in config.test_generators:
        gen_feat_path = feat_dir / f"test_{gen}_features.npy"
        if gen_feat_path.exists():
            print(f"  {gen}: already exists, skipping.")
            continue

        gen_dir = config.data_dir / "test" / gen
        if not gen_dir.exists() or not any(gen_dir.rglob("*.jpg")):
            print(f"  {gen}: no test data, skipping.")
            continue

        test_ds = AIDetectTestDataset(config.data_dir, generator=gen, transform=transform)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_feats, test_labels = extract_features(visual_encoder, test_loader, device)
        np.save(feat_dir / f"test_{gen}_features.npy", test_feats.astype(np.float16))
        np.save(feat_dir / f"test_{gen}_labels.npy", test_labels)
        print(f"  {gen}: {test_feats.shape}")
        torch.mps.empty_cache() if device.type == "mps" else None

    # Summary
    print("\n=== Feature Extraction Complete ===")
    total_size = sum(f.stat().st_size for f in feat_dir.glob("*.npy"))
    print(f"Total feature files: {total_size / 1e6:.1f}MB")
    for f in sorted(feat_dir.glob("*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
