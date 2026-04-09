"""
Extract CLIP features for the FULL dataset using chunked CPU processing.

Processes in chunks to avoid MPS memory stalls. Uses CPU for stability.
"""

import gc
import shutil
from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.dataset import AIDetectDataset, AIDetectTestDataset
from src.seed import fix_seeds
from src.transforms import get_eval_transforms

CHUNK_SIZE = 5000  # Process 5K images at a time


def extract_chunked(
    encoder: torch.nn.Module,
    dataset: "Dataset",
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features in chunks for memory stability."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_feats = []
    all_labels = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            images = batch["image"].to(device)
            labels = batch["label"]
            feats = encoder(images).cpu().float().numpy()
            all_feats.append(feats)
            all_labels.append(labels.numpy())

            # Periodic cleanup
            if len(all_feats) % 100 == 0 and device.type == "mps":
                torch.mps.empty_cache()
                gc.collect()

    return np.concatenate(all_feats), np.concatenate(all_labels)


def main() -> None:
    """Extract full dataset features."""
    config = Config()
    fix_seeds(config.seed)

    # Use MPS but with smaller batches
    device = config.device
    print(f"Device: {device}")
    print(f"Disk: {shutil.disk_usage(Path.home())[2] / 1e9:.1f}GB free")

    feat_dir = config.project_root / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Backup old features
    for f in feat_dir.glob("*.npy"):
        backup = f.with_suffix(".npy.bak")
        if not backup.exists():
            f.rename(backup)
        else:
            f.unlink()

    # Load CLIP
    print("Loading CLIP ViT-B/16...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        config.clip_model_name, pretrained=config.clip_pretrained
    )
    encoder = clip_model.visual.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    transform = get_eval_transforms()

    # Train features (full 192K)
    print("\n=== Train Features (192K) ===")
    train_ds = AIDetectDataset(config.data_dir, split="train", transform=transform)
    print(f"Dataset size: {len(train_ds)}")
    train_feats, train_labels = extract_chunked(encoder, train_ds, device, batch_size=32)
    np.save(feat_dir / "train_features.npy", train_feats.astype(np.float16))
    np.save(feat_dir / "train_labels.npy", train_labels)
    print(f"Saved: {train_feats.shape}, real={sum(train_labels==0)}, fake={sum(train_labels==1)}")
    del train_feats, train_labels
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Val features (full 48K)
    print("\n=== Val Features (48K) ===")
    val_ds = AIDetectDataset(config.data_dir, split="val", transform=transform)
    print(f"Dataset size: {len(val_ds)}")
    val_feats, val_labels = extract_chunked(encoder, val_ds, device, batch_size=32)
    np.save(feat_dir / "val_features.npy", val_feats.astype(np.float16))
    np.save(feat_dir / "val_labels.npy", val_labels)
    print(f"Saved: {val_feats.shape}")
    del val_feats, val_labels
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Test features (per generator)
    print("\n=== Test Features ===")
    for gen in config.test_generators:
        gen_dir = config.data_dir / "test" / gen
        if not gen_dir.exists():
            continue

        test_ds = AIDetectTestDataset(config.data_dir, generator=gen, transform=transform)
        test_feats, test_labels = extract_chunked(encoder, test_ds, device, batch_size=32)
        np.save(feat_dir / f"test_{gen}_features.npy", test_feats.astype(np.float16))
        np.save(feat_dir / f"test_{gen}_labels.npy", test_labels)
        print(f"  {gen}: {test_feats.shape}")
        del test_feats, test_labels
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # Cleanup backups
    for f in feat_dir.glob("*.npy.bak"):
        f.unlink()

    print("\n=== Done ===")
    total = sum(f.stat().st_size for f in feat_dir.glob("*.npy"))
    print(f"Total: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
