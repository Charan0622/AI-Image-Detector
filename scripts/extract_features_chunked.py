"""
Extract CLIP features for the FULL dataset in chunks.

Processes 4000 images at a time, saves intermediate results,
and clears MPS memory between chunks. Can resume if interrupted.
"""

import gc
import shutil
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import Config
from src.dataset import AIDetectDataset, AIDetectTestDataset
from src.seed import fix_seeds
from src.transforms import get_eval_transforms

CHUNK_SIZE = 4000


def extract_chunk(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for one chunk."""
    feats, labs = [], []
    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            f = encoder(images).cpu().float().numpy()
            feats.append(f)
            labs.append(batch["label"].numpy())
    return np.concatenate(feats), np.concatenate(labs)


def extract_split(
    encoder: torch.nn.Module,
    dataset: AIDetectDataset,
    device: torch.device,
    output_path: Path,
    label_path: Path,
) -> None:
    """Extract features for a full split in chunks."""
    if output_path.exists():
        existing = np.load(output_path)
        if existing.shape[0] == len(dataset):
            print(f"  Already complete: {output_path.name} ({existing.shape})")
            return
        print(f"  Partial: {existing.shape[0]}/{len(dataset)}, re-extracting...")

    n = len(dataset)
    all_feats, all_labs = [], []

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        subset = Subset(dataset, list(range(start, end)))
        loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=2)

        print(f"  Chunk {start}-{end} of {n}...")
        feats, labs = extract_chunk(encoder, loader, device)
        all_feats.append(feats)
        all_labs.append(labs)

        # Clear memory between chunks
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    all_feats = np.concatenate(all_feats)
    all_labs = np.concatenate(all_labs)

    np.save(output_path, all_feats.astype(np.float16))
    np.save(label_path, all_labs)
    print(f"  Saved: {all_feats.shape} to {output_path.name}")


def main() -> None:
    """Extract full dataset features."""
    config = Config()
    fix_seeds(config.seed)
    device = config.device

    feat_dir = config.project_root / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Disk: {shutil.disk_usage(Path.home())[2] / 1e9:.1f}GB free")

    # Load CLIP
    print("\nLoading CLIP ViT-B/16...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        config.clip_model_name, pretrained=config.clip_pretrained
    )
    encoder = clip_model.visual.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    transform = get_eval_transforms()

    # Full train (192K)
    print(f"\n=== Train Split ===")
    train_ds = AIDetectDataset(config.data_dir, split="train", transform=transform)
    print(f"  Dataset: {len(train_ds)} images")
    extract_split(encoder, train_ds, device,
                  feat_dir / "train_features.npy", feat_dir / "train_labels.npy")

    # Full val (48K)
    print(f"\n=== Val Split ===")
    val_ds = AIDetectDataset(config.data_dir, split="val", transform=transform)
    print(f"  Dataset: {len(val_ds)} images")
    extract_split(encoder, val_ds, device,
                  feat_dir / "val_features.npy", feat_dir / "val_labels.npy")

    # Test (per generator, already small)
    print(f"\n=== Test Splits ===")
    for gen in config.test_generators:
        gen_dir = config.data_dir / "test" / gen
        if not gen_dir.exists():
            continue
        test_ds = AIDetectTestDataset(config.data_dir, generator=gen, transform=transform)
        extract_split(encoder, test_ds, device,
                      feat_dir / f"test_{gen}_features.npy",
                      feat_dir / f"test_{gen}_labels.npy")

    print("\n=== Done ===")
    total = sum(f.stat().st_size for f in feat_dir.glob("*.npy"))
    print(f"Total: {total / 1e6:.1f} MB")
    for f in sorted(feat_dir.glob("*_features.npy")):
        arr = np.load(f)
        print(f"  {f.name}: {arr.shape}")


if __name__ == "__main__":
    main()
