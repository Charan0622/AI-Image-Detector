"""
Expand the training set with diverse real-world and modern-AI samples.

The GenImage training set covers 2023-era generators and ImageNet-style real
photos. To improve real-world detection we inject:
    +1500 smartphone-style real photos (Lorem Picsum / random Unsplash)
    +650  modern AI generations (Pollinations.ai SANA / Flux / SDXL)

Pipeline per new image:
    download → canonicalize_for_inference (LANCZOS-224 + JPEG Q=95) → save
    → extract CLIP features → append to data/features/{train,val}_features.npy

15% of the new pool is held out into val (so calibration / OOD bookkeeping
sees the new distribution). Existing arrays are backed up to .npy.bak before
overwriting.

Usage:
    python -m scripts.expand_training_data
    python -m scripts.expand_training_data --n_real 800 --n_fake 300
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import open_clip
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Reuse Phase-0 downloader logic (same auth-free sources)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_realworld_eval import (  # noqa: E402
    PROMPTS,
    _is_valid_jpeg,
    download_picsum,
    download_pollinations,
)
from src.config import Config
from src.seed import fix_seeds
from src.transforms import get_eval_transforms


def canonicalize(img: Image.Image) -> Image.Image:
    """LANCZOS-224 center-crop + JPEG Q=95 round-trip — matches live /detect."""
    rgb = img.convert("RGB")
    short = min(rgb.size)
    left = (rgb.size[0] - short) // 2
    top = (rgb.size[1] - short) // 2
    rgb = rgb.crop((left, top, left + short, top + short))
    rgb = rgb.resize((224, 224), Image.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class CanonicalisedFolderDataset(Dataset):
    """Read image paths, run canonicalize + eval transform on each."""

    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        img = canonicalize(img)
        return self.transform(img)


def diverse_prompts(n: int) -> list[str]:
    """Recycle PROMPTS for `n` items by adding seeded modifiers."""
    modifiers = [
        "", " in winter", " at night", " black and white", " warm tones",
        " cool tones", " minimal", " high contrast", " moody",
        " backlit", " overhead view", " close up", " wide shot",
    ]
    rng = random.Random(0)
    out: list[str] = []
    while len(out) < n:
        for p in PROMPTS:
            mod = rng.choice(modifiers)
            out.append((p + mod).strip())
            if len(out) >= n:
                break
    return out


@torch.no_grad()
def extract_features(paths: list[Path], device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Run CLIP ViT-B/16 over canonicalised images, return (N, 512) on CPU."""
    config = Config()
    print("  loading CLIP encoder...", flush=True)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        config.clip_model_name, pretrained=config.clip_pretrained
    )
    encoder = clip_model.visual.to(device).eval()

    transform = get_eval_transforms()
    ds = CanonicalisedFolderDataset(paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    feats: list[np.ndarray] = []
    for batch in tqdm(loader, desc="  CLIP forward", leave=False):
        batch = batch.to(device)
        f = encoder(batch).cpu().float().numpy()
        feats.append(f)
        if device.type == "mps":
            torch.mps.empty_cache()
    return np.concatenate(feats)


def append_to_npy(feat_dir: Path, name: str, new_arr: np.ndarray) -> None:
    """Append new_arr to {name}.npy, backing up the original first."""
    path = feat_dir / f"{name}.npy"
    if path.exists():
        bak = feat_dir / f"{name}.npy.before_expand.bak"
        if not bak.exists():
            shutil.copy2(path, bak)
        old = np.load(path)
        merged = np.concatenate([old, new_arr.astype(old.dtype)])
    else:
        merged = new_arr
    np.save(path, merged)
    print(f"    {name}.npy now has {merged.shape[0]} rows (was {merged.shape[0] - new_arr.shape[0]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_real", type=int, default=1500)
    parser.add_argument("--n_fake", type=int, default=650)
    parser.add_argument("--val_holdout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloads, use whatever is already in data/processed/train/{real,fake}_extra")
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)
    device = config.device
    print(f"Device: {device}")

    real_dir = config.data_dir / "train" / "real_extra"
    fake_dir = config.data_dir / "train" / "fake_extra"

    # ---------- Download ----------
    if not args.skip_download:
        print(f"\nDownloading {args.n_real} real photos (picsum)...")
        download_picsum(real_dir, args.n_real)
        print(f"\nDownloading {args.n_fake} modern AI fakes (Pollinations.ai)...")
        download_pollinations(fake_dir, diverse_prompts(args.n_fake), "modern_ai")
    else:
        print("Skipping downloads (--skip_download).")

    # ---------- Canonicalise existing files in place ----------
    real_paths = sorted(real_dir.glob("*.jpg")) if real_dir.exists() else []
    fake_paths = sorted(fake_dir.glob("*.jpg")) if fake_dir.exists() else []
    print(f"\nFound {len(real_paths)} real and {len(fake_paths)} fake images on disk.")

    print("Canonicalising images in place (LANCZOS-224 + JPEG Q=95)...")
    for p in tqdm(real_paths + fake_paths, desc="  canonicalise"):
        try:
            with Image.open(p) as im:
                if im.size == (224, 224) and p.suffix == ".jpg":
                    # Probably already canonicalised — re-encode to be safe
                    pass
                canon = canonicalize(im.copy())
            canon.save(p, format="JPEG", quality=95)
        except Exception as e:
            tqdm.write(f"    skip {p}: {e}")

    # Re-list in case canonicalise left some unreadable; drop bad ones
    real_paths = [p for p in real_paths if _is_valid_jpeg(p.read_bytes())]
    fake_paths = [p for p in fake_paths if _is_valid_jpeg(p.read_bytes())]
    print(f"  retained: {len(real_paths)} real, {len(fake_paths)} fake")

    # ---------- Extract features ----------
    print("\nExtracting CLIP features for new images...")
    real_feats = extract_features(real_paths, device, args.batch_size) if real_paths else np.zeros((0, 512), dtype=np.float32)
    fake_feats = extract_features(fake_paths, device, args.batch_size) if fake_paths else np.zeros((0, 512), dtype=np.float32)
    print(f"  real_feats={real_feats.shape}  fake_feats={fake_feats.shape}")
    if real_feats.shape[0] == 0 and fake_feats.shape[0] == 0:
        print("Nothing to append. Exiting.")
        return

    # ---------- Train/val split (stratified, NO shuffling — features align with paths positionally) ----------
    rng = np.random.default_rng(0)
    def stratified_split(feats: np.ndarray, paths: list[Path], label: int):
        n = feats.shape[0]
        if n == 0:
            empty_f = np.zeros((0, 512), dtype=np.float32)
            empty_l = np.zeros(0, dtype=np.int64)
            return empty_f, empty_l, [], empty_f, empty_l, []
        n_val = int(round(n * args.val_holdout))
        idx = rng.permutation(n)
        val_idx = sorted(idx[:n_val].tolist())
        tr_idx = sorted(idx[n_val:].tolist())
        return (
            feats[tr_idx], np.full(len(tr_idx), label, dtype=np.int64), [paths[i] for i in tr_idx],
            feats[val_idx], np.full(len(val_idx), label, dtype=np.int64), [paths[i] for i in val_idx],
        )

    r_tr, r_tr_l, r_tr_p, r_val, r_val_l, r_val_p = stratified_split(real_feats, real_paths, 0)
    f_tr, f_tr_l, f_tr_p, f_val, f_val_l, f_val_p = stratified_split(fake_feats, fake_paths, 1)

    # Concat in real→fake order so training paths align positionally with features
    train_new_feats = np.concatenate([r_tr, f_tr], axis=0).astype(np.float16)
    train_new_labels = np.concatenate([r_tr_l, f_tr_l], axis=0).astype(np.int64)
    train_new_paths = r_tr_p + f_tr_p
    val_new_feats = np.concatenate([r_val, f_val], axis=0).astype(np.float16)
    val_new_labels = np.concatenate([r_val_l, f_val_l], axis=0).astype(np.int64)
    val_new_paths = r_val_p + f_val_p

    # ---------- Append to feature cache ----------
    feat_dir = config.project_root / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nAppending to feature cache at {feat_dir} (originals backed up to *.before_expand.bak)...")
    append_to_npy(feat_dir, "train_features", train_new_feats)
    append_to_npy(feat_dir, "train_labels", train_new_labels)
    append_to_npy(feat_dir, "val_features", val_new_feats)
    append_to_npy(feat_dir, "val_labels", val_new_labels)

    # Save the path lists so the v2 trainer can read DCT maps for the
    # appended rows. These paths are in the same positional order as the
    # newly appended feature rows.
    with open(feat_dir / "train_extra_paths.txt", "w") as f:
        for p in train_new_paths:
            f.write(f"{p}\n")
    with open(feat_dir / "val_extra_paths.txt", "w") as f:
        for p in val_new_paths:
            f.write(f"{p}\n")
    print(f"  wrote train_extra_paths.txt ({len(train_new_paths)} rows) and val_extra_paths.txt ({len(val_new_paths)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
