"""
Aggressive v3 data expansion:
    +3000 diverse real photos (parallel Lorem Picsum downloader)
    +1500 modern AI images (HuggingFace `poloclub/diffusiondb` SDXL subset)

Saves to:
    data/processed/train/real_extra/   (canonicalised)
    data/processed/train/fake_extra/   (canonicalised)
    data/features/train_features.npy   (appended; backup at .v2.bak)
    data/features/val_features.npy     (appended; backup at .v2.bak)
    data/features/{train,val}_extra_paths.txt   (overwritten)

Notes
-----
This *appends to whatever's already in the feature cache*. If you've already
run scripts/expand_training_data.py, those rows stay; this adds more on top.
The `.v2.bak` copy preserves the post-Phase-2 state so we can roll back to
v2 cleanly if needed.

Usage
-----
    python -m scripts.expand_v3
    python -m scripts.expand_v3 --n_real 1500 --n_fake 800
    python -m scripts.expand_v3 --skip_download   # only extract+append
"""

from __future__ import annotations

import argparse
import asyncio
import io
import random
import shutil
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config
from src.seed import fix_seeds
from src.transforms import get_eval_transforms

from scripts.expand_training_data import (  # noqa: E402
    CanonicalisedFolderDataset,
    canonicalize,
)


def _is_valid_jpeg(content: bytes) -> bool:
    return len(content) > 5000 and content[:2] == b"\xff\xd8"


# ---------- Real photos: parallel Lorem Picsum ----------
async def _fetch_one(session: aiohttp.ClientSession, seed: int, out_dir: Path) -> Path | None:
    fn = out_dir / f"picsum_v3_{seed:06d}.jpg"
    if fn.exists() and fn.stat().st_size > 5000:
        return fn  # already have it
    url = f"https://picsum.photos/seed/{seed}/512/512"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            content = await r.read()
        if r.status == 200 and _is_valid_jpeg(content):
            fn.write_bytes(content)
            return fn
    except Exception:
        return None
    return None


async def _picsum_async(out_dir: Path, n: int, concurrency: int = 16) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0xc0ffee)
    seeds = rng.sample(range(100000, 999999), n * 2)  # extras for retries
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        async def bounded(s: int) -> Path | None:
            async with sem:
                return await _fetch_one(session, s, out_dir)

        ok = 0
        pbar = tqdm(total=n, desc="picsum (parallel)")
        # Process in waves so progress updates flow
        seed_iter = iter(seeds)
        wave_size = concurrency * 4
        while ok < n:
            wave = list(itertools_islice(seed_iter, wave_size))
            if not wave:
                break
            results = await asyncio.gather(*(bounded(s) for s in wave))
            for r in results:
                if r and ok < n:
                    ok += 1
                    pbar.update(1)
        pbar.close()
        return ok


def itertools_islice(it, n):
    out = []
    for _ in range(n):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out


def download_picsum_parallel(out_dir: Path, n: int, concurrency: int = 16) -> int:
    return asyncio.run(_picsum_async(out_dir, n, concurrency))


# ---------- AI photos: HuggingFace diffusiondb (SDXL subset) ----------
def download_diffusiondb(out_dir: Path, n: int) -> int:
    """Stream a small slice of poloclub/diffusiondb SDXL images.

    Falls back to Pollinations.ai (slow but always available) if HF
    streaming fails or auth is required.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets package not installed — falling back to Pollinations")
        return _download_pollinations_fallback(out_dir, n)

    print("  trying poloclub/diffusiondb (large_first_1k SDXL subset)...")
    try:
        # The 'large_first_1k' subset is the smallest, ~1000 SDXL images
        ds = load_dataset("poloclub/diffusiondb",
                          name="large_random_1k",
                          split="train",
                          streaming=True,
                          trust_remote_code=False)
    except Exception as e:
        print(f"  HF dataset load failed: {e}")
        print("  falling back to Pollinations.ai")
        return _download_pollinations_fallback(out_dir, n)

    saved = 0
    for i, row in enumerate(tqdm(ds, total=n, desc="diffusiondb")):
        if saved >= n:
            break
        try:
            img = row["image"]  # PIL.Image
            fn = out_dir / f"diffdb_{i:04d}.jpg"
            img.convert("RGB").save(fn, format="JPEG", quality=95)
            saved += 1
        except Exception as e:
            tqdm.write(f"  skip row {i}: {e}")
    return saved


def _download_pollinations_fallback(out_dir: Path, n: int) -> int:
    """Last-resort fallback (slow). Uses scripts.build_realworld_eval logic."""
    from scripts.build_realworld_eval import PROMPTS, download_pollinations
    rng = random.Random(99)
    prompts = [PROMPTS[i % len(PROMPTS)] + (
        " " + rng.choice(["realistic", "cinematic", "photographic", "candid"])
    ) for i in range(n)]
    rows = download_pollinations(out_dir, prompts, "v3_fake")
    return len(rows)


# ---------- CLIP feature extraction ----------
@torch.no_grad()
def extract_features(paths, device, batch_size=32):
    config = Config()
    print("  loading CLIP encoder...", flush=True)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        config.clip_model_name, pretrained=config.clip_pretrained
    )
    encoder = clip_model.visual.to(device).eval()
    transform = get_eval_transforms()
    ds = CanonicalisedFolderDataset(paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    feats = []
    for batch in tqdm(loader, desc="  CLIP forward"):
        feats.append(encoder(batch.to(device)).cpu().float().numpy())
        if device.type == "mps":
            torch.mps.empty_cache()
    return np.concatenate(feats) if feats else np.zeros((0, 512), np.float32)


def append_to_npy(feat_dir: Path, name: str, new_arr: np.ndarray) -> None:
    """Append new_arr to {name}.npy with .v2.bak preserved (only once)."""
    path = feat_dir / f"{name}.npy"
    if path.exists():
        bak = feat_dir / f"{name}.npy.v2.bak"
        if not bak.exists():
            shutil.copy2(path, bak)
        old = np.load(path)
        merged = np.concatenate([old, new_arr.astype(old.dtype)])
    else:
        merged = new_arr
    np.save(path, merged)
    print(f"  {name}.npy: {merged.shape[0]} rows (added {new_arr.shape[0]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_real", type=int, default=3000)
    parser.add_argument("--n_fake", type=int, default=1500)
    parser.add_argument("--val_holdout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Parallel HTTP requests for picsum")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)
    device = config.device
    print(f"Device: {device}")

    real_dir = config.data_dir / "train" / "real_extra"
    fake_dir = config.data_dir / "train" / "fake_extra"

    if not args.skip_download:
        print(f"\n[1/3] {args.n_real} real photos via parallel picsum (concurrency={args.concurrency})...")
        n_got = download_picsum_parallel(real_dir, args.n_real, args.concurrency)
        print(f"  got {n_got} real photos")

        print(f"\n[2/3] {args.n_fake} modern AI images via HuggingFace diffusiondb...")
        n_got_fake = download_diffusiondb(fake_dir, args.n_fake)
        print(f"  got {n_got_fake} fake images")
    else:
        print("Skipping download.")

    # Reuse expand_training_data's canonicalise + extract-and-append logic
    real_paths = sorted(real_dir.glob("*.jpg"))
    fake_paths = sorted(fake_dir.glob("*.jpg"))
    print(f"\nFound on disk: {len(real_paths)} real, {len(fake_paths)} fake")

    print("Canonicalising in place (LANCZOS-224 + JPEG Q=95)...")
    for p in tqdm(real_paths + fake_paths):
        try:
            with Image.open(p) as im:
                canon = canonicalize(im.copy())
            canon.save(p, format="JPEG", quality=95)
        except Exception as e:
            tqdm.write(f"  skip {p}: {e}")

    real_paths = [p for p in real_paths if _is_valid_jpeg(p.read_bytes())]
    fake_paths = [p for p in fake_paths if _is_valid_jpeg(p.read_bytes())]
    print(f"  retained: {len(real_paths)} real, {len(fake_paths)} fake")

    if not real_paths and not fake_paths:
        print("Nothing to extract. Exiting.")
        return

    print(f"\n[3/3] Extracting CLIP features for new images...")
    real_feats = extract_features(real_paths, device, args.batch_size) if real_paths else np.zeros((0, 512), np.float32)
    fake_feats = extract_features(fake_paths, device, args.batch_size) if fake_paths else np.zeros((0, 512), np.float32)
    print(f"  real_feats={real_feats.shape}  fake_feats={fake_feats.shape}")

    rng = np.random.default_rng(7)
    def split(feats, paths, label):
        n = feats.shape[0]
        if n == 0:
            return np.zeros((0,512), np.float32), np.zeros(0, np.int64), [], np.zeros((0,512), np.float32), np.zeros(0, np.int64), []
        n_val = int(round(n * args.val_holdout))
        idx = rng.permutation(n)
        val_idx = sorted(idx[:n_val].tolist())
        tr_idx = sorted(idx[n_val:].tolist())
        return (feats[tr_idx], np.full(len(tr_idx), label, np.int64), [paths[i] for i in tr_idx],
                feats[val_idx], np.full(len(val_idx), label, np.int64), [paths[i] for i in val_idx])

    r_tr, r_tr_l, r_tr_p, r_val, r_val_l, r_val_p = split(real_feats, real_paths, 0)
    f_tr, f_tr_l, f_tr_p, f_val, f_val_l, f_val_p = split(fake_feats, fake_paths, 1)

    train_new_feats = np.concatenate([r_tr, f_tr]).astype(np.float16)
    train_new_labels = np.concatenate([r_tr_l, f_tr_l]).astype(np.int64)
    train_new_paths = r_tr_p + f_tr_p
    val_new_feats = np.concatenate([r_val, f_val]).astype(np.float16)
    val_new_labels = np.concatenate([r_val_l, f_val_l]).astype(np.int64)
    val_new_paths = r_val_p + f_val_p

    feat_dir = config.project_root / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nAppending to feature cache at {feat_dir} (originals backed up to *.v2.bak)...")
    append_to_npy(feat_dir, "train_features", train_new_feats)
    append_to_npy(feat_dir, "train_labels", train_new_labels)
    append_to_npy(feat_dir, "val_features", val_new_feats)
    append_to_npy(feat_dir, "val_labels", val_new_labels)

    # Path files: APPEND to existing extra_paths.txt (positional alignment)
    train_extra_path = feat_dir / "train_extra_paths.txt"
    val_extra_path = feat_dir / "val_extra_paths.txt"
    with open(train_extra_path, "a") as f:
        for p in train_new_paths:
            f.write(f"{p}\n")
    with open(val_extra_path, "a") as f:
        for p in val_new_paths:
            f.write(f"{p}\n")
    print(f"  appended {len(train_new_paths)} train paths, {len(val_new_paths)} val paths")
    print("\nDone.")


if __name__ == "__main__":
    main()
