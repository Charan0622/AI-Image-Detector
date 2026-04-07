"""
Download GenImage dataset subsets from HuggingFace.

Downloads from RohanRamesh/genimage-224 (already 224x224):
- Full train split (240K images, ~3.4GB) for training
- Subset of test split (1000 real + 1000 fake per generator) for cross-gen eval

Generator mapping:
    0: adm, 1: glide, 2: midjourney, 3: sd15, 4: vqdm, 5: wukong

Label mapping:
    0: ai (fake), 1: nature (real)

Checks disk space before every download step.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

# Fix seeds
random.seed(42)
np.random.seed(42)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Generator mapping (from dataset README)
GENERATOR_MAP = {
    0: "adm",
    1: "glide",
    2: "midjourney",
    3: "sd15",
    4: "vqdm",
    5: "wukong",
}

# Label mapping (from dataset README)
# 0 = ai (fake), 1 = nature (real)
LABEL_MAP = {0: "fake", 1: "real"}

# How many images per class per generator for test evaluation
TEST_SAMPLES_PER_CLASS = 1000


def check_disk_space(min_gb: float = 3.0) -> bool:
    """Check if there's enough free disk space."""
    total, used, free = shutil.disk_usage(Path.home())
    free_gb = free / 1e9
    print(f"Disk space: {free_gb:.1f}GB free")
    if free_gb < min_gb:
        print(f"WARNING: Less than {min_gb}GB free. Aborting.")
        return False
    return True


def download_train_split() -> None:
    """Download the full train split from HuggingFace."""
    from datasets import load_dataset

    print("\n=== Downloading Train Split (240K images, ~3.4GB) ===")

    if not check_disk_space(5.0):
        sys.exit(1)

    # Create output directories
    train_real = PROCESSED_DIR / "train" / "real"
    train_fake = PROCESSED_DIR / "train" / "fake"
    train_real.mkdir(parents=True, exist_ok=True)
    train_fake.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_real = len(list(train_real.glob("*.jpg")))
    existing_fake = len(list(train_fake.glob("*.jpg")))
    if existing_real > 1000 and existing_fake > 1000:
        print(f"Train data already exists: {existing_real} real, {existing_fake} fake")
        return

    # Load dataset with streaming to avoid OOM
    print("Loading train split from HuggingFace (streaming)...")
    ds = load_dataset("RohanRamesh/genimage-224", split="train", streaming=True)

    real_count = 0
    fake_count = 0
    gen_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"real": 0, "fake": 0})

    for i, row in enumerate(tqdm(ds, total=240000, desc="Downloading train")):
        label = row["label"]  # 0=ai/fake, 1=nature/real
        generator = row["generator"]
        gen_name = GENERATOR_MAP[generator]
        label_name = LABEL_MAP[label]
        img: Image.Image = row["image"]

        # Save to appropriate directory
        if label == 1:  # real
            out_path = train_real / f"{gen_name}_{real_count:06d}.jpg"
            real_count += 1
        else:  # fake
            out_path = train_fake / f"{gen_name}_{fake_count:06d}.jpg"
            fake_count += 1

        gen_counts[gen_name][label_name] += 1

        # Save as JPEG Q=95 (normalize compression)
        img.save(out_path, "JPEG", quality=95)

        # Progress update every 10K
        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i+1}/240000 | Real: {real_count}, Fake: {fake_count}")

    print(f"\nTrain download complete: {real_count} real, {fake_count} fake")
    print("Per-generator counts:")
    for gen, counts in sorted(gen_counts.items()):
        print(f"  {gen}: real={counts['real']}, fake={counts['fake']}")


def download_test_split() -> None:
    """Download a balanced subset of the test split for cross-gen evaluation."""
    from datasets import load_dataset

    print(f"\n=== Downloading Test Split ({TEST_SAMPLES_PER_CLASS} per class per gen) ===")

    if not check_disk_space(2.0):
        sys.exit(1)

    # Create output directories for each generator
    for gen_name in GENERATOR_MAP.values():
        for label in ["real", "fake"]:
            out_dir = PROCESSED_DIR / "test" / gen_name / label
            out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    test_dir = PROCESSED_DIR / "test"
    existing = sum(1 for _ in test_dir.rglob("*.jpg"))
    if existing > 1000:
        print(f"Test data already exists: {existing} images")
        return

    # Track per-generator, per-label counts
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"real": 0, "fake": 0})
    target = TEST_SAMPLES_PER_CLASS

    print("Loading test split from HuggingFace (streaming)...")
    ds = load_dataset("RohanRamesh/genimage-224", split="test", streaming=True)

    total_saved = 0
    total_target = len(GENERATOR_MAP) * 2 * target  # 6 gens * 2 labels * 1000

    for row in tqdm(ds, desc="Downloading test"):
        label = row["label"]
        generator = row["generator"]
        gen_name = GENERATOR_MAP[generator]
        label_name = LABEL_MAP[label]

        # Skip if we have enough for this generator+label combo
        if counts[gen_name][label_name] >= target:
            # Check if we have all we need
            if all(
                counts[g][l] >= target
                for g in GENERATOR_MAP.values()
                for l in ["real", "fake"]
            ):
                break
            continue

        img: Image.Image = row["image"]
        idx = counts[gen_name][label_name]
        out_path = PROCESSED_DIR / "test" / gen_name / label_name / f"{idx:04d}.jpg"
        img.save(out_path, "JPEG", quality=95)

        counts[gen_name][label_name] += 1
        total_saved += 1

        if total_saved % 1000 == 0:
            print(f"  Saved {total_saved}/{total_target} test images")

    print(f"\nTest download complete: {total_saved} images")
    print("Per-generator counts:")
    for gen, c in sorted(counts.items()):
        print(f"  {gen}: real={c['real']}, fake={c['fake']}")


def create_val_split() -> None:
    """Create validation split from train data (80/20 stratified split)."""
    print("\n=== Creating Validation Split (20% of train) ===")

    train_real = PROCESSED_DIR / "train" / "real"
    train_fake = PROCESSED_DIR / "train" / "fake"
    val_real = PROCESSED_DIR / "val" / "real"
    val_fake = PROCESSED_DIR / "val" / "fake"

    val_real.mkdir(parents=True, exist_ok=True)
    val_fake.mkdir(parents=True, exist_ok=True)

    # Check if already done
    if len(list(val_real.glob("*.jpg"))) > 100:
        print("Val split already exists, skipping.")
        return

    for label_name, src_dir, dst_dir in [
        ("real", train_real, val_real),
        ("fake", train_fake, val_fake),
    ]:
        files = sorted(src_dir.glob("*.jpg"))
        random.shuffle(files)
        val_count = int(len(files) * 0.2)
        val_files = files[:val_count]

        print(f"Moving {val_count} {label_name} images to val split...")
        for f in tqdm(val_files, desc=f"Val {label_name}"):
            f.rename(dst_dir / f.name)

        print(f"  {label_name}: {len(files) - val_count} train, {val_count} val")


def generate_manifest() -> None:
    """Generate data manifest JSON with counts and metadata."""
    print("\n=== Generating Data Manifest ===")

    manifest: dict = {
        "source": "RohanRamesh/genimage-224 (HuggingFace)",
        "generators": list(GENERATOR_MAP.values()),
        "label_mapping": {"0_ai": "fake", "1_nature": "real"},
        "image_size": [224, 224],
        "jpeg_quality": 95,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        split_dir = PROCESSED_DIR / split
        if not split_dir.exists():
            continue

        if split == "test":
            # Test has per-generator subdirectories
            split_data: dict = {}
            for gen_dir in sorted(split_dir.iterdir()):
                if gen_dir.is_dir():
                    gen_data: dict = {}
                    for label_dir in sorted(gen_dir.iterdir()):
                        if label_dir.is_dir():
                            count = len(list(label_dir.glob("*.jpg")))
                            gen_data[label_dir.name] = count
                    split_data[gen_dir.name] = gen_data
            manifest["splits"]["test"] = split_data
        else:
            split_data = {}
            for label_dir in sorted(split_dir.iterdir()):
                if label_dir.is_dir():
                    count = len(list(label_dir.glob("*.jpg")))
                    split_data[label_dir.name] = count
            manifest["splits"][split] = split_data

    # Disk usage
    total_size = sum(
        f.stat().st_size for f in PROCESSED_DIR.rglob("*.jpg")
    )
    manifest["total_disk_usage_mb"] = round(total_size / 1e6, 1)

    manifest_path = DATA_DIR / "data_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    """Run the full data download pipeline."""
    print("=" * 60)
    print("GenImage Dataset Download Script")
    print("=" * 60)

    # Step 0: Disk check
    if not check_disk_space(5.0):
        print("Not enough disk space. Need at least 5GB free.")
        sys.exit(1)

    # Step 1: Download train split
    download_train_split()

    # Step 2: Download test split (subset for cross-gen eval)
    download_test_split()

    # Step 3: Create val split from train
    create_val_split()

    # Step 4: Generate manifest
    generate_manifest()

    # Final disk check
    print("\n=== Final Disk Check ===")
    check_disk_space(1.0)
    print("\n✅ Data download and preprocessing complete!")


if __name__ == "__main__":
    main()
