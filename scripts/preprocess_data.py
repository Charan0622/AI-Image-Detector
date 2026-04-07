"""
Preprocess all downloaded images.

Since we download from RohanRamesh/genimage-224 which is already
224x224 and we save as JPEG Q=95 during download, this script
handles:
    1. Verify all images are valid and 224x224
    2. Re-save any non-JPEG images as JPEG Q=95
    3. Create 80/20 train/val split (if not already done by download script)
    4. Generate data_manifest.json with counts
    5. Print data statistics

The download script already handles most preprocessing, so this
script is mainly for verification and fixing any issues.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def verify_images() -> dict:
    """Verify all images are valid JPEGs at 224x224.

    Returns:
        Dictionary with verification results.
    """
    print("=== Verifying Images ===")
    stats: dict = {"valid": 0, "invalid": 0, "wrong_size": 0, "fixed": 0}

    for img_path in tqdm(list(PROCESSED_DIR.rglob("*.jpg")), desc="Verifying"):
        try:
            img = Image.open(img_path)
            img.verify()
            img = Image.open(img_path)
            if img.size != (224, 224):
                stats["wrong_size"] += 1
                # Resize and re-save
                img = img.resize((224, 224), Image.LANCZOS)
                img.save(img_path, "JPEG", quality=95)
                stats["fixed"] += 1
            else:
                stats["valid"] += 1
        except Exception as e:
            stats["invalid"] += 1
            print(f"  Invalid: {img_path} — {e}")

    print(f"Results: {stats}")
    return stats


def count_images() -> dict:
    """Count images per split, per class, per generator.

    Returns:
        Nested dictionary with counts.
    """
    print("\n=== Image Counts ===")
    counts: dict = {}

    # Train and val splits
    for split in ["train", "val"]:
        split_dir = PROCESSED_DIR / split
        if not split_dir.exists():
            continue
        split_counts: dict = {}
        gen_counts: dict = defaultdict(int)
        for label in ["real", "fake"]:
            label_dir = split_dir / label
            if label_dir.exists():
                files = list(label_dir.glob("*.jpg"))
                split_counts[label] = len(files)
                for f in files:
                    gen = f.stem.rsplit("_", 1)[0]
                    gen_counts[gen] += 1
        counts[split] = {"labels": split_counts, "generators": dict(gen_counts)}
        total = sum(split_counts.values())
        print(f"{split}: {total} images ({split_counts})")

    # Test split (per-generator)
    test_dir = PROCESSED_DIR / "test"
    if test_dir.exists():
        test_counts: dict = {}
        for gen_dir in sorted(test_dir.iterdir()):
            if gen_dir.is_dir():
                gen_data: dict = {}
                for label_dir in sorted(gen_dir.iterdir()):
                    if label_dir.is_dir():
                        count = len(list(label_dir.glob("*.jpg")))
                        gen_data[label_dir.name] = count
                if gen_data:
                    test_counts[gen_dir.name] = gen_data
        counts["test"] = test_counts
        print(f"test: {test_counts}")

    return counts


def main() -> None:
    """Run preprocessing verification."""
    print("=" * 60)
    print("Data Preprocessing & Verification")
    print("=" * 60)

    # Verify images
    stats = verify_images()
    if stats["invalid"] > 0:
        print(f"\nWARNING: {stats['invalid']} invalid images found!")

    # Count images
    counts = count_images()

    # Save manifest
    manifest = {
        "source": "RohanRamesh/genimage-224 (HuggingFace)",
        "generators": ["adm", "glide", "midjourney", "sd15", "vqdm", "wukong"],
        "image_size": [224, 224],
        "jpeg_quality": 95,
        "splits": counts,
        "verification": stats,
    }

    manifest_path = DATA_DIR / "data_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
