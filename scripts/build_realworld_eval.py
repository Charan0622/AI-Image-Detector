"""
Build a real-world held-out evaluation set for the AI image detector.

The GenImage benchmark only covers 2023-era generators and ImageNet-style real
photos. Real-world uploads (smartphone snaps, modern AI generations) sit
outside that distribution. This script curates ~200 images from free-tier
sources so we can measure the true real-world FPR.

Sources (no API keys required):
    Real:
        - Lorem Picsum (https://picsum.photos) — random Unsplash photos, broad
          smartphone/portrait/landscape coverage.
    Fakes (modern AI):
        - Pollinations.ai Flux — modern flow-matching generator.
        - Pollinations.ai SDXL — Stable Diffusion XL.

Outputs:
    data/realworld_eval/{real,flux,sdxl}/*.jpg   — images, originals
    data/realworld_eval/manifest.csv             — path,source,label,subset

Usage:
    python -m scripts.build_realworld_eval
    python -m scripts.build_realworld_eval --n_real 100 --n_flux 30 --n_sdxl 30
"""

from __future__ import annotations

import argparse
import csv
import random
import time
import urllib.parse
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import Config


# Curated prompts for AI fakes, diverse content matching the kinds of images
# users actually upload (people, places, food, objects).
PROMPTS = [
    "a young woman sitting at a cafe in autumn, candid",
    "a beach sunset with palm trees, golden hour",
    "a cat sleeping on a wooden floor, natural light",
    "a busy street market in tokyo at night, neon",
    "a plate of pasta carbonara on a restaurant table",
    "a man walking his dog in central park, winter",
    "a mountain range in the swiss alps, snow",
    "a freshly baked sourdough loaf on a kitchen counter",
    "an old book on a wooden desk with reading glasses",
    "a colorful indian temple at dawn",
    "a kid playing in a sprinkler in summer",
    "a rural farm in tuscany at sunset",
    "a cup of espresso on a marble table, overhead",
    "a vintage car in havana cuba on a street",
    "a high school basketball game from the bleachers",
    "a chef plating a dish in a fine dining kitchen",
    "an old fishing village in norway with red cabins",
    "a woman doing yoga at sunrise on the beach",
    "a child blowing out birthday candles, family party",
    "a hiker on a trail in the rocky mountains",
    "a weathered old man with a kind smile, portrait",
    "a vintage typewriter on a writers desk",
    "a small bookstore interior with warm lighting",
    "an autumn forest path with golden leaves",
    "a couple kissing under fairy lights",
    "a freshly poured beer in a pub",
    "a modern apartment living room, scandinavian style",
    "a busy newsroom with reporters at desks",
    "a fisherman pulling in nets at dawn",
    "a busy farmers market with fresh produce",
    "a craft cocktail with citrus garnish, bar",
    "a foggy morning over a still lake",
    "a vintage record player with vinyl",
    "a chef sharpening a knife in a kitchen",
    "a dog running through a meadow",
    "a couple sharing a milkshake at a diner",
    "a girl reading on a windowsill, raining outside",
    "a wedding ceremony in a vineyard",
    "a child making cookies with their grandmother",
    "a busy airport terminal, travelers walking",
    "a man fixing a vintage motorcycle in a garage",
    "a woman painting at an easel in a studio",
    "a misty forest with tall pine trees",
    "a busy city intersection at night, time exposure",
    "a small wooden cabin in the woods, snow",
    "an indian wedding ceremony, vibrant colors",
    "a barista pouring latte art at a coffee shop",
    "a vintage bicycle leaning against a brick wall",
    "a violinist performing on stage, dramatic light",
    "a family playing board games at home",
    "a fishing boat on calm sea at sunrise",
    "a rooftop garden in a busy city",
    "a child looking through a magnifying glass",
    "a baker dusting flour on a workbench",
    "a small village in the swiss countryside",
    "a man reading a newspaper at a cafe",
    "a colorful hot air balloon festival at dawn",
    "a cellist practicing alone in a concert hall",
    "a kitchen full of fresh vegetables and herbs",
    "a runner stretching before a sunrise run",
]


def _is_valid_jpeg(content: bytes) -> bool:
    """Check JPEG magic bytes + minimum size."""
    return len(content) > 5000 and content[:2] == b"\xff\xd8"


def download_picsum(out_dir: Path, n: int) -> list[tuple[str, str]]:
    """Download n random photos from Lorem Picsum (random Unsplash photos)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    rng = random.Random(42)
    seeds = rng.sample(range(1, 100000), n * 3)  # plenty of retries
    seed_iter = iter(seeds)
    pbar = tqdm(total=n, desc="picsum")
    while len(rows) < n:
        seed = next(seed_iter, None)
        if seed is None:
            break
        url = f"https://picsum.photos/seed/{seed}/512/512"
        fn = out_dir / f"picsum_{seed:06d}.jpg"
        try:
            r = requests.get(url, timeout=25, allow_redirects=True)
            if r.status_code == 200 and _is_valid_jpeg(r.content):
                fn.write_bytes(r.content)
                rows.append((str(fn), "picsum"))
                pbar.update(1)
        except Exception:
            pass
        time.sleep(0.2)
    pbar.close()
    return rows


def download_pollinations(out_dir: Path, prompts: list[str], subset_label: str) -> list[tuple[str, str]]:
    """Download Pollinations.ai images for a list of prompts.

    Pollinations.ai is rate-limited and slow (30-60s per generation).
    We retry up to 3 times with backoff.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    for i, prompt in enumerate(tqdm(prompts, desc=f"pollinations/{subset_label}")):
        encoded = urllib.parse.quote(prompt)
        seed = 1000 + i
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&seed={seed}&nologo=true"
        fn = out_dir / f"{subset_label}_{i:03d}.jpg"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=180)
                if r.status_code == 200 and _is_valid_jpeg(r.content):
                    fn.write_bytes(r.content)
                    rows.append((str(fn), f"pollinations/{subset_label}"))
                    break
                else:
                    tqdm.write(f"  attempt {attempt + 1}: bad response (size={len(r.content)}, status={r.status_code})")
            except Exception as e:
                tqdm.write(f"  attempt {attempt + 1} for {prompt[:40]}: {e}")
            time.sleep(2.0 * (attempt + 1))
        time.sleep(1.0)  # be polite between prompts
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_real", type=int, default=100,
                        help="Number of real photos from Lorem Picsum")
    parser.add_argument("--n_modern_ai", type=int, default=60,
                        help="Number of modern AI generations from Pollinations.ai")
    parser.add_argument("--out_dir", type=Path,
                        default=Config().project_root / "data" / "realworld_eval")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building real-world eval set under {out_dir}")
    print(f"  {args.n_real} real (picsum)")
    print(f"  {args.n_modern_ai} fakes (Pollinations.ai modern model)")

    rows: list[tuple[str, str, str, str]] = []  # (path, source, label, subset)

    # Real photos
    real_dir = out_dir / "real"
    for path, src in download_picsum(real_dir, args.n_real):
        rows.append((path, src, "real", "real_picsum"))

    # Modern AI fakes — Pollinations.ai (currently serves SANA, a 2024 generator)
    fake_prompts = PROMPTS[: args.n_modern_ai]
    fake_dir = out_dir / "modern_ai"
    for path, src in download_pollinations(fake_dir, fake_prompts, "modern_ai"):
        rows.append((path, src, "fake", "fake_modern_ai"))

    # Manifest
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "label", "subset"])
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {manifest_path}")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r[3]] = counts.get(r[3], 0) + 1
    for subset, n in counts.items():
        print(f"  {subset:20s}  {n}")


if __name__ == "__main__":
    main()
