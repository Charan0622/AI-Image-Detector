"""
Run robustness evaluation for all 5 model heads in ONE sweep.

Key insight: CLIP ViT-B/16 on MPS is the bottleneck (~48 img/s). The 5 trained
heads are tiny. So we extract CLIP features (+ DCT maps) ONCE per
(generator, degradation) and run all 5 heads on the cached tensors.

Model heads (all take pre-extracted 512-dim CLIP features):
    clip_probe             — LinearProbeHead
    hybrid                 — HybridFromFeatures             (needs DCT)
    hybrid_robust          — HybridRobustFromFeatures       (needs DCT)
    freq_guided_no_robust  — FreqGuidedFromFeatures         (needs DCT)
    freq_guided            — FreqGuidedFromFeatures         (needs DCT)

Degradations (8): clean, jpeg_q70/50/30, blur_s1/s2/s3, resize_112.

Outputs (one JSON per model):
    results/metrics/{model}_robustness.json

Usage:
    python -m scripts.run_all_robustness
    python -m scripts.run_all_robustness --gens sd15 midjourney --max_per_class 300
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import Config
from src.models.freq_guided import FreqGuidedFromFeatures
from src.seed import fix_seeds
from src.train_freq_guided import HybridRobustFromFeatures
from src.train_hybrid import HybridFromFeatures
from src.train_probe import LinearProbeHead
from src.transforms import compute_dct_map, get_eval_transforms


VARIANTS = {
    "clip_probe": {"cls": LinearProbeHead, "ckpt": "clip_probe_best.pth", "needs_dct": False},
    "hybrid": {"cls": HybridFromFeatures, "ckpt": "hybrid_best.pth", "needs_dct": True},
    "hybrid_robust": {"cls": HybridRobustFromFeatures, "ckpt": "hybrid_robust_best.pth", "needs_dct": True},
    "freq_guided_no_robust": {"cls": FreqGuidedFromFeatures, "ckpt": "freq_guided_no_robust_best.pth", "needs_dct": True},
    "freq_guided": {"cls": FreqGuidedFromFeatures, "ckpt": "freq_guided_best.pth", "needs_dct": True},
}


def apply_jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_resize(img: Image.Image, small: int) -> Image.Image:
    w, h = img.size
    s = img.resize((small, small), Image.LANCZOS)
    return s.resize((w, h), Image.LANCZOS)


def deg_clean(im): return im
def deg_jpeg70(im): return apply_jpeg(im, 70)
def deg_jpeg50(im): return apply_jpeg(im, 50)
def deg_jpeg30(im): return apply_jpeg(im, 30)
def deg_blur1(im): return apply_blur(im, 1.0)
def deg_blur2(im): return apply_blur(im, 2.0)
def deg_blur3(im): return apply_blur(im, 3.0)
def deg_resize112(im): return apply_resize(im, 112)


DEGRADATIONS = {
    "clean": deg_clean,
    "jpeg_q70": deg_jpeg70,
    "jpeg_q50": deg_jpeg50,
    "jpeg_q30": deg_jpeg30,
    "blur_s1": deg_blur1,
    "blur_s2": deg_blur2,
    "blur_s3": deg_blur3,
    "resize_112": deg_resize112,
}


class DegradedDataset(Dataset):
    """Decodes, degrades, transforms, and computes DCT in worker processes."""

    def __init__(self, paths: list[Path], deg_fn, transform) -> None:
        self.paths = paths
        self.deg_fn = deg_fn
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        im = Image.open(self.paths[idx]).convert("RGB")
        im = self.deg_fn(im)
        rgb = self.transform(im)
        d = compute_dct_map(im)
        dct = torch.from_numpy(d).unsqueeze(0).float()
        return rgb, dct


def load_clip_encoder(config: Config, device: torch.device) -> nn.Module:
    clip_model, _, _ = open_clip.create_model_and_transforms(
        config.clip_model_name, pretrained=config.clip_pretrained
    )
    encoder = clip_model.visual.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_head(name: str, config: Config, device: torch.device) -> nn.Module:
    info = VARIANTS[name]
    if info["cls"] is LinearProbeHead:
        kwargs = {"input_dim": config.clip_embed_dim}
    else:
        kwargs = {
            "clip_dim": config.clip_embed_dim,
            "freq_out_dim": config.freq_branch_out_dim,
            "fusion_hidden": config.fusion_hidden_dim,
            "fusion_dropout": config.fusion_dropout,
        }
    head = info["cls"](**kwargs)
    ckpt_path = config.checkpoint_dir / info["ckpt"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    head.load_state_dict(state)
    return head.to(device).eval()


def collect_test_paths(
    data_dir: Path, generator: str, max_per_class: int = 0
) -> tuple[list[Path], list[int]]:
    gen_dir = data_dir / "test" / generator
    paths: list[Path] = []
    labels: list[int] = []
    for label_name, label_int in [("real", 0), ("fake", 1)]:
        d = gen_dir / label_name
        if d.exists():
            class_paths = sorted(d.glob("*.jpg"))
            if max_per_class and len(class_paths) > max_per_class:
                class_paths = class_paths[:max_per_class]
            paths.extend(class_paths)
            labels.extend([label_int] * len(class_paths))
    return paths, labels


@torch.no_grad()
def extract_features(
    paths: list[Path],
    deg_fn,
    clip_encoder: nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream images through CLIP once; return stacked (feats, dct) on CPU."""
    ds = DegradedDataset(paths, deg_fn, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    feats_all: list[torch.Tensor] = []
    dct_all: list[torch.Tensor] = []
    for rgb, dct in loader:
        rgb = rgb.to(device)
        f = clip_encoder(rgb).cpu()
        feats_all.append(f)
        dct_all.append(dct)
    return torch.cat(feats_all), torch.cat(dct_all)


@torch.no_grad()
def score_heads(
    heads: dict[str, nn.Module],
    feats: torch.Tensor,
    dct: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run every head across the shared (feats, dct). Returns {name: (preds, probs)}."""
    results: dict[str, tuple[list[int], list[float]]] = {n: ([], []) for n in heads}
    n = feats.shape[0]
    for i in range(0, n, batch_size):
        f_b = feats[i : i + batch_size].to(device)
        d_b = dct[i : i + batch_size].to(device)
        for name, head in heads.items():
            needs_dct = VARIANTS[name]["needs_dct"]
            logits = head(f_b, d_b) if needs_dct else head(f_b)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            results[name][0].extend(preds.tolist())
            results[name][1].extend(probs.tolist())
    return {n: (np.array(p), np.array(q)) for n, (p, q) in results.items()}


def score_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = 0.5
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "auc": round(auc, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness eval on all model variants")
    parser.add_argument("--models", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    parser.add_argument("--gens", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_per_class", type=int, default=500, help="0 = use all")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)
    device = config.device
    gens = args.gens or config.test_generators
    transform = get_eval_transforms()

    print(f"Device: {device} | Models: {args.models}")
    print(f"Gens: {gens} | BS: {args.batch_size} | max_per_class: {args.max_per_class}", flush=True)

    print("Loading CLIP encoder (shared)...", flush=True)
    clip_encoder = load_clip_encoder(config, device)

    print("Loading 5 heads...", flush=True)
    heads: dict[str, nn.Module] = {}
    for m in args.models:
        try:
            heads[m] = load_head(m, config, device)
            print(f"  loaded {m}", flush=True)
        except FileNotFoundError as e:
            print(f"  SKIP {m}: {e}", flush=True)

    out_dir = config.results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # results_per_model[model][gen][deg] = {"accuracy": ..., "auc": ...}
    results_per_model: dict[str, dict] = {m: {} for m in heads}

    # Build path lists once
    paths_by_gen: dict = {}
    for gen in gens:
        p, l = collect_test_paths(config.data_dir, gen, args.max_per_class)
        if p:
            paths_by_gen[gen] = (p, np.array(l))
    total_imgs = sum(len(v[0]) for v in paths_by_gen.values())
    print(f"Total images/deg: {total_imgs} across {len(paths_by_gen)} gens\n", flush=True)

    total_t0 = time.time()
    for gen, (paths, labels) in paths_by_gen.items():
        for deg_name, deg_fn in DEGRADATIONS.items():
            t0 = time.time()
            feats, dct = extract_features(
                paths, deg_fn, clip_encoder, transform, device,
                args.batch_size, args.num_workers,
            )
            extr_dt = time.time() - t0

            head_scores = score_heads(heads, feats, dct, device)
            for m, (preds, probs) in head_scores.items():
                metrics = score_metrics(labels, preds, probs)
                results_per_model[m].setdefault(gen, {})[deg_name] = metrics

            deg_dt = time.time() - t0
            aucs = {
                m: results_per_model[m][gen][deg_name]["auc"]
                for m in heads
            }
            auc_str = "  ".join(f"{m[:10]}={a:.3f}" for m, a in aucs.items())
            print(f"  {gen}/{deg_name:10s} [{deg_dt:.1f}s] {auc_str}", flush=True)

            # Release memory between iterations (helps MPS especially)
            del feats, dct, head_scores
            if device.type == "mps":
                torch.mps.empty_cache()

        # Save incrementally after each generator so partial results are usable
        for m, per_gen_data in results_per_model.items():
            out_path = out_dir / f"{m}_robustness.json"
            with open(out_path, "w") as f:
                json.dump(per_gen_data, f, indent=2)
        print(f"  checkpointed all robustness JSONs after {gen}", flush=True)

    print(f"\nDone. Total wall time: {(time.time() - total_t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
