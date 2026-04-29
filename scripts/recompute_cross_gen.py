"""
Re-evaluate all 5 model variants on the cross-generator test set, using the
same pipeline the live /detect endpoint uses (LANCZOS-224 + JPEG-Q95
canonicalization, shared CLIP encoder, temperature-calibrated softmax).

Outputs one JSON per model at results/metrics/{model}_cross_gen.json with
accuracy, AUC, precision, recall, F1, and a confusion matrix per generator.

Usage:
    python -m scripts.recompute_cross_gen
    python -m scripts.recompute_cross_gen --gens sd15 vqdm
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
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
    "clip_probe":             {"cls": LinearProbeHead,         "ckpt": "clip_probe_best.pth",            "needs_dct": False},
    "hybrid":                 {"cls": HybridFromFeatures,      "ckpt": "hybrid_best.pth",                "needs_dct": True},
    "hybrid_robust":          {"cls": HybridRobustFromFeatures,"ckpt": "hybrid_robust_best.pth",         "needs_dct": True},
    "freq_guided_no_robust":  {"cls": FreqGuidedFromFeatures,  "ckpt": "freq_guided_no_robust_best.pth", "needs_dct": True},
    "freq_guided":            {"cls": FreqGuidedFromFeatures,  "ckpt": "freq_guided_best.pth",           "needs_dct": True},
}


def canonicalize(img: Image.Image) -> Image.Image:
    """LANCZOS-224 center crop + JPEG Q=95 round-trip — matches the live /detect path."""
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


class TestDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = canonicalize(Image.open(self.paths[i]))
        rgb = self.transform(img)
        dct = torch.from_numpy(compute_dct_map(img)).unsqueeze(0).float()
        return rgb, dct


def load_clip(config, device):
    m, _, _ = open_clip.create_model_and_transforms(config.clip_model_name, pretrained=config.clip_pretrained)
    enc = m.visual.to(device).eval()
    for p in enc.parameters(): p.requires_grad = False
    return enc


def load_head(name, config, device):
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
    ckpt = torch.load(config.checkpoint_dir / info["ckpt"], map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    head.load_state_dict(state)
    return head.to(device).eval()


def collect_paths(data_dir: Path, gen: str) -> tuple[list[Path], np.ndarray]:
    paths, labels = [], []
    for label_name, label_int in [("real", 0), ("fake", 1)]:
        d = data_dir / "test" / gen / label_name
        if d.exists():
            for p in sorted(d.glob("*.jpg")):
                paths.append(p); labels.append(label_int)
    return paths, np.array(labels, dtype=np.int64)


@torch.no_grad()
def extract(paths, clip_enc, transform, device, bs, nw):
    """Returns (feats: (N, 512) cpu, dcts: (N, 1, 224, 224) cpu)."""
    ds = TestDataset(paths, transform)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=False)
    feats, dcts = [], []
    for rgb, dct in tqdm(loader, desc="  extract", leave=False):
        feats.append(clip_enc(rgb.to(device)).cpu())
        dcts.append(dct)
    return torch.cat(feats), torch.cat(dcts)


@torch.no_grad()
def score_head(head, needs_dct, feats, dct, T, device, bs=256):
    n = feats.shape[0]
    probs_all = []
    preds_all = []
    for i in range(0, n, bs):
        f_b = feats[i:i + bs].to(device)
        if needs_dct:
            d_b = dct[i:i + bs].to(device)
            logits = head(f_b, d_b)
        else:
            logits = head(f_b)
        probs = F.softmax(logits / max(T, 1e-3), dim=1)
        probs_all.append(probs[:, 1].cpu().numpy())
        preds_all.append(probs.argmax(dim=1).cpu().numpy())
    return np.concatenate(probs_all), np.concatenate(preds_all)


def metrics_for(labels, preds, probs):
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = 0.5
    return {
        "accuracy":  round(float(accuracy_score(labels, preds)), 4),
        "auc":       round(auc, 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "n_samples": int(len(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    parser.add_argument("--gens", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)
    device = config.device
    gens = args.gens or config.test_generators
    transform = get_eval_transforms()

    # Load calibration temperatures
    calib_path = config.results_dir / "metrics" / "calibration.json"
    temps = {}
    if calib_path.exists():
        with open(calib_path) as f:
            temps = {m: float(v["temperature"]) for m, v in json.load(f).items()}
    print(f"Device: {device}  Models: {args.models}  Gens: {gens}")
    print(f"Temperatures: { {m: round(temps.get(m, 1.0), 3) for m in args.models} }")

    print("\nLoading CLIP encoder + 5 heads…")
    clip_enc = load_clip(config, device)
    heads = {m: load_head(m, config, device) for m in args.models}

    out_dir = config.results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract features ONCE per generator, score every head on that pool
    per_model_results: dict[str, dict] = {m: {} for m in args.models}

    total_t0 = time.time()
    for gen in gens:
        paths, labels = collect_paths(config.data_dir, gen)
        if not paths:
            print(f"\n[{gen}] no test data, skipping")
            continue
        print(f"\n[{gen}] {len(paths)} images")
        t0 = time.time()
        feats, dct = extract(paths, clip_enc, transform, device, args.batch_size, args.num_workers)
        print(f"  features extracted in {time.time() - t0:.1f}s")

        for m in args.models:
            T = temps.get(m, 1.0)
            probs, preds = score_head(heads[m], VARIANTS[m]["needs_dct"], feats, dct, T, device)
            metrics = metrics_for(labels, preds, probs)
            per_model_results[m][gen] = metrics
            print(f"    {m:24s}  acc={metrics['accuracy']:.4f}  auc={metrics['auc']:.4f}  f1={metrics['f1']:.4f}")

        del feats, dct
        if device.type == "mps":
            torch.mps.empty_cache()

        # Persist after each generator so partial results are usable
        for m in args.models:
            with open(out_dir / f"{m}_cross_gen.json", "w") as f:
                json.dump(per_model_results[m], f, indent=2)

    print(f"\nWrote {len(args.models)} cross_gen.json files to {out_dir}")
    print(f"Total wall time: {(time.time() - total_t0) / 60:.1f} min")

    # Summary
    print("\n=== Summary (avg AUC across generators) ===")
    for m in args.models:
        aucs = [v["auc"] for v in per_model_results[m].values()]
        accs = [v["accuracy"] for v in per_model_results[m].values()]
        print(f"  {m:24s}  acc={np.mean(accs):.4f}  auc={np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()
