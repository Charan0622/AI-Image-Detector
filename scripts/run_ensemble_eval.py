"""
Evaluate ensemble models on cross-generator clean + robustness.

Reuses the shared-CLIP architecture from run_all_robustness: for each
(generator, degradation) pair, extract features once and score every head.
We then form 3 ensembles from the calibrated probabilities:

    ensemble_all      — equal-weight mean of all 5 heads
    ensemble_top3     — mean of (hybrid_robust, freq_guided_no_robust, clip_probe).
                        These three have the most diverse inductive biases:
                        concat+aug fusion, freq-guided attention, and pure
                        semantic CLIP, respectively. Pairs well for
                        decorrelated ensemble members.
    ensemble_weighted — probs weighted by each model's val AUC (from training JSON)

Calibrated probabilities are used (softmax(logits / T)) when calibration.json
is present; otherwise T=1.

Outputs:
    results/metrics/ensemble_cross_gen.json      — per-gen clean metrics for each ensemble
    results/metrics/ensemble_robustness.json     — per-gen x deg metrics for each ensemble
    results/tables/ensemble_comparison.md        — ensemble vs best-single summary

Usage:
    python -m scripts.run_ensemble_eval
    python -m scripts.run_ensemble_eval --gens sd15 midjourney --max_per_class 300
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
import torch.nn.functional as F
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

ENSEMBLE_TOP3 = ["hybrid_robust", "freq_guided_no_robust", "clip_probe"]


def apply_jpeg(img, q):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
    return Image.open(buf).copy()
def apply_blur(img, s): return img.filter(ImageFilter.GaussianBlur(radius=s))
def apply_resize(img, small):
    w, h = img.size
    return img.resize((small, small), Image.LANCZOS).resize((w, h), Image.LANCZOS)

def deg_clean(im): return im
def deg_jpeg70(im): return apply_jpeg(im, 70)
def deg_jpeg50(im): return apply_jpeg(im, 50)
def deg_jpeg30(im): return apply_jpeg(im, 30)
def deg_blur1(im): return apply_blur(im, 1.0)
def deg_blur2(im): return apply_blur(im, 2.0)
def deg_blur3(im): return apply_blur(im, 3.0)
def deg_resize112(im): return apply_resize(im, 112)

DEGRADATIONS = {
    "clean": deg_clean, "jpeg_q70": deg_jpeg70, "jpeg_q50": deg_jpeg50, "jpeg_q30": deg_jpeg30,
    "blur_s1": deg_blur1, "blur_s2": deg_blur2, "blur_s3": deg_blur3, "resize_112": deg_resize112,
}


class DegradedDataset(Dataset):
    def __init__(self, paths, deg_fn, transform):
        self.paths = paths; self.deg_fn = deg_fn; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        im = self.deg_fn(im)
        rgb = self.transform(im)
        dct = torch.from_numpy(compute_dct_map(im)).unsqueeze(0).float()
        return rgb, dct


def load_clip(config, device):
    m, _, _ = open_clip.create_model_and_transforms(config.clip_model_name, pretrained=config.clip_pretrained)
    enc = m.visual.to(device).eval()
    for p in enc.parameters(): p.requires_grad = False
    return enc


def load_head(name, config, device):
    info = VARIANTS[name]
    kwargs = {"input_dim": config.clip_embed_dim} if info["cls"] is LinearProbeHead else {
        "clip_dim": config.clip_embed_dim, "freq_out_dim": config.freq_branch_out_dim,
        "fusion_hidden": config.fusion_hidden_dim, "fusion_dropout": config.fusion_dropout,
    }
    head = info["cls"](**kwargs)
    ckpt = torch.load(config.checkpoint_dir / info["ckpt"], map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    head.load_state_dict(state)
    return head.to(device).eval()


def collect_test_paths(data_dir, gen, cap):
    paths, labels = [], []
    d = data_dir / "test" / gen
    for ln, li in [("real", 0), ("fake", 1)]:
        cd = d / ln
        if cd.exists():
            ps = sorted(cd.glob("*.jpg"))
            if cap and len(ps) > cap: ps = ps[:cap]
            paths.extend(ps); labels.extend([li] * len(ps))
    return paths, labels


@torch.no_grad()
def extract(paths, deg_fn, enc, transform, device, bs, nw):
    ds = DegradedDataset(paths, deg_fn, transform)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=False)
    feats, dcts = [], []
    for rgb, dct in loader:
        feats.append(enc(rgb.to(device)).cpu())
        dcts.append(dct)
    return torch.cat(feats), torch.cat(dcts)


@torch.no_grad()
def all_probs(heads, temps, feats, dct, device, bs=256):
    """Returns {head_name: np.ndarray of P(fake), shape (N,)}."""
    out = {n: [] for n in heads}
    n = feats.shape[0]
    for i in range(0, n, bs):
        f_b = feats[i:i + bs].to(device)
        d_b = dct[i:i + bs].to(device)
        for name, head in heads.items():
            needs_dct = VARIANTS[name]["needs_dct"]
            logits = head(f_b, d_b) if needs_dct else head(f_b)
            T = max(temps.get(name, 1.0), 1e-3)
            p_fake = F.softmax(logits / T, dim=1)[:, 1].cpu().numpy()
            out[name].append(p_fake)
    return {n: np.concatenate(v) for n, v in out.items()}


def metrics(labels, p_fake):
    preds = (p_fake >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(labels, p_fake))
    except ValueError:
        auc = 0.5
    return {"accuracy": round(float(accuracy_score(labels, preds)), 4), "auc": round(auc, 4)}


def load_val_aucs(config):
    """Read best_val_auc per model from training JSONs. Returns {name: auc}."""
    out = {}
    for name in VARIANTS:
        p = config.results_dir / "metrics" / f"{name}_training.json"
        if p.exists():
            with open(p) as f:
                out[name] = float(json.load(f).get("best_val_auc", 1.0))
        else:
            out[name] = 1.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_per_class", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)
    device = config.device
    gens = args.gens or config.test_generators
    transform = get_eval_transforms()

    print(f"Device: {device} | gens: {gens} | cap: {args.max_per_class}", flush=True)

    # Temperatures (optional)
    calib_path = config.results_dir / "metrics" / "calibration.json"
    temps = {}
    if calib_path.exists():
        with open(calib_path) as f:
            temps = {m: float(v["temperature"]) for m, v in json.load(f).items()}
        print(f"Loaded temperatures: {temps}", flush=True)
    else:
        print("No calibration.json — using T=1.0 for all", flush=True)

    val_aucs = load_val_aucs(config)
    # Weights for weighted ensemble: softmax over val AUC with inverse-temperature 50
    # (sharpens so best model has ~35% weight, 5th model ~10%)
    w = np.array([val_aucs[n] for n in VARIANTS])
    w = np.exp(50 * (w - w.max()))
    w /= w.sum()
    weights = {n: float(w[i]) for i, n in enumerate(VARIANTS)}
    print(f"Weighted-ensemble weights: {weights}", flush=True)

    print("Loading CLIP encoder + 5 heads...", flush=True)
    clip_enc = load_clip(config, device)
    heads = {n: load_head(n, config, device) for n in VARIANTS}

    paths_by_gen = {}
    for g in gens:
        p, l = collect_test_paths(config.data_dir, g, args.max_per_class)
        if p:
            paths_by_gen[g] = (p, np.array(l))

    per_model_cross_gen = {m: {} for m in VARIANTS}
    ensemble_cross_gen = {"ensemble_all": {}, "ensemble_top3": {}, "ensemble_weighted": {}}
    per_model_rob = {m: {} for m in VARIANTS}
    ensemble_rob = {"ensemble_all": {}, "ensemble_top3": {}, "ensemble_weighted": {}}

    t_start = time.time()
    for gen, (paths, labels) in paths_by_gen.items():
        for deg_name, deg_fn in DEGRADATIONS.items():
            t0 = time.time()
            feats, dct = extract(paths, deg_fn, clip_enc, transform, device,
                                 args.batch_size, args.num_workers)
            probs = all_probs(heads, temps, feats, dct, device)
            # Individual model metrics
            for name, p_fake in probs.items():
                m = metrics(labels, p_fake)
                if deg_name == "clean":
                    per_model_cross_gen[name][gen] = {**m, "precision": None, "recall": None, "f1": None}
                per_model_rob[name].setdefault(gen, {})[deg_name] = m

            # Ensembles
            p_all = np.mean(np.stack([probs[n] for n in VARIANTS]), axis=0)
            p_top3 = np.mean(np.stack([probs[n] for n in ENSEMBLE_TOP3]), axis=0)
            p_w = np.sum(np.stack([weights[n] * probs[n] for n in VARIANTS]), axis=0)

            for tag, p in [("ensemble_all", p_all), ("ensemble_top3", p_top3), ("ensemble_weighted", p_w)]:
                m = metrics(labels, p)
                if deg_name == "clean":
                    ensemble_cross_gen[tag][gen] = m
                ensemble_rob[tag].setdefault(gen, {})[deg_name] = m

            # Print a compact row
            best_head = max(probs, key=lambda n: metrics(labels, probs[n])["auc"])
            best_head_auc = metrics(labels, probs[best_head])["auc"]
            ens_aucs = {tag: ensemble_rob[tag][gen][deg_name]["auc"] for tag in ensemble_rob}
            ens_str = "  ".join(f"{k.replace('ensemble_', 'e_')}={v:.3f}" for k, v in ens_aucs.items())
            print(f"  {gen}/{deg_name:10s} [{time.time() - t0:.1f}s] "
                  f"best_single={best_head}={best_head_auc:.3f}  {ens_str}", flush=True)

            del feats, dct, probs
            if device.type == "mps":
                torch.mps.empty_cache()

        # Checkpoint
        metrics_dir = config.results_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "ensemble_cross_gen.json", "w") as f:
            json.dump(ensemble_cross_gen, f, indent=2)
        with open(metrics_dir / "ensemble_robustness.json", "w") as f:
            json.dump(ensemble_rob, f, indent=2)
        print(f"  checkpointed after {gen}", flush=True)

    # Comparison table: best single vs each ensemble
    def avg_clean_auc(d):  # d: {gen: {"auc": ..., "accuracy": ...}}
        aucs = [v["auc"] for v in d.values()]; return float(np.mean(aucs)) if aucs else float("nan")

    def avg_rob_auc(d):  # d: {gen: {deg: {"auc": ...}}}
        vals = []
        for gen_dict in d.values():
            for deg_name, m in gen_dict.items():
                if deg_name != "clean":
                    vals.append(m["auc"])
        return float(np.mean(vals)) if vals else float("nan")

    rows = []
    for name in VARIANTS:
        rows.append((name, avg_clean_auc(per_model_cross_gen[name]), avg_rob_auc(per_model_rob[name])))
    for name in ensemble_cross_gen:
        rows.append((name, avg_clean_auc(ensemble_cross_gen[name]), avg_rob_auc(ensemble_rob[name])))

    # Write comparison table
    tables_dir = config.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ensemble Comparison",
        "",
        "All probabilities are temperature-calibrated from val set.",
        "Clean AUC = mean across 6 generators; Robust AUC = mean over 7 degradations x 6 generators.",
        "",
        "| Model | Clean AUC | Robust AUC |",
        "|-------|-----------|------------|",
    ]
    for name, c_auc, r_auc in rows:
        lines.append(f"| {name} | {c_auc:.4f} | {r_auc:.4f} |")
    out = tables_dir / "ensemble_comparison.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
    print(f"Saved tables/ensemble_comparison.md")


if __name__ == "__main__":
    main()
