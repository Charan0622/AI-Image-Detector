"""
Test-Time Augmentation (TTA) evaluation.

TTA strategy: horizontal flip. For each image we average the logits of the
original and horizontally-flipped versions. The class of "real vs AI" should
be near-invariant to mirror, so this should reduce variance without bias.

We evaluate one model (default: hybrid_robust, the best single model per the
ablation) across all 6 test generators on (clean, blur_s3, jpeg_q30,
resize_112). This is a cheap subset that captures clean + the three hardest
degradations.

Compares TTA vs no-TTA for the same model using the same (image, degradation)
tuples, so any delta is pure TTA effect.

Outputs:
    results/metrics/tta_{model}.json
    results/tables/tta_comparison.md

Usage:
    python -m scripts.run_tta_eval
    python -m scripts.run_tta_eval --model freq_guided_no_robust --gens sd15 midjourney
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
import torchvision.transforms.functional as TF
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

# Focused degradation set: clean + 3 worst cases
DEG_SUBSET = ["clean", "jpeg_q30", "blur_s3", "resize_112"]


def apply_jpeg(img, q):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
    return Image.open(buf).copy()
def apply_blur(img, s): return img.filter(ImageFilter.GaussianBlur(radius=s))
def apply_resize(img, small):
    w, h = img.size
    return img.resize((small, small), Image.LANCZOS).resize((w, h), Image.LANCZOS)


def deg_clean(im): return im
def deg_jpeg30(im): return apply_jpeg(im, 30)
def deg_blur3(im): return apply_blur(im, 3.0)
def deg_resize112(im): return apply_resize(im, 112)


DEG_FNS = {
    "clean": deg_clean,
    "jpeg_q30": deg_jpeg30,
    "blur_s3": deg_blur3,
    "resize_112": deg_resize112,
}


class TTADataset(Dataset):
    """Emits (rgb, rgb_flipped, dct, dct_flipped) for each image."""
    def __init__(self, paths, deg_fn, transform):
        self.paths = paths; self.deg_fn = deg_fn; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        im = self.deg_fn(im)
        im_flip = TF.hflip(im)
        rgb = self.transform(im)
        rgb_f = self.transform(im_flip)
        dct = torch.from_numpy(compute_dct_map(im)).unsqueeze(0).float()
        dct_f = torch.from_numpy(compute_dct_map(im_flip)).unsqueeze(0).float()
        return rgb, rgb_f, dct, dct_f


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


def metrics(labels, p_fake):
    preds = (p_fake >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(labels, p_fake))
    except ValueError:
        auc = 0.5
    return {"accuracy": round(float(accuracy_score(labels, preds)), 4), "auc": round(auc, 4)}


@torch.no_grad()
def eval_with_tta(head, needs_dct, temp, paths, labels, deg_fn, clip_enc, transform,
                  device, bs, nw):
    """Returns (no_tta_metrics, tta_metrics). TTA = mean logit of (orig, hflip)."""
    ds = TTADataset(paths, deg_fn, transform)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=False)

    p_no_tta = []
    p_tta = []
    T = max(temp, 1e-3)
    for rgb, rgb_f, dct, dct_f in loader:
        rgb = rgb.to(device); rgb_f = rgb_f.to(device)
        dct = dct.to(device); dct_f = dct_f.to(device)
        f_o = clip_enc(rgb)
        f_f = clip_enc(rgb_f)
        if needs_dct:
            l_o = head(f_o, dct)
            l_f = head(f_f, dct_f)
        else:
            l_o = head(f_o)
            l_f = head(f_f)
        # No TTA: use orig only
        p_no_tta.append(F.softmax(l_o / T, dim=1)[:, 1].cpu().numpy())
        # TTA: average the two logit tensors
        l_tta = 0.5 * (l_o + l_f)
        p_tta.append(F.softmax(l_tta / T, dim=1)[:, 1].cpu().numpy())

    p_no = np.concatenate(p_no_tta); p_tt = np.concatenate(p_tta)
    return metrics(labels, p_no), metrics(labels, p_tt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["hybrid_robust", "freq_guided_no_robust"],
                        choices=list(VARIANTS.keys()))
    parser.add_argument("--gens", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=48,
                        help="Smaller than usual — we do 2x CLIP per image.")
    parser.add_argument("--max_per_class", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)
    device = config.device
    gens = args.gens or config.test_generators
    transform = get_eval_transforms()

    # Load temperatures
    calib_path = config.results_dir / "metrics" / "calibration.json"
    temps = {}
    if calib_path.exists():
        with open(calib_path) as f:
            temps = {m: float(v["temperature"]) for m, v in json.load(f).items()}

    print(f"Device: {device} | models: {args.models} | gens: {gens}", flush=True)
    print("Loading CLIP encoder + heads...", flush=True)
    clip_enc = load_clip(config, device)

    paths_by_gen = {}
    for g in gens:
        p, l = collect_test_paths(config.data_dir, g, args.max_per_class)
        if p: paths_by_gen[g] = (p, np.array(l))

    # results[model][gen][deg] = {"no_tta": {...}, "tta": {...}, "delta_auc": ...}
    results = {m: {} for m in args.models}
    t0 = time.time()
    for model_name in args.models:
        info = VARIANTS[model_name]
        head = load_head(model_name, config, device)
        T = temps.get(model_name, 1.0)
        print(f"\n=== {model_name} (T={T:.4f}) ===", flush=True)

        for gen, (paths, labels) in paths_by_gen.items():
            results[model_name][gen] = {}
            for deg_name in DEG_SUBSET:
                deg_fn = DEG_FNS[deg_name]
                dt0 = time.time()
                no_tta, tta = eval_with_tta(
                    head, info["needs_dct"], T, paths, labels, deg_fn, clip_enc, transform,
                    device, args.batch_size, args.num_workers,
                )
                delta = round(tta["auc"] - no_tta["auc"], 4)
                results[model_name][gen][deg_name] = {
                    "no_tta": no_tta, "tta": tta, "delta_auc": delta,
                }
                sign = "+" if delta >= 0 else ""
                print(f"  {gen}/{deg_name:10s} [{time.time() - dt0:.1f}s]  "
                      f"no_tta={no_tta['auc']:.3f}  tta={tta['auc']:.3f}  Δ={sign}{delta:+.4f}",
                      flush=True)

        # Save after each model
        out_path = config.results_dir / "metrics" / f"tta_{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(results[model_name], f, indent=2)
        print(f"  saved -> {out_path}", flush=True)

        del head
        if device.type == "mps":
            torch.mps.empty_cache()

    # Summary table
    lines = [
        "# Test-Time Augmentation Results",
        "",
        "TTA = mean logit of (original, horizontal-flip). Evaluated on 4 conditions:",
        "clean + 3 hardest degradations (JPEG-30, Blur-σ3, Resize-112) across 6 generators.",
        "",
        "| Model | Condition | No-TTA AUC (avg) | TTA AUC (avg) | Δ AUC |",
        "|-------|-----------|------------------|---------------|-------|",
    ]
    for model_name in args.models:
        for deg_name in DEG_SUBSET:
            no_ttas = [results[model_name][g][deg_name]["no_tta"]["auc"]
                       for g in results[model_name]]
            ttas = [results[model_name][g][deg_name]["tta"]["auc"]
                    for g in results[model_name]]
            mn = np.mean(no_ttas); mt = np.mean(ttas); d = mt - mn
            lines.append(f"| {model_name} | {deg_name} | {mn:.4f} | {mt:.4f} | {d:+.4f} |")
        # Overall
        all_no = [v["no_tta"]["auc"] for g in results[model_name].values() for v in g.values()]
        all_tt = [v["tta"]["auc"] for g in results[model_name].values() for v in g.values()]
        mn = np.mean(all_no); mt = np.mean(all_tt); d = mt - mn
        lines.append(f"| **{model_name}** | **overall** | **{mn:.4f}** | **{mt:.4f}** | **{d:+.4f}** |")

    tables_dir = config.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "tta_comparison.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
