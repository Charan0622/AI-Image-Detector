"""
Fit temperature scaling (Guo et al., ICML 2017) on the validation set.

For each trained head we find a scalar T > 0 such that softmax(logits / T)
is best-calibrated under NLL. Minimizes expected calibration error (ECE)
too, as long as NLL goes down.

Uses the already-cached `val_features.npy` (CLIP features) and recomputes
DCT maps on-the-fly from the same filesystem order.

Outputs:
    results/metrics/calibration.json with per-model T and before/after NLL + ECE.

Usage:
    python -m scripts.fit_temperature
    python -m scripts.fit_temperature --max_samples 3000 --models hybrid_robust freq_guided
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import Config
from src.dataset import AIDetectDataset
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
    "hybrid_robust_v2": {"cls": HybridRobustFromFeatures, "ckpt": "hybrid_robust_v2_best.pth", "needs_dct": True},
    "hybrid_robust_v3": {"cls": HybridRobustFromFeatures, "ckpt": "hybrid_robust_v3_best.pth", "needs_dct": True},
    "freq_guided_no_robust": {"cls": FreqGuidedFromFeatures, "ckpt": "freq_guided_no_robust_best.pth", "needs_dct": True},
    "freq_guided": {"cls": FreqGuidedFromFeatures, "ckpt": "freq_guided_best.pth", "needs_dct": True},
}


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
        raise FileNotFoundError(str(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    head.load_state_dict(state)
    return head.to(device).eval()


class ValDCTDataset(Dataset):
    """Replays AIDetectDataset val ordering to emit DCT maps in sync with cached features."""

    def __init__(self, data_dir: Path) -> None:
        ds = AIDetectDataset(data_dir, split="val", transform=get_eval_transforms())
        self.samples = ds.samples  # list of (path, label), same order as val_features.npy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        img = Image.open(self.samples[idx][0]).convert("RGB")
        dct = compute_dct_map(img)
        return torch.from_numpy(dct).unsqueeze(0).float()


@torch.no_grad()
def collect_logits(
    head: nn.Module,
    feats: torch.Tensor,
    dct_loader: DataLoader | None,
    needs_dct: bool,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Returns logits tensor (N, 2) on CPU."""
    out: list[torch.Tensor] = []
    n = feats.shape[0]

    if needs_dct:
        # Iterate dct_loader in lockstep with feats
        dct_iter = iter(dct_loader)
        idx = 0
        for dct_batch in tqdm(dct_loader, desc="  logits (w/ dct)", total=len(dct_loader), leave=False):
            b = dct_batch.shape[0]
            f_b = feats[idx : idx + b].to(device)
            d_b = dct_batch.to(device)
            logits = head(f_b, d_b).cpu()
            out.append(logits)
            idx += b
    else:
        for i in tqdm(range(0, n, batch_size), desc="  logits", leave=False):
            f_b = feats[i : i + batch_size].to(device)
            logits = head(f_b).cpu()
            out.append(logits)

    return torch.cat(out)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a scalar T > 0 that minimizes NLL of softmax(logits / T)."""
    T = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Compute expected calibration error on binary task.

    probs: (N, 2) softmax probabilities.
    labels: (N,) int64.
    """
    confidences, preds = probs.max(dim=1)
    accuracies = preds.eq(labels).float()

    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = labels.shape[0]
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        acc = accuracies[in_bin].mean().item()
        conf = confidences[in_bin].mean().item()
        ece += (in_bin.sum().item() / n) * abs(acc - conf)
    return ece


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()))
    parser.add_argument("--max_samples", type=int, default=4000,
                        help="Cap val samples for calibration. 0 = all.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)
    device = config.device

    # Load cached val features
    feat_dir = config.project_root / "data" / "features"
    print(f"Loading {feat_dir}/val_features.npy ...")
    feats_np = np.load(feat_dir / "val_features.npy").astype(np.float32)
    labels_np = np.load(feat_dir / "val_labels.npy").astype(np.int64)
    print(f"  feats: {feats_np.shape}  labels: {labels_np.shape}")

    # Subsample preserving class balance
    if args.max_samples and args.max_samples < len(labels_np):
        per_class = args.max_samples // 2
        idx_real = np.where(labels_np == 0)[0][:per_class]
        idx_fake = np.where(labels_np == 1)[0][:per_class]
        idx = np.concatenate([idx_real, idx_fake])
        feats_np = feats_np[idx]
        labels_np = labels_np[idx]
        print(f"  subsampled to {len(idx)} ({per_class}/class)")

    feats = torch.from_numpy(feats_np)
    labels = torch.from_numpy(labels_np)

    # If any model needs DCT, build a DCT loader aligned with the cached feature order.
    need_dct_any = any(VARIANTS[m]["needs_dct"] for m in args.models)
    dct_loader = None
    if need_dct_any:
        full_ds = ValDCTDataset(config.data_dir)
        # Select the same indices
        if args.max_samples and args.max_samples < len(full_ds):
            per_class = args.max_samples // 2
            idx_real_full = [i for i, (_, l) in enumerate(full_ds.samples) if l == 0][:per_class]
            idx_fake_full = [i for i, (_, l) in enumerate(full_ds.samples) if l == 1][:per_class]
            sub_idx = idx_real_full + idx_fake_full
            full_ds = torch.utils.data.Subset(full_ds, sub_idx)
        dct_loader = DataLoader(
            full_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        print(f"  dct dataset: {len(full_ds)}")

    results: dict = {}
    for model_name in args.models:
        info = VARIANTS[model_name]
        print(f"\n=== {model_name} ===")
        head = load_head(model_name, config, device)
        logits = collect_logits(
            head, feats, dct_loader, info["needs_dct"],
            device, args.batch_size,
        )

        # Metrics before
        probs_before = F.softmax(logits, dim=1)
        nll_before = F.cross_entropy(logits, labels).item()
        ece_before = expected_calibration_error(probs_before, labels)

        # Fit T
        T = fit_temperature(logits, labels)
        scaled = logits / T
        probs_after = F.softmax(scaled, dim=1)
        nll_after = F.cross_entropy(scaled, labels).item()
        ece_after = expected_calibration_error(probs_after, labels)

        results[model_name] = {
            "temperature": round(T, 4),
            "nll_before": round(nll_before, 4),
            "nll_after": round(nll_after, 4),
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
            "n_samples": int(len(labels)),
        }
        print(
            f"  T={T:.4f}  NLL {nll_before:.4f} -> {nll_after:.4f}  "
            f"ECE {ece_before:.4f} -> {ece_after:.4f}"
        )
        del head
        if device.type == "mps":
            torch.mps.empty_cache()

    out_path = config.results_dir / "metrics" / "calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
