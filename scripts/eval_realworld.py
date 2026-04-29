"""
Evaluate the deployed inference pipeline on the curated real-world test set.

Reads ``data/realworld_eval/manifest.csv``, runs each row through every
selected model via ``ModelManager.predict``, and writes per-subset metrics
and per-image predictions to ``results/metrics/realworld_<tag>.json``.

Reports:
    - Per-subset accuracy, false-positive rate (real → AI), three-band breakdown
    - Confusion matrices at p(AI) thresholds 0.5 and 0.85
    - Mean OOD score per subset

Usage:
    python -m scripts.eval_realworld
    python -m scripts.eval_realworld --tag v2
    python -m scripts.eval_realworld --models hybrid_robust freq_guided --tag v2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from backend.inference import ModelManager
from src.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Config().project_root / "data" / "realworld_eval" / "manifest.csv",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["hybrid_robust", "hybrid", "freq_guided", "freq_guided_no_robust", "clip_probe"],
    )
    parser.add_argument("--tag", default="baseline", help="Suffix for output filename")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    rows: list[dict] = []
    with open(args.manifest) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"Loaded {len(rows)} rows from {args.manifest}")

    print("Initialising ModelManager (loads CLIP + heads + OOD centroid)...")
    mgr = ModelManager()

    out: dict = {
        "manifest": str(args.manifest),
        "n_images": len(rows),
        "models": {},
        "subsets": {},
        "tag": args.tag,
        "timestamp": time.time(),
    }

    # Per-image record table: model -> [{subset, label, fake_prob, verdict, band, ood_score}]
    per_model_records: dict[str, list[dict]] = {m: [] for m in args.models}

    config = Config()
    for r in rows:
        path = Path(r["path"])
        if not path.exists():
            print(f"  missing file: {path}")
            continue
        img = Image.open(path)
        for m in args.models:
            try:
                pred = mgr.predict(img, model_name=m)
            except Exception as e:
                print(f"  error on {path}/{m}: {e}")
                continue
            per_model_records[m].append({
                "path": str(path),
                "subset": r["subset"],
                "label": r["label"],
                "fake_prob": pred.get("fake_probability"),
                "verdict": pred.get("verdict"),
                "band": pred.get("band"),
                "ood_score": pred.get("ood_score"),
            })

    # Aggregate metrics per (model, subset)
    for model_name, records in per_model_records.items():
        by_subset: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            by_subset[rec["subset"]].append(rec)

        subsets_summary: dict[str, dict] = {}
        for subset, recs in by_subset.items():
            true_label = recs[0]["label"]  # all rows in a subset share the label
            n = len(recs)
            preds_05 = [int(r["fake_prob"] >= 0.5) for r in recs]
            preds_85 = [int(r["fake_prob"] >= 0.85) for r in recs]
            true_int = 1 if true_label == "fake" else 0
            correct_05 = sum(p == true_int for p in preds_05)
            correct_85 = sum(p == true_int for p in preds_85)
            band_counts = Counter(r["band"] for r in recs)
            mean_ood = float(np.mean([r["ood_score"] or 0.0 for r in recs]))
            mean_fp = float(np.mean([r["fake_prob"] for r in recs]))
            # FPR = % real → flagged AI ; only meaningful for real subsets
            fpr_05 = sum(p == 1 for p in preds_05) / n if true_label == "real" else None
            fpr_85 = sum(p == 1 for p in preds_85) / n if true_label == "real" else None
            subsets_summary[subset] = {
                "n": n,
                "true_label": true_label,
                "acc@0.5": round(correct_05 / n, 4),
                "acc@0.85": round(correct_85 / n, 4),
                "fpr@0.5": round(fpr_05, 4) if fpr_05 is not None else None,
                "fpr@0.85": round(fpr_85, 4) if fpr_85 is not None else None,
                "mean_fake_prob": round(mean_fp, 4),
                "mean_ood_score": round(mean_ood, 4),
                "bands": dict(band_counts),
            }

        # Overall accuracy across all subsets (weight by n)
        all_correct_05 = sum(s["acc@0.5"] * s["n"] for s in subsets_summary.values())
        total_n = sum(s["n"] for s in subsets_summary.values())
        overall_acc = round(all_correct_05 / total_n, 4) if total_n else 0.0
        # Real-only FPR (avg over real subsets, weighted)
        real_fpr_05 = None
        real_n = sum(s["n"] for s in subsets_summary.values() if s["true_label"] == "real")
        if real_n > 0:
            real_fpr_05 = round(sum(
                (s["fpr@0.5"] or 0) * s["n"] for s in subsets_summary.values() if s["true_label"] == "real"
            ) / real_n, 4)

        out["models"][model_name] = {
            "overall_acc@0.5": overall_acc,
            "real_fpr@0.5": real_fpr_05,
            "subsets": subsets_summary,
        }

    # Save records (large) and summary (small)
    out_dir = Config().results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"realworld_{args.tag}.json"
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)

    records_path = out_dir / f"realworld_{args.tag}_records.json"
    with open(records_path, "w") as f:
        json.dump(per_model_records, f, indent=2)

    print(f"\nSaved summary → {summary_path}")
    print(f"Saved records → {records_path}\n")

    # Print compact comparison
    print(f"{'model':<25s} {'overall_acc':>12s} {'real_fpr':>10s}")
    for m, d in out["models"].items():
        fpr = d["real_fpr@0.5"]
        print(f"  {m:<23s} {d['overall_acc@0.5']:>12.4f} {fpr:>10.4f}" if fpr is not None
              else f"  {m:<23s} {d['overall_acc@0.5']:>12.4f} {'n/a':>10s}")


if __name__ == "__main__":
    main()
