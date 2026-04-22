"""
Regenerate results/tables/ablation_table.md with clean AND robustness metrics.

Reads:
    results/metrics/{model}_cross_gen.json   (required)
    results/metrics/{model}_robustness.json  (optional — skipped if missing)

Writes:
    results/tables/ablation_table.md         (markdown, model x metric)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import Config


MODELS_ORDERED = [
    "clip_probe",
    "hybrid",
    "hybrid_robust",
    "freq_guided_no_robust",
    "freq_guided",
]
PRETTY = {
    "clip_probe": "CLIP Linear Probe",
    "hybrid": "AIDE-style Hybrid",
    "hybrid_robust": "Hybrid + Robust Aug",
    "freq_guided_no_robust": "FreqGuided (no robust)",
    "freq_guided": "FreqGuided (full, our final)",
}
DEG_ORDER = ["jpeg_q70", "jpeg_q50", "jpeg_q30", "blur_s1", "blur_s2", "blur_s3", "resize_112"]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _cross_avg(cross: dict | None, key: str) -> float:
    if not cross:
        return float("nan")
    return float(np.mean([v[key] for v in cross.values()]))


def _rob_deg_avg(rob: dict | None, deg: str, metric: str) -> float:
    if not rob:
        return float("nan")
    vals = [r[deg][metric] for r in rob.values() if deg in r]
    return float(np.mean(vals)) if vals else float("nan")


def _rob_summary(rob: dict | None, metric: str) -> float:
    if not rob:
        return float("nan")
    vals = [_rob_deg_avg(rob, d, metric) for d in DEG_ORDER]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _fmt(x: float) -> str:
    return "—" if np.isnan(x) else f"{x:.4f}"


def main() -> None:
    config = Config()
    metrics_dir = config.results_dir / "metrics"
    tables_dir = config.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in MODELS_ORDERED:
        cross = _load(metrics_dir / f"{m}_cross_gen.json")
        rob = _load(metrics_dir / f"{m}_robustness.json")
        rows.append({
            "model": m,
            "clean_acc": _cross_avg(cross, "accuracy"),
            "clean_auc": _cross_avg(cross, "auc"),
            "jpeg30_acc": _rob_deg_avg(rob, "jpeg_q30", "accuracy"),
            "blur3_acc": _rob_deg_avg(rob, "blur_s3", "accuracy"),
            "resize_acc": _rob_deg_avg(rob, "resize_112", "accuracy"),
            "rob_auc": _rob_summary(rob, "auc"),
            "rob_acc": _rob_summary(rob, "accuracy"),
        })

    # Base = clip_probe clean acc, for delta computation
    base_cross = rows[0]["clean_auc"]

    lines: list[str] = []
    lines.append("# Ablation Study Results\n")
    lines.append(
        "Cross-gen = averaged across the 6 unseen test generators (adm, glide, "
        "midjourney, sd15, vqdm, wukong). Robust = averaged over 7 degradations "
        "(JPEG Q=70/50/30, Blur σ=1/2/3, Resize 112) × the same generators.\n"
    )

    header = (
        "| # | Model | Clean Acc | Clean AUC | JPEG-30 Acc | Blur-σ3 Acc | "
        "Resize Acc | Robust AUC (avg) | Robust Acc (avg) | Δ AUC vs Probe |"
    )
    sep = (
        "|---|-------|-----------|-----------|-------------|-------------|"
        "------------|------------------|------------------|----------------|"
    )
    lines.append(header)
    lines.append(sep)

    for i, r in enumerate(rows, 1):
        delta = r["clean_auc"] - base_cross
        delta_s = "—" if np.isnan(delta) else f"{delta:+.4f}"
        lines.append(
            f"| {i} | {PRETTY[r['model']]} | {_fmt(r['clean_acc'])} | "
            f"{_fmt(r['clean_auc'])} | {_fmt(r['jpeg30_acc'])} | "
            f"{_fmt(r['blur3_acc'])} | {_fmt(r['resize_acc'])} | "
            f"{_fmt(r['rob_auc'])} | {_fmt(r['rob_acc'])} | {delta_s} |"
        )

    # Degradation-only breakdown table (per model, per degradation)
    lines.append("\n## Robustness Breakdown (AUC averaged across generators)\n")
    hdr = "| Model | Clean | JPEG-70 | JPEG-50 | JPEG-30 | Blur-1 | Blur-2 | Blur-3 | Resize-112 |"
    sp = "|-------|-------|---------|---------|---------|--------|--------|--------|------------|"
    lines.append(hdr)
    lines.append(sp)
    for r in rows:
        rob = _load(metrics_dir / f"{r['model']}_robustness.json")
        if not rob:
            continue
        clean = _rob_deg_avg(rob, "clean", "auc")
        cells = [_fmt(clean)] + [_fmt(_rob_deg_avg(rob, d, "auc")) for d in DEG_ORDER]
        lines.append(f"| {PRETTY[r['model']]} | " + " | ".join(cells) + " |")

    out = tables_dir / "ablation_table.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
