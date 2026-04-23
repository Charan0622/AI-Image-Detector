"""
Generate all result plots from metrics JSONs.

Plots produced (all saved to results/plots/):
    training_curves.png        — train/val loss, acc, AUC per model
    cross_gen_heatmap.png      — accuracy heatmap: models x generators
    cross_gen_bars.png         — grouped bar chart (same data)
    ablation_summary.png       — clean vs cross-gen vs robustness summary
    robustness_{model}.png     — per-model: acc vs degradation severity
    robustness_all.png         — all models overlaid per degradation family

Only reads existing JSONs; no model inference. Safe to re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import Config


MODELS = ["clip_probe", "hybrid", "hybrid_robust", "freq_guided_no_robust", "freq_guided"]
ENSEMBLES = ["ensemble_all", "ensemble_top3", "ensemble_weighted"]
PRETTY = {
    "clip_probe": "CLIP Probe",
    "hybrid": "Hybrid",
    "hybrid_robust": "Hybrid+Robust",
    "freq_guided_no_robust": "FreqGuided (no robust)",
    "freq_guided": "FreqGuided (full)",
    "ensemble_all": "Ensemble (all 5)",
    "ensemble_top3": "Ensemble (top 3)",
    "ensemble_weighted": "Ensemble (weighted)",
}
COLORS = {
    "clip_probe": "#6b7280",
    "hybrid": "#3b82f6",
    "hybrid_robust": "#8b5cf6",
    "freq_guided_no_robust": "#f59e0b",
    "freq_guided": "#10b981",
    "ensemble_all": "#ec4899",
    "ensemble_top3": "#14b8a6",
    "ensemble_weighted": "#f43f5e",
}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_training_curves(metrics_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    titles = ["Train Loss", "Validation Accuracy", "Validation AUC"]
    keys = ["train_loss", "val_acc", "val_auc"]

    for ax, title, key in zip(axes, titles, keys):
        for m in MODELS:
            data = load_json(metrics_dir / f"{m}_training.json")
            if data is None or "history" not in data:
                continue
            epochs = [h["epoch"] for h in data["history"]]
            values = [h.get(key) for h in data["history"]]
            if any(v is None for v in values):
                continue
            ax.plot(
                epochs, values,
                label=PRETTY[m], color=COLORS[m],
                linewidth=2, marker="o", markersize=3,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_ylabel("AUC")
    axes[-1].legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path.name}")


def plot_cross_gen(metrics_dir: Path, out_heatmap: Path, out_bars: Path) -> None:
    data = {m: load_json(metrics_dir / f"{m}_cross_gen.json") for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    if not data:
        return

    generators = sorted({g for d in data.values() for g in d.keys()})
    matrix = np.full((len(data), len(generators)), np.nan)
    for i, m in enumerate(data):
        for j, g in enumerate(generators):
            if g in data[m]:
                matrix[i, j] = data[m][g]["accuracy"]

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(generators) + 3), 0.6 * len(data) + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(generators)))
    ax.set_xticklabels(generators, rotation=30, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([PRETTY[m] for m in data.keys()])
    for i in range(len(data)):
        for j in range(len(generators)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        color="black" if matrix[i, j] > 0.75 else "white", fontsize=9)
    ax.set_title("Cross-Generator Accuracy", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    plt.savefig(out_heatmap, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_heatmap.name}")

    # --- Grouped bar chart ---
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(generators) + 2), 5))
    x = np.arange(len(generators))
    w = 0.8 / len(data)
    for i, m in enumerate(data.keys()):
        vals = [data[m].get(g, {}).get("accuracy", np.nan) for g in generators]
        offset = (i - len(data) / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=PRETTY[m], color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(generators)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Cross-Generator Accuracy by Model", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_bars, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_bars.name}")


DEG_ORDER = ["clean", "jpeg_q70", "jpeg_q50", "jpeg_q30",
             "blur_s1", "blur_s2", "blur_s3", "resize_112"]
DEG_PRETTY = {
    "clean": "Clean",
    "jpeg_q70": "JPEG Q=70", "jpeg_q50": "JPEG Q=50", "jpeg_q30": "JPEG Q=30",
    "blur_s1": "Blur σ=1", "blur_s2": "Blur σ=2", "blur_s3": "Blur σ=3",
    "resize_112": "Resize 112",
}


def _avg_over_gens(rob_data: dict, deg: str, metric: str = "auc") -> float:
    vals = [r[deg][metric] for r in rob_data.values() if deg in r]
    return float(np.mean(vals)) if vals else np.nan


def plot_robustness(metrics_dir: Path, plots_dir: Path) -> None:
    rob = {m: load_json(metrics_dir / f"{m}_robustness.json") for m in MODELS}
    rob = {m: d for m, d in rob.items() if d}
    # Merge ensembles from ensemble_robustness.json
    ens = load_json(metrics_dir / "ensemble_robustness.json")
    if ens:
        for e_name, per_gen in ens.items():
            rob[e_name] = per_gen
    if not rob:
        print("  (no robustness JSONs yet — skipping robustness plots)")
        return

    # Per-model line plot (only for the base MODELS, not ensembles)
    for m, data in rob.items():
        if m not in MODELS:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        gens = sorted(data.keys())
        for g in gens:
            accs = [data[g][d]["accuracy"] if d in data[g] else np.nan for d in DEG_ORDER]
            ax.plot(DEG_ORDER, accs, marker="o", label=g, linewidth=1.6)
        # Avg line
        avg = [_avg_over_gens(data, d, "accuracy") for d in DEG_ORDER]
        ax.plot(DEG_ORDER, avg, marker="s", label="avg", linewidth=2.5, color="black", linestyle="--")
        ax.set_title(f"{PRETTY[m]} — Robustness (Accuracy vs Degradation)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, loc="lower left")
        plt.tight_layout()
        out = plots_dir / f"robustness_{m}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> {out.name}")

    # Side-by-side: all models, average AUC per degradation (single-model view)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for metric, ax in zip(["accuracy", "auc"], axes):
        for m, data in rob.items():
            if m not in MODELS:
                continue
            vals = [_avg_over_gens(data, d, metric) for d in DEG_ORDER]
            ax.plot(DEG_ORDER, vals, marker="o", label=PRETTY[m], color=COLORS[m], linewidth=2)
        ax.set_title(f"Average {metric.upper()} across generators", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    out = plots_dir / "robustness_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.name}")

    # Ensemble-vs-single comparison plot (AUC only, ensembles included)
    has_ens = any(k.startswith("ensemble_") for k in rob)
    if has_ens:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        # Highlight best single model and ensembles only (skip the bottom 3 singles)
        focus = ["clip_probe", "hybrid_robust", "freq_guided_no_robust"] + ENSEMBLES
        for m in focus:
            if m not in rob:
                continue
            data = rob[m]
            vals = [_avg_over_gens(data, d, "auc") for d in DEG_ORDER]
            ls = "--" if m.startswith("ensemble_") else "-"
            ax.plot(DEG_ORDER, vals, marker="o", label=PRETTY[m], color=COLORS[m],
                    linewidth=2.2, linestyle=ls)
        ax.set_title("Ensembles vs Best Singles — AUC across degradations",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.55, 1.02)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9, loc="lower left")
        plt.tight_layout()
        out = plots_dir / "robustness_ensembles.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> {out.name}")


def plot_ablation_summary(metrics_dir: Path, out_path: Path) -> None:
    cross = {m: load_json(metrics_dir / f"{m}_cross_gen.json") for m in MODELS}
    rob = {m: load_json(metrics_dir / f"{m}_robustness.json") for m in MODELS}

    rows = []
    for m in MODELS:
        c = cross.get(m)
        r = rob.get(m)
        clean_acc = float(np.mean([v["accuracy"] for v in c.values()])) if c else np.nan
        clean_auc = float(np.mean([v["auc"] for v in c.values()])) if c else np.nan
        if r:
            rob_degs = [d for d in DEG_ORDER if d != "clean"]
            rob_auc = float(np.mean([_avg_over_gens(r, d, "auc") for d in rob_degs]))
        else:
            rob_auc = np.nan
        rows.append((m, clean_acc, clean_auc, rob_auc))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    w = 0.25
    accs = [r[1] for r in rows]
    aucs = [r[2] for r in rows]
    rob_aucs = [r[3] for r in rows]
    ax.bar(x - w, accs, w, label="Clean Acc", color="#3b82f6")
    ax.bar(x, aucs, w, label="Clean AUC", color="#10b981")
    ax.bar(x + w, rob_aucs, w, label="Robust AUC (avg)", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m] for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Ablation Summary", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path.name}")


def main() -> None:
    config = Config()
    metrics_dir = config.results_dir / "metrics"
    plots_dir = config.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Generating training curves...")
    plot_training_curves(metrics_dir, plots_dir / "training_curves.png")

    print("Generating cross-generator plots...")
    plot_cross_gen(metrics_dir, plots_dir / "cross_gen_heatmap.png", plots_dir / "cross_gen_bars.png")

    print("Generating robustness plots...")
    plot_robustness(metrics_dir, plots_dir)

    print("Generating ablation summary...")
    plot_ablation_summary(metrics_dir, plots_dir / "ablation_summary.png")

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
