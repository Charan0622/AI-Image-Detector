"""Generate a baseline-vs-v2 comparison plot for the real-world eval set."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import Config


def main() -> None:
    metrics_dir = Config().results_dir / "metrics"
    plots_dir = Config().results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "realworld_baseline.json") as f:
        baseline = json.load(f)
    with open(metrics_dir / "realworld_v2.json") as f:
        v2 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: overall accuracy + real FPR per model
    models = ["clip_probe", "hybrid", "hybrid_robust"]
    extras = ["hybrid_robust_v2"] if "hybrid_robust_v2" in v2["models"] else []
    all_models = models + extras

    accs = []
    fprs = []
    for m in all_models:
        # Use baseline number if exists, else v2 (for the new v2 row)
        src = v2["models"] if m in v2["models"] else baseline["models"]
        if m in src:
            accs.append(src[m]["overall_acc@0.5"])
            fprs.append(src[m]["real_fpr@0.5"] or 0.0)
        else:
            accs.append(0); fprs.append(0)

    x = np.arange(len(all_models))
    w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, accs, w, label="overall acc", color="#9ec27e")
    ax.bar(x + w/2, fprs, w, label="real FPR", color="#d88280")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in all_models], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Real-world test set (117 images)\noverall accuracy and real-class FPR")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    # Right: side-by-side band breakdown for hybrid_robust v1 vs v2 on real_picsum
    ax = axes[1]
    real_v1 = baseline["models"]["hybrid_robust"]["subsets"]["real_picsum"]["bands"]
    real_v2 = v2["models"]["hybrid_robust_v2"]["subsets"]["real_picsum"]["bands"]
    keys = sorted(set(real_v1) | set(real_v2),
                  key=lambda k: ["real", "likely_real", "uncertain", "fake"].index(k)
                  if k in ["real", "likely_real", "uncertain", "fake"] else 99)
    band_colors = {"real": "#9ec27e", "likely_real": "#bcd9a3",
                   "uncertain": "#d9c97a", "fake": "#d88280"}
    v1_vals = [real_v1.get(k, 0) for k in keys]
    v2_vals = [real_v2.get(k, 0) for k in keys]
    x2 = np.arange(len(keys))
    ax.bar(x2 - w/2, v1_vals, w, label="hybrid_robust (v1)", color="#7a7a7a")
    ax.bar(x2 + w/2, v2_vals, w, label="hybrid_robust_v2",
           color=[band_colors.get(k, "#aaa") for k in keys])
    ax.set_xticks(x2)
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_title("Band assignment on 100 real picsum photos")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = plots_dir / "realworld_improvement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
