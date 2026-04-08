"""
Cross-generator evaluation suite.

For each test generator:
    - Accuracy, AUC, Precision, Recall, F1
    - Confusion matrix

Also evaluates robustness:
    - JPEG compression at Q=70, 50, 30
    - Gaussian blur at sigma=1, 2, 3
    - Resize 112->224

Outputs:
    - results/metrics/{model_name}_cross_gen.json
    - results/metrics/{model_name}_robustness.json
    - results/tables/{model_name}_results.md
    - results/plots/{model_name}_roc_curves.png

Usage:
    python -m src.evaluate --model clip_probe
    python -m src.evaluate --model hybrid
    python -m src.evaluate --model freq_guided
"""

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.dataset import AIDetectDCTDataset, AIDetectTestDataset
from src.models.model_zoo import get_model
from src.seed import fix_seeds
from src.transforms import get_eval_transforms


@torch.no_grad()
def evaluate_generator(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    needs_dct: bool = False,
) -> dict:
    """Evaluate model on a single generator's test set.

    Args:
        model: Trained model.
        loader: Test data loader for one generator.
        device: Device.
        needs_dct: Whether model needs DCT input.

    Returns:
        Dictionary with accuracy, AUC, precision, recall, F1.
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        if needs_dct:
            dct_maps = batch["dct_map"].to(device)
            logits = model(images, dct_maps)
        else:
            logits = model(images)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "auc": round(auc, 4),
        "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "f1": round(f1_score(all_labels, all_preds, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "n_samples": len(all_labels),
        "labels": all_labels.tolist(),
        "probs": all_probs.tolist(),
    }


def apply_jpeg_degradation(img_pil: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression to a PIL image.

    Args:
        img_pil: Input image.
        quality: JPEG quality (1-100).

    Returns:
        Degraded image.
    """
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_blur_degradation(img_pil: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur to a PIL image.

    Args:
        img_pil: Input image.
        sigma: Blur sigma.

    Returns:
        Degraded image.
    """
    return img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_resize_degradation(img_pil: Image.Image, small_size: int) -> Image.Image:
    """Downscale then upscale to simulate social media processing.

    Args:
        img_pil: Input image.
        small_size: Intermediate small size.

    Returns:
        Degraded image.
    """
    w, h = img_pil.size
    small = img_pil.resize((small_size, small_size), Image.LANCZOS)
    return small.resize((w, h), Image.LANCZOS)


@torch.no_grad()
def evaluate_robustness(
    model: nn.Module,
    data_dir: Path,
    generator: str,
    device: torch.device,
    needs_dct: bool = False,
    batch_size: int = 32,
) -> dict:
    """Evaluate model robustness to image degradations.

    Args:
        model: Trained model.
        data_dir: Path to processed data directory.
        generator: Generator name for test data.
        device: Device.
        needs_dct: Whether model needs DCT input.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with robustness metrics per degradation type.
    """
    from src.transforms import compute_dct_map

    model.eval()
    transform = get_eval_transforms()

    degradations = {
        "clean": lambda img: img,
        "jpeg_q70": lambda img: apply_jpeg_degradation(img, 70),
        "jpeg_q50": lambda img: apply_jpeg_degradation(img, 50),
        "jpeg_q30": lambda img: apply_jpeg_degradation(img, 30),
        "blur_s1": lambda img: apply_blur_degradation(img, 1.0),
        "blur_s2": lambda img: apply_blur_degradation(img, 2.0),
        "blur_s3": lambda img: apply_blur_degradation(img, 3.0),
        "resize_112": lambda img: apply_resize_degradation(img, 112),
    }

    results: dict = {}

    # Load test images as PIL
    gen_dir = data_dir / "test" / generator
    image_paths = []
    labels = []
    for label_name, label_int in [("real", 0), ("fake", 1)]:
        label_dir = gen_dir / label_name
        if label_dir.exists():
            for p in sorted(label_dir.glob("*.jpg")):
                image_paths.append(p)
                labels.append(label_int)

    for deg_name, deg_fn in degradations.items():
        all_preds = []
        all_probs = []

        for i in tqdm(
            range(0, len(image_paths), batch_size),
            desc=f"  {deg_name}",
            leave=False,
        ):
            batch_paths = image_paths[i : i + batch_size]
            batch_imgs = []
            batch_dcts = []

            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                img = deg_fn(img)

                if needs_dct:
                    dct = compute_dct_map(img)
                    dct_tensor = torch.from_numpy(dct).unsqueeze(0).float()
                    batch_dcts.append(dct_tensor)

                batch_imgs.append(transform(img))

            batch_tensor = torch.stack(batch_imgs).to(device)

            if needs_dct:
                dct_tensor = torch.stack(batch_dcts).to(device)
                logits = model(batch_tensor, dct_tensor)
            else:
                logits = model(batch_tensor)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)

        labels_arr = np.array(labels)
        preds_arr = np.array(all_preds)
        probs_arr = np.array(all_probs)

        try:
            auc = roc_auc_score(labels_arr, probs_arr)
        except ValueError:
            auc = 0.5

        results[deg_name] = {
            "accuracy": round(accuracy_score(labels_arr, preds_arr), 4),
            "auc": round(auc, 4),
        }

    return results


def generate_results_table(
    cross_gen_results: dict, model_name: str, output_path: Path
) -> str:
    """Generate a markdown results table.

    Args:
        cross_gen_results: Per-generator evaluation results.
        model_name: Name of the model.
        output_path: Path to save the markdown table.

    Returns:
        Markdown table string.
    """
    header = "| Generator | Accuracy | AUC | Precision | Recall | F1 |"
    separator = "|-----------|----------|-----|-----------|--------|-----|"
    rows = [f"# {model_name} — Cross-Generator Results\n", header, separator]

    accs = []
    aucs = []
    for gen, metrics in sorted(cross_gen_results.items()):
        row = (
            f"| {gen} | {metrics['accuracy']:.4f} | {metrics['auc']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |"
        )
        rows.append(row)
        accs.append(metrics["accuracy"])
        aucs.append(metrics["auc"])

    avg_row = f"| **Average** | **{np.mean(accs):.4f}** | **{np.mean(aucs):.4f}** | | | |"
    rows.append(avg_row)

    table = "\n".join(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table)

    return table


def plot_roc_curves(
    cross_gen_results: dict, model_name: str, output_path: Path
) -> None:
    """Plot ROC curves for each generator.

    Args:
        cross_gen_results: Per-generator evaluation results (with labels/probs).
        model_name: Model name for the title.
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for gen, metrics in sorted(cross_gen_results.items()):
        if "labels" in metrics and "probs" in metrics:
            fpr, tpr, _ = roc_curve(metrics["labels"], metrics["probs"])
            ax.plot(fpr, tpr, label=f"{gen} (AUC={metrics['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curves by Generator")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate AI image detector")
    parser.add_argument(
        "--model",
        type=str,
        default="clip_probe",
        choices=["clip_probe", "hybrid", "freq_guided"],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--robustness", action="store_true", help="Run robustness evaluation"
    )
    parser.add_argument(
        "--robustness_gen",
        type=str,
        default="sd15",
        help="Generator to use for robustness eval",
    )
    args = parser.parse_args()

    config = Config()
    fix_seeds(config.seed)
    device = config.device
    needs_dct = args.model in ("hybrid", "freq_guided")

    print(f"Evaluating {args.model} on {device}")

    # Load model
    model = get_model(args.model, config).to(device)
    ckpt_path = config.checkpoint_dir / f"{args.model}_best.pth"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}")

    model.eval()

    # Cross-generator evaluation
    print("\n=== Cross-Generator Evaluation ===")
    cross_gen_results: dict = {}

    for gen in config.test_generators:
        gen_dir = config.data_dir / "test" / gen
        if not gen_dir.exists() or not any(gen_dir.rglob("*.jpg")):
            print(f"  Skipping {gen} — no test data")
            continue

        if needs_dct:
            test_ds = AIDetectDCTDataset(
                config.data_dir,
                split="test",
                transform=get_eval_transforms(),
                is_test=True,
                generator=gen,
            )
        else:
            test_ds = AIDetectTestDataset(
                config.data_dir, generator=gen, transform=get_eval_transforms()
            )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        result = evaluate_generator(model, test_loader, device, needs_dct)
        cross_gen_results[gen] = result
        print(
            f"  {gen:12s}: acc={result['accuracy']:.4f} auc={result['auc']:.4f} "
            f"f1={result['f1']:.4f}"
        )

    # Save cross-gen results (without raw labels/probs for JSON)
    save_results = {}
    for gen, r in cross_gen_results.items():
        save_results[gen] = {k: v for k, v in r.items() if k not in ("labels", "probs")}

    results_path = config.results_dir / "metrics" / f"{args.model}_cross_gen.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)

    # Generate table and plot
    table_path = config.results_dir / "tables" / f"{args.model}_results.md"
    table = generate_results_table(save_results, args.model, table_path)
    print(f"\n{table}")

    plot_path = config.results_dir / "plots" / f"{args.model}_roc_curves.png"
    plot_roc_curves(cross_gen_results, args.model, plot_path)
    print(f"\nROC curves saved to {plot_path}")

    # Robustness evaluation
    if args.robustness:
        print(f"\n=== Robustness Evaluation ({args.robustness_gen}) ===")
        rob_results = evaluate_robustness(
            model,
            config.data_dir,
            args.robustness_gen,
            device,
            needs_dct,
            args.batch_size,
        )

        for deg, metrics in rob_results.items():
            print(f"  {deg:12s}: acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f}")

        rob_path = config.results_dir / "metrics" / f"{args.model}_robustness.json"
        with open(rob_path, "w") as f:
            json.dump(rob_results, f, indent=2)
        print(f"Robustness results saved to {rob_path}")


if __name__ == "__main__":
    main()
