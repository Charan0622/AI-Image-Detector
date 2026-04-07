"""Verify that the environment is correctly configured."""

import sys
import torch
import platform


def main() -> None:
    """Run all environment verification checks."""
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.randn(2, 3, 224, 224, device=device)
        print(f"MPS tensor test: {x.shape} on {x.device} ✅")
    else:
        print("⚠️ MPS not available — will fall back to CPU")

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"OpenCLIP ViT-B/16 loaded ✅ — {param_count:.1f}M params")

    # Disk check
    import shutil

    total, used, free = shutil.disk_usage("/")
    print(f"Disk: {free / 1e9:.1f}GB free of {total / 1e9:.1f}GB total")

    print("\n🟢 Environment setup complete!")


if __name__ == "__main__":
    main()
