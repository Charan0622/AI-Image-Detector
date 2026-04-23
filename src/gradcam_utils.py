"""
Spatial explainability for the detector.

The primary heatmap is now **CLIP attention rollout** (Abnar & Zuidema 2020,
Chefer et al. 2021 style). Rollout propagates multi-head self-attention
through all ViT-B/16 blocks to estimate which image patches the CLS token
actually attended to, producing a genuinely spatial 14x14 map that
meaningfully aligns with the original image.

This replaces the earlier "Grad-CAM on the DCT frequency branch" heatmap,
which was misleading: the DCT map is a spectral representation — its
coordinates index frequency coefficients, not image regions — so overlaying
a DCT-space heatmap on the original image was nonsensical.

Functions:
    clip_attention_rollout()  — spatial heatmap from a forward pass
    gradcam_freq_branch()     — kept for internal diagnostic use only
    create_heatmap_overlay()  — blend a (H, W) heatmap on a PIL image
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def clip_attention_rollout(
    clip_encoder: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    discard_ratio: float = 0.5,
) -> np.ndarray:
    """Compute a spatial heatmap via CLIP ViT attention rollout.

    Args:
        clip_encoder: An OpenCLIP ``VisionTransformer`` (``clip_model.visual``).
        image_tensor: Preprocessed image tensor, shape ``(1, 3, 224, 224)``.
        device: Compute device.
        discard_ratio: Fraction of lowest attention weights to zero out per head
            before rollout. 0 = keep everything; 0.85 matches Abnar &
            Zuidema's noise-suppression recommendation for ViT.

    Returns:
        np.ndarray shape ``(224, 224)``, values in [0, 1]. Higher values =
        regions the CLS token attended to most through the full stack.
    """
    clip_encoder.eval()
    image_tensor = image_tensor.to(device)
    attentions: list[torch.Tensor] = []

    # Wrap each block's attention with a forward_hook that also returns weights.
    # OpenCLIP's resblock.attn is a nn.MultiheadAttention called with need_weights=False
    # by default, so we monkey-patch call it ourselves via a pre-hook and capture.
    blocks = clip_encoder.transformer.resblocks
    handles = []

    def make_attn_hook(block):
        orig_forward = block.attn.forward

        def new_forward(*args, **kwargs):
            # Force need_weights=True, average_attn_weights=True → shape (B, N, N)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            out, weights = orig_forward(*args, **kwargs)
            attentions.append(weights.detach())
            return out, weights  # callers that ignored weights still work

        block.attn.forward = new_forward
        return lambda: setattr(block.attn, "forward", orig_forward)

    restore_fns = [make_attn_hook(b) for b in blocks]

    try:
        _ = clip_encoder(image_tensor)
    finally:
        for r in restore_fns:
            r()

    if not attentions:
        # Fallback: return a uniform map
        return np.full((224, 224), 0.5, dtype=np.float32)

    # Stack: each tensor is (B, N, N) where N = num_tokens (1 CLS + 196 patches)
    # Apply discard threshold per layer per head-avg map
    processed = []
    for A in attentions:
        # Add identity (residual pathway through attention)
        A = A + torch.eye(A.size(-1), device=A.device).unsqueeze(0)
        if discard_ratio > 0:
            flat = A.view(A.size(0), -1)
            k = int(flat.size(-1) * discard_ratio)
            if k > 0:
                thresh, _ = flat.topk(k, dim=-1, largest=False)
                t = thresh.max(dim=-1, keepdim=True).values
                A = torch.where(A < t.unsqueeze(-1), torch.zeros_like(A), A)
        # Row-normalise
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        processed.append(A)

    # Rollout = matmul of all layer attentions
    rollout = processed[0]
    for A in processed[1:]:
        rollout = torch.matmul(A, rollout)

    # CLS row (index 0), drop CLS self-attention to CLS, keep patch tokens
    cls_to_patches = rollout[0, 0, 1:]  # (196,)
    n = int(cls_to_patches.numel() ** 0.5)
    heatmap = cls_to_patches.reshape(n, n).cpu().numpy().astype(np.float32)

    # Percentile-based contrast stretch so the overlay reads well even when
    # raw attention is sparse. Maps 5th percentile → 0 and 95th → 1, clipped.
    lo, hi = np.percentile(heatmap, [5, 95])
    if hi > lo:
        heatmap = np.clip((heatmap - lo) / (hi - lo), 0, 1)
    else:
        heatmap = np.zeros_like(heatmap)

    # Upscale 14x14 -> 224x224 with bilinear filtering
    pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    pil = pil.resize((224, 224), Image.BILINEAR)
    return np.array(pil).astype(np.float32) / 255.0


def get_freq_attention_map(
    model: torch.nn.Module,
    dct_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Get frequency spatial attention map from the model.

    Args:
        model: FreqGuidedFromFeatures or HybridFromFeatures model.
        dct_tensor: DCT map tensor, shape (1, 1, 224, 224).
        device: Compute device.

    Returns:
        Attention map as numpy array, shape (224, 224), values in [0, 1].
    """
    model.eval()
    dct_tensor = dct_tensor.to(device)

    with torch.no_grad():
        if hasattr(model, "freq_encoder") and hasattr(model.freq_encoder, "spatial_attn"):
            # FreqGuidedFromFeatures — has spatial attention
            _, attn = model.freq_encoder(dct_tensor)
            attn_map = attn.squeeze().cpu().numpy()  # (14, 14)
        else:
            # Fallback: use frequency CNN activations
            feat = model.freq_encoder.features(dct_tensor)
            attn_map = feat.mean(dim=1).squeeze().cpu().numpy()
            # Normalize
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upscale to 224x224
    attn_pil = Image.fromarray((attn_map * 255).astype(np.uint8))
    attn_pil = attn_pil.resize((224, 224), Image.BILINEAR)
    return np.array(attn_pil).astype(np.float32) / 255.0


def gradcam_freq_branch(
    model: torch.nn.Module,
    clip_feat: torch.Tensor,
    dct_tensor: torch.Tensor,
    device: torch.device,
    target_class: int = 1,
) -> np.ndarray:
    """Compute Grad-CAM on the frequency branch.

    Uses gradients of the target class w.r.t. the last conv layer
    to create a class-discriminative heatmap.

    Args:
        model: Model with freq_encoder.
        clip_feat: Pre-extracted CLIP features, shape (1, 512).
        dct_tensor: DCT map, shape (1, 1, 224, 224).
        device: Compute device.
        target_class: Class to explain (1=fake, 0=real).

    Returns:
        Grad-CAM heatmap, shape (224, 224), values in [0, 1].
    """
    model.eval()
    clip_feat = clip_feat.to(device)
    dct_tensor = dct_tensor.to(device).requires_grad_(True)

    # Find the last conv layer in freq_encoder
    target_layer = None
    activation = {}
    gradient = {}

    def find_last_conv(module, prefix=""):
        nonlocal target_layer
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Conv2d):
                target_layer = child
            find_last_conv(child, full_name)

    find_last_conv(model.freq_encoder)

    if target_layer is None:
        # Fallback to freq attention
        return get_freq_attention_map(model, dct_tensor.detach(), device)

    # Register hooks
    def forward_hook(module, input, output):
        activation["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradient["value"] = grad_output[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    logits = model(clip_feat, dct_tensor)
    score = logits[0, target_class]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Remove hooks
    fwd_handle.remove()
    bwd_handle.remove()

    # Compute Grad-CAM
    grads = gradient["value"]  # (1, C, H, W)
    acts = activation["value"]  # (1, C, H, W)

    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = F.relu(cam)

    # Normalize
    cam = cam.squeeze().detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Upscale to 224x224
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((224, 224), Image.BILINEAR)
    return np.array(cam_pil).astype(np.float32) / 255.0


def create_heatmap_overlay(
    image_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "inferno",
) -> Image.Image:
    """Overlay a heatmap on an image with alpha varying by heatmap intensity.

    Dark / low-attention regions stay as the original image; bright / high-
    attention regions show a warm colour tint. This reads as "the model paid
    attention here" without destroying the underlying photo the way a flat
    alpha-blend would.

    Args:
        image_pil: Original PIL image.
        heatmap: Heatmap array, shape (H, W), values in [0, 1].
        alpha: Maximum blend strength at heatmap=1. Low-attention regions
            blend at 0.
        colormap: Matplotlib colormap. ``inferno`` is perceptually uniform
            and reads as warm light on dark.

    Returns:
        PIL Image with heatmap overlay.
    """
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) RGB float
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored).resize(image_pil.size, Image.BILINEAR)

    # Alpha map: low attention → transparent, high attention → alpha.
    # Gamma < 1 boosts mid-range so the overlay is clearly visible.
    attn = np.array(Image.fromarray((heatmap * 255).astype(np.uint8))
                    .resize(image_pil.size, Image.BILINEAR)).astype(np.float32) / 255.0
    attn = np.power(attn, 0.7) * alpha  # (H_img, W_img)

    base = np.array(image_pil.convert("RGB")).astype(np.float32) / 255.0  # (H, W, 3)
    colored = np.array(heatmap_pil).astype(np.float32) / 255.0            # (H, W, 3)
    blended = base * (1 - attn[..., None]) + colored * attn[..., None]
    return Image.fromarray((blended * 255).astype(np.uint8))


def generate_gradcam_visualization(
    model: torch.nn.Module,
    image_path: str,
    clip_feat: torch.Tensor,
    device: torch.device,
    target_class: int = 1,
) -> dict:
    """Generate complete Grad-CAM visualization for one image.

    Args:
        model: Trained model.
        image_path: Path to the original image.
        clip_feat: Pre-extracted CLIP features for this image.
        device: Compute device.
        target_class: Class to explain.

    Returns:
        Dictionary with:
            - 'original': PIL Image
            - 'heatmap': numpy array (224, 224)
            - 'overlay': PIL Image with heatmap
            - 'freq_attn': numpy array (224, 224) if available
    """
    from src.transforms import compute_dct_map

    image_pil = Image.open(image_path).convert("RGB")

    # Compute DCT
    dct_map = compute_dct_map(image_pil)
    dct_tensor = torch.from_numpy(dct_map).unsqueeze(0).unsqueeze(0).float()

    # Get Grad-CAM heatmap
    clip_feat_tensor = clip_feat.unsqueeze(0) if clip_feat.dim() == 1 else clip_feat
    heatmap = gradcam_freq_branch(model, clip_feat_tensor, dct_tensor, device, target_class)

    # Get frequency attention if available
    freq_attn = None
    if hasattr(model, "freq_encoder") and hasattr(model.freq_encoder, "spatial_attn"):
        freq_attn = get_freq_attention_map(model, dct_tensor, device)

    # Create overlay
    overlay = create_heatmap_overlay(image_pil, heatmap, alpha=0.4)

    return {
        "original": image_pil,
        "heatmap": heatmap,
        "overlay": overlay,
        "freq_attn": freq_attn,
    }
