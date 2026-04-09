"""
Grad-CAM and attention-based explainability for the detector.

Two complementary approaches:

1. Frequency Spatial Attention Map (from FreqGuidedDetector)
   - Directly available from the multi-scale frequency CNN
   - Shows which spatial regions have frequency anomalies
   - Resolution: 14x14 (upscaled to image size)

2. Grad-CAM on the frequency CNN
   - Gradient-weighted activation maps from the last conv layer
   - Shows which frequency regions drive the classification

3. Combined heatmap overlay on original image

Output:
    - Heatmap overlay on original image (PIL Image)
    - Raw attention weights (numpy array)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a heatmap on an image.

    Args:
        image_pil: Original PIL image.
        heatmap: Heatmap array, shape (H, W), values in [0, 1].
        alpha: Overlay transparency (0=image only, 1=heatmap only).
        colormap: Matplotlib colormap name.

    Returns:
        PIL Image with heatmap overlay.
    """
    import matplotlib.cm as cm

    # Get colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # (H, W, 3) RGB float
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored).resize(image_pil.size, Image.BILINEAR)

    # Blend
    overlay = Image.blend(image_pil.convert("RGB"), heatmap_pil, alpha)
    return overlay


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
