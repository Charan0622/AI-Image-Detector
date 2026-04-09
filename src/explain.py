"""
Generate text explanations for WHY an image is classified as AI-generated.

Approach:
    1. Get the frequency attention/Grad-CAM heatmap
    2. Identify high-activation regions
    3. Analyze DCT spectrum for known artifact signatures
    4. Generate human-readable explanation strings

Returns a list of explanation strings like:
    - "High-frequency artifacts detected in the background region"
    - "Unnatural texture smoothness in central area"
    - "Spectral anomaly consistent with diffusion model upsampling"
"""

import numpy as np
from PIL import Image

from src.transforms import compute_dct_map


def analyze_frequency_spectrum(image_pil: Image.Image) -> dict:
    """Analyze the DCT spectrum for AI-generation indicators.

    Args:
        image_pil: Input PIL image.

    Returns:
        Dictionary with spectral analysis metrics.
    """
    dct_map = compute_dct_map(image_pil)
    h, w = dct_map.shape

    # Split into frequency bands
    low_freq = dct_map[: h // 4, : w // 4].mean()
    mid_freq = dct_map[h // 4 : h // 2, w // 4 : w // 2].mean()
    high_freq = dct_map[h // 2 :, w // 2 :].mean()

    # Frequency rolloff (ratio of high to low)
    rolloff = high_freq / (low_freq + 1e-8)

    # Spectral entropy
    flat = dct_map.flatten()
    flat = flat / (flat.sum() + 1e-8)
    entropy = -np.sum(flat * np.log(flat + 1e-10))

    # Grid artifact detection (look for periodic peaks)
    fft_of_dct = np.abs(np.fft.fft2(dct_map))
    center = fft_of_dct[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    peak_ratio = center.max() / (center.mean() + 1e-8)

    return {
        "low_freq_energy": float(low_freq),
        "mid_freq_energy": float(mid_freq),
        "high_freq_energy": float(high_freq),
        "freq_rolloff": float(rolloff),
        "spectral_entropy": float(entropy),
        "grid_artifact_score": float(peak_ratio),
    }


def get_region_name(row: int, col: int, grid_size: int = 3) -> str:
    """Convert grid position to human-readable region name.

    Args:
        row: Row in grid (0=top, 2=bottom).
        col: Column in grid (0=left, 2=right).
        grid_size: Grid divisions.

    Returns:
        Region name string.
    """
    v_names = ["top", "center", "bottom"]
    h_names = ["left", "center", "right"]

    v = v_names[min(row, grid_size - 1)]
    h = h_names[min(col, grid_size - 1)]

    if v == "center" and h == "center":
        return "central area"
    elif v == "center":
        return f"{h} side"
    elif h == "center":
        return f"{v} area"
    else:
        return f"{v}-{h} corner"


def identify_suspicious_regions(
    heatmap: np.ndarray, threshold: float = 0.6
) -> list[dict]:
    """Identify high-activation regions in the heatmap.

    Args:
        heatmap: Attention/Grad-CAM heatmap, shape (H, W), values [0, 1].
        threshold: Activation threshold for "suspicious" regions.

    Returns:
        List of dicts with region info.
    """
    h, w = heatmap.shape
    grid = 3
    cell_h, cell_w = h // grid, w // grid

    regions = []
    for r in range(grid):
        for c in range(grid):
            cell = heatmap[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            mean_activation = cell.mean()
            max_activation = cell.max()

            if mean_activation > threshold:
                regions.append(
                    {
                        "region": get_region_name(r, c, grid),
                        "mean_activation": float(mean_activation),
                        "max_activation": float(max_activation),
                        "severity": float(min(mean_activation / threshold, 1.5)),
                    }
                )

    # Sort by severity
    regions.sort(key=lambda x: x["severity"], reverse=True)
    return regions


def generate_explanations(
    image_pil: Image.Image,
    heatmap: np.ndarray,
    confidence: float,
    verdict: str,
) -> list[dict]:
    """Generate human-readable explanations for the detection result.

    Args:
        image_pil: Original PIL image.
        heatmap: Grad-CAM/attention heatmap, shape (H, W).
        confidence: Model confidence score (0-1).
        verdict: "AI-Generated" or "Real".

    Returns:
        List of explanation dicts with 'region', 'reason', 'severity'.
    """
    explanations = []

    # Spectral analysis
    spectrum = analyze_frequency_spectrum(image_pil)

    # Frequency rolloff analysis
    if spectrum["freq_rolloff"] < 0.3:
        explanations.append(
            {
                "region": "global",
                "reason": "Unusually smooth frequency rolloff — consistent with diffusion model generation",
                "severity": 0.8,
            }
        )
    elif spectrum["freq_rolloff"] > 0.7:
        explanations.append(
            {
                "region": "global",
                "reason": "Abnormal high-frequency energy — possible upsampling artifacts",
                "severity": 0.7,
            }
        )

    # Grid artifact detection
    if spectrum["grid_artifact_score"] > 15:
        explanations.append(
            {
                "region": "global",
                "reason": "Periodic spectral patterns detected — consistent with GAN grid artifacts",
                "severity": 0.9,
            }
        )

    # Spectral entropy
    if spectrum["spectral_entropy"] < 8.0:
        explanations.append(
            {
                "region": "global",
                "reason": "Low spectral entropy — image frequency distribution is unnaturally uniform",
                "severity": 0.6,
            }
        )

    # Region-specific explanations from heatmap
    suspicious = identify_suspicious_regions(heatmap, threshold=0.5)
    for region_info in suspicious[:3]:  # Top 3 regions
        severity = region_info["severity"]
        region = region_info["region"]

        if severity > 1.2:
            reason = f"Strong frequency anomaly in {region} — highly inconsistent with natural photography"
        elif severity > 0.8:
            reason = f"Moderate texture irregularity detected in {region}"
        else:
            reason = f"Subtle frequency pattern deviation in {region}"

        explanations.append(
            {
                "region": region,
                "reason": reason,
                "severity": min(float(severity), 1.0),
            }
        )

    # Add confidence-based explanation
    if verdict == "AI-Generated" and confidence > 0.9:
        explanations.insert(
            0,
            {
                "region": "overall",
                "reason": f"High confidence ({confidence:.0%}) detection — multiple AI-generation indicators found",
                "severity": confidence,
            },
        )

    # Sort by severity
    explanations.sort(key=lambda x: x["severity"], reverse=True)
    return explanations[:5]  # Return top 5
