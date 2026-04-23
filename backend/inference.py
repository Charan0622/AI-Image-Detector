"""
Model loading and prediction pipeline for the API.

Loads all three model variants and provides inference functions
that return verdicts, confidence scores, and heatmaps.
"""

import base64
import io
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from src.config import Config
from src.explain import generate_explanations
from src.freq_heuristic import freq_heuristic_score
from src.gradcam_utils import create_heatmap_overlay, gradcam_freq_branch
from src.models.freq_guided import FreqGuidedFromFeatures
from src.models.hybrid import FrequencyCNN
from src.seed import fix_seeds
from src.train_freq_guided import HybridRobustFromFeatures
from src.train_hybrid import HybridFromFeatures
from src.train_probe import LinearProbeHead
from src.transforms import compute_dct_map, get_eval_transforms


class ModelManager:
    """Manages loading and inference for all model variants.

    Loads models lazily on first use and caches them.
    """

    def __init__(self) -> None:
        self.config = Config()
        self.device = self.config.device
        self.transform = get_eval_transforms()
        self._clip_encoder = None
        self._models: dict = {}
        fix_seeds(self.config.seed)

        # Load per-model temperature scaling, if present. Defaults to 1.0 (no-op).
        self.temperatures: dict[str, float] = {}
        calib_path = self.config.results_dir / "metrics" / "calibration.json"
        if calib_path.exists():
            import json as _json
            with open(calib_path) as _f:
                data = _json.load(_f)
            self.temperatures = {m: float(v.get("temperature", 1.0)) for m, v in data.items()}

    def _get_clip_encoder(self) -> torch.nn.Module:
        """Load and cache CLIP visual encoder."""
        if self._clip_encoder is None:
            clip_model, _, _ = open_clip.create_model_and_transforms(
                self.config.clip_model_name, pretrained=self.config.clip_pretrained
            )
            self._clip_encoder = clip_model.visual.to(self.device)
            self._clip_encoder.eval()
            for p in self._clip_encoder.parameters():
                p.requires_grad = False
        return self._clip_encoder

    def _extract_clip_features(self, image_pil: Image.Image) -> torch.Tensor:
        """Extract CLIP features from a PIL image."""
        encoder = self._get_clip_encoder()
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = encoder(img_tensor)
        return features  # (1, 512)

    def _load_model(self, name: str) -> torch.nn.Module:
        """Load a model by name."""
        if name in self._models:
            return self._models[name]

        ckpt_dir = self.config.checkpoint_dir

        common_kwargs = dict(
            clip_dim=self.config.clip_embed_dim,
            freq_out_dim=self.config.freq_branch_out_dim,
            fusion_hidden=self.config.fusion_hidden_dim,
            fusion_dropout=self.config.fusion_dropout,
        )
        if name == "clip_probe":
            model = LinearProbeHead(input_dim=512)
            ckpt_path = ckpt_dir / "clip_probe_best.pth"
        elif name == "hybrid":
            model = HybridFromFeatures(**common_kwargs)
            ckpt_path = ckpt_dir / "hybrid_best.pth"
        elif name == "hybrid_robust":
            model = HybridRobustFromFeatures(**common_kwargs)
            ckpt_path = ckpt_dir / "hybrid_robust_best.pth"
        elif name == "freq_guided":
            model = FreqGuidedFromFeatures(**common_kwargs)
            ckpt_path = ckpt_dir / "freq_guided_best.pth"
        elif name == "freq_guided_no_robust":
            model = FreqGuidedFromFeatures(**common_kwargs)
            ckpt_path = ckpt_dir / "freq_guided_no_robust_best.pth"
        else:
            raise ValueError(f"Unknown model: {name}")

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        model = model.to(self.device)
        model.eval()
        self._models[name] = model
        return model

    def predict(self, image_pil: Image.Image, model_name: str = "hybrid_robust") -> dict:
        """Run prediction on a single image.

        Args:
            image_pil: Input PIL image.
            model_name: Model to use.

        Returns:
            Dict with verdict, confidence, heatmap_base64, explanations.
        """
        import time

        t0 = time.time()

        # Extract features
        clip_feat = self._extract_clip_features(image_pil)
        dct_map = compute_dct_map(image_pil)
        dct_tensor = torch.from_numpy(dct_map).unsqueeze(0).unsqueeze(0).float().to(self.device)

        model = self._load_model(model_name)

        # Model inference
        with torch.no_grad():
            needs_dct = model_name != "clip_probe"
            logits = model(clip_feat, dct_tensor) if needs_dct else model(clip_feat)

            # Apply temperature scaling for calibrated probabilities.
            # argmax (and thus the verdict) is T-invariant; only the
            # confidence values change to be honest about uncertainty.
            T = self.temperatures.get(model_name, 1.0)
            probs = torch.softmax(logits / max(T, 1e-3), dim=1)[0]
            fake_prob = probs[1].item()
            real_prob = probs[0].item()

        # Frequency heuristic (for explanations only, not verdict)
        heuristic_score, heuristic_reasons = freq_heuristic_score(image_pil)

        verdict = "AI-Generated" if fake_prob > 0.5 else "Real"
        confidence = fake_prob if verdict == "AI-Generated" else real_prob

        # Generate heatmap (for models with freq branch)
        heatmap_b64 = None
        heatmap_np = None
        if model_name in ("hybrid", "hybrid_robust", "freq_guided", "freq_guided_no_robust"):
            heatmap_np = gradcam_freq_branch(model, clip_feat, dct_tensor, self.device, target_class=1)
            overlay = create_heatmap_overlay(image_pil, heatmap_np, alpha=0.4)
            buffer = io.BytesIO()
            overlay.save(buffer, format="PNG")
            heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Generate explanations (combine model + heuristic)
        explanations = []
        if heatmap_np is not None:
            explanations = generate_explanations(image_pil, heatmap_np, confidence, verdict)

        # Add heuristic reasons
        for reason in heuristic_reasons[:3]:
            explanations.append({
                "region": "frequency analysis",
                "reason": reason,
                "severity": min(heuristic_score, 1.0),
            })

        inference_time = (time.time() - t0) * 1000

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "heatmap_base64": f"data:image/png;base64,{heatmap_b64}" if heatmap_b64 else None,
            "explanations": explanations,
            "model_name": model_name,
            "inference_time_ms": round(inference_time, 1),
        }

    def predict_comparative(self, image_pil: Image.Image) -> dict:
        """Run prediction with all models for comparison.

        Args:
            image_pil: Input PIL image.

        Returns:
            Dict with results from all models.
        """
        results = []
        for name in ["clip_probe", "hybrid", "hybrid_robust", "freq_guided"]:
            result = self.predict(image_pil, model_name=name)
            results.append(result)

        return {"models": results}
