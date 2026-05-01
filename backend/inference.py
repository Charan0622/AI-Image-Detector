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
from src.gradcam_utils import (
    clip_attention_rollout,
    create_heatmap_overlay,
    gradcam_freq_branch,
)
from src.models.freq_guided import FreqGuidedFromFeatures
from src.models.hybrid import FrequencyCNN
from src.seed import fix_seeds
from src.train_freq_guided import HybridRobustFromFeatures
from src.train_hybrid import HybridFromFeatures
from src.train_probe import LinearProbeHead
from src.transforms import compute_dct_map, get_eval_transforms


def canonicalize_for_inference(img: Image.Image) -> Image.Image:
    """Match the exact preprocessing applied during training.

    All training images went through: LANCZOS resize to 224x224, then re-saved
    as JPEG Q=95. Any image that skips the re-encode looks out-of-distribution
    to the trained models, especially uncompressed PNGs and high-quality JPEGs.

    This applies the same pipeline at inference so user uploads match the
    training distribution.
    """
    rgb = img.convert("RGB")
    # Resize to 224 preserving aspect ratio via center crop
    short = min(rgb.size)
    left = (rgb.size[0] - short) // 2
    top = (rgb.size[1] - short) // 2
    rgb = rgb.crop((left, top, left + short, top + short))
    rgb = rgb.resize((224, 224), Image.LANCZOS)
    # Re-encode as JPEG Q=95, then reload
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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

        # Out-of-distribution detector. Uses cached CLIP features to estimate
        # how far a new upload sits from the training distribution. If we can't
        # load the cache (or it doesn't exist), fall back to disabling OOD.
        self.ood_centroid: np.ndarray | None = None
        self.ood_scale: float = 1.0  # std of training-sample distances; used to normalise
        try:
            feat_path = self.config.project_root / "data" / "features" / "train_features.npy"
            if feat_path.exists():
                feats = np.load(feat_path).astype(np.float32)            # (N, 512)
                feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
                self.ood_centroid = feats.mean(axis=0)                   # (512,)
                self.ood_centroid /= np.linalg.norm(self.ood_centroid) + 1e-8
                # Sample 10k features to estimate distance scale
                rng = np.random.default_rng(0)
                idx = rng.choice(feats.shape[0], size=min(10000, feats.shape[0]), replace=False)
                dists = 1.0 - feats[idx] @ self.ood_centroid             # cosine distances
                self.ood_mean = float(dists.mean())
                self.ood_scale = float(dists.std() + 1e-6)
                print(f"[ood] training centroid loaded. mean dist={self.ood_mean:.3f}, std={self.ood_scale:.3f}")
        except Exception as e:
            print(f"[ood] failed to build training centroid: {e}")

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

    def _ood_score(self, clip_feat: torch.Tensor) -> float:
        """Cosine-distance OOD score, z-scored against training distances.

        Returns a value in roughly [0, 1] where:
            ~0.0  → very close to the training distribution
            ~0.5  → typical training sample (1 sigma above the mean)
            >=1.0 → noticeably out of distribution (3+ sigma)
        """
        if self.ood_centroid is None:
            return 0.0
        f = clip_feat.detach().cpu().numpy().reshape(-1).astype(np.float32)
        f = f / (np.linalg.norm(f) + 1e-8)
        cos_dist = float(1.0 - f @ self.ood_centroid)
        # Convert to a 0–1 OOD level: 0 at training mean, 1 at +3 sigma
        z = (cos_dist - self.ood_mean) / (3.0 * self.ood_scale)
        return float(max(0.0, min(1.0, z)))

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
        elif name == "hybrid_robust_v2":
            # v2 = hybrid_robust warm-fine-tuned with smartphone-aesthetic +
            # double-JPEG augmentation on the expanded training set
            # (see scripts/expand_training_data.py + src/train_hybrid_robust_v2.py).
            model = HybridRobustFromFeatures(**common_kwargs)
            ckpt_path = ckpt_dir / "hybrid_robust_v2_best.pth"
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

    def predict(self, image_pil: Image.Image, model_name: str = "hybrid_robust_v2") -> dict:
        """Run prediction on a single image.

        Args:
            image_pil: Input PIL image.
            model_name: Model to use.

        Returns:
            Dict with verdict, confidence, heatmap_base64, explanations.
        """
        import time

        t0 = time.time()

        # Canonicalize the upload to match training preprocessing exactly.
        # Training data was 224x224 LANCZOS + JPEG Q=95. Skipping this step
        # makes PNGs and high-quality JPEGs look out-of-distribution.
        image_pil = canonicalize_for_inference(image_pil)

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

        # OOD score: how far this image is from the training distribution
        ood_score = self._ood_score(clip_feat)

        # If the input is meaningfully out of distribution, the chosen head
        # tends to saturate (p(AI) ≈ 1.0 for anything not matching training),
        # so fall back to an ensemble average across all 5 heads. Ensembling
        # softens the saturated signals and gives a more honest probability
        # when no single model is reliable.
        if ood_score >= 0.5:
            ensemble_probs = []
            for ens_name in ("clip_probe", "hybrid", "hybrid_robust", "hybrid_robust_v2", "freq_guided"):
                try:
                    em = self._load_model(ens_name)
                    ens_T = self.temperatures.get(ens_name, 1.0)
                    if ens_name == "clip_probe":
                        en_logits = em(clip_feat)
                    else:
                        en_logits = em(clip_feat, dct_tensor)
                    en_p = torch.softmax(en_logits / max(ens_T, 1e-3), dim=1)[0, 1].item()
                    ensemble_probs.append(en_p)
                except Exception as e:
                    print(f"[ensemble] {ens_name} failed: {e}")
            if ensemble_probs:
                ensemble_fake = sum(ensemble_probs) / len(ensemble_probs)
                # Use ensemble probability as the primary signal under OOD
                fake_prob = ensemble_fake
                real_prob = 1.0 - fake_prob

        # Verdict logic with a prior-toward-Real adjustment.
        #
        # The trained heads saturate to p(AI) ≈ 1.0 on out-of-distribution
        # inputs (smartphone photos, modern AI generators). In deployment,
        # most real uploads are real photos, so the right Bayesian move is
        # to require strong evidence *and* an in-distribution input before
        # we flag "AI". Otherwise we lean toward Real with a hedge.
        #
        # IN-DISTRIBUTION (ood < 0.40):
        #     p > 0.85   → AI-Generated
        #     p < 0.35   → Real
        #     0.50–0.85  → Likely AI
        #     0.35–0.50  → Likely Real
        # OUT-OF-DISTRIBUTION (ood ≥ 0.40):
        #     The head's confident "AI" reading is unreliable. Treat it as
        #     suggestive at best.
        #     p > 0.95   → Likely AI       (still hedge — never confident)
        #     p < 0.50   → Likely Real
        #     0.50–0.95  → Likely Real     (prior wins)
        # Strong OOD (≥ 0.85): genuinely Inconclusive, model can't say.
        if ood_score >= 0.85:
            verdict = "Uncertain"
            band = "uncertain"
            ood_reason = "outside training distribution"
        elif ood_score >= 0.40:
            # OOD region — the head's p(AI) is unreliable here (it saturates
            # near 1.0 on most off-distribution inputs, real or fake). The
            # right Bayesian move in deployment is to fall back to the prior:
            # most uploads are real photos, so default Likely Real and
            # surface the OOD caveat explicitly.
            verdict = "Likely Real"
            band = "likely_real"
            ood_reason = "head is unreliable on this input — defaulted to prior"
        else:
            # In-distribution — trust the head
            if fake_prob > 0.85:
                verdict = "AI-Generated"
                band = "fake"
                ood_reason = None
            elif fake_prob < 0.35:
                verdict = "Real"
                band = "real"
                ood_reason = None
            elif fake_prob <= 0.50:
                verdict = "Likely Real"
                band = "likely_real"
                ood_reason = "lightly edited possible"
            else:
                verdict = "Likely AI"
                band = "likely_ai"
                ood_reason = "borderline confidence"

        if verdict == "AI-Generated":
            confidence = fake_prob
        elif verdict == "Likely AI":
            confidence = fake_prob
        elif verdict in ("Real", "Likely Real"):
            confidence = 1 - fake_prob
        else:
            confidence = max(fake_prob, 1 - fake_prob)

        # Spatial heatmap via CLIP attention rollout — this actually aligns
        # with the original image, unlike the previous DCT-space Grad-CAM.
        heatmap_b64 = None
        heatmap_np = None
        try:
            encoder = self._get_clip_encoder()
            img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            heatmap_np = clip_attention_rollout(encoder, img_tensor, self.device)
            # For real (not fake) verdicts, the story is "here is what the
            # model finds salient". For fake verdicts, show same region as
            # "what the model found suspicious".
            overlay = create_heatmap_overlay(image_pil, heatmap_np, alpha=0.65)
            buffer = io.BytesIO()
            overlay.save(buffer, format="PNG")
            heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            # Fallback gracefully: no heatmap rather than crashing the request
            print(f"[inference] attention rollout failed: {e}")

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
            "band": band,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "ood_score": round(ood_score, 3),
            "ood_reason": ood_reason,
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
