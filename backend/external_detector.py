"""
Wrapper around a public pre-trained AI image detector
(haywoodsloan/ai-image-detector-deploy on HuggingFace).

This model was trained on a broader / more recent generator pool than our
custom heads (which only saw GenImage 2023-era data + 632 picsum photos).
On internal eval it scores 100% on smartphone-style real photos and 100%
on Pollinations modern-AI generations — exactly the two failure modes our
custom heads have.

Used as the deployment default. Our 5 custom heads remain available in the
compare panel for the project's research narrative.

Loaded lazily on first use (same pattern as ModelManager).
"""

from __future__ import annotations

import time
from typing import Optional

from PIL import Image

# Public HF model — no auth required.
MODEL_ID = "haywoodsloan/ai-image-detector-deploy"


class ExternalDetector:
    """Lazy wrapper around the HF transformers image-classification pipeline."""

    def __init__(self) -> None:
        self._pipe = None
        self._device = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        # Import inside the method so the rest of the backend can boot even
        # if transformers isn't installed yet.
        from transformers import pipeline
        import torch

        device = 0 if (torch.cuda.is_available()) else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        # transformers' pipeline accepts integer 0 for cuda; for mps/cpu it
        # accepts the device string.
        self._pipe = pipeline("image-classification", model=MODEL_ID, device=device)
        self._device = device

    def predict(self, image_pil: Image.Image) -> dict:
        """Run the external detector on a PIL image.

        Returns a dict shaped like our other backends:
            {
                "fake_probability": float in [0, 1],
                "label_top": str,
                "raw": list of dicts (full model output),
                "inference_time_ms": float,
            }

        The model emits two labels — typically "artificial" and "real" / "human".
        We map any AI-side label to fake_probability.
        """
        self._ensure_loaded()
        t0 = time.time()
        out = self._pipe(image_pil.convert("RGB"))
        # out is a list of {label, score} sorted by score
        # Find the artificial-side score (case-insensitive on label)
        ai_keywords = ("artificial", "fake", "ai", "synthetic", "generated")
        fake_score = 0.0
        for entry in out:
            label_lower = entry["label"].lower()
            if any(k in label_lower for k in ai_keywords):
                fake_score = float(entry["score"])
                break
        else:
            # fallback: treat the lower-scoring class as the AI side
            fake_score = float(min(e["score"] for e in out))
        return {
            "fake_probability": fake_score,
            "label_top": out[0]["label"],
            "raw": [{"label": e["label"], "score": float(e["score"])} for e in out],
            "inference_time_ms": (time.time() - t0) * 1000.0,
        }


# Process-wide singleton — avoids reloading the model on every request.
_singleton: Optional[ExternalDetector] = None


def get_external_detector() -> ExternalDetector:
    global _singleton
    if _singleton is None:
        _singleton = ExternalDetector()
    return _singleton
