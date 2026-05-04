"""
FastAPI backend for AI-Generated Image Detector.

Endpoints:
    POST /detect          — Single image detection
    POST /detect/compare  — Comparative detection (all models)
    GET  /dashboard/data  — Evaluation metrics for dashboard
    GET  /health          — Health check
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import ModelManager

app = FastAPI(
    title="AI Image Detector",
    description="Detect AI-generated images with Grad-CAM explainability",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager (lazy loading)
manager: ModelManager | None = None


def get_manager() -> ModelManager:
    """Get or create the model manager."""
    global manager
    if manager is None:
        manager = ModelManager()
    return manager


SUPPORTED_MODELS = (
    "external",
    "clip_probe",
    "hybrid",
    "hybrid_robust",
    "hybrid_robust_v2",
    "hybrid_robust_v3",
    "freq_guided",
    "freq_guided_no_robust",
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "models": list(SUPPORTED_MODELS)}


@app.post("/detect")
async def detect_image(
    file: UploadFile = File(...),
    model: str = "external",
) -> dict:
    """Detect if an image is AI-generated.

    Args:
        file: Uploaded image file.
        model: One of clip_probe, hybrid, hybrid_robust, freq_guided,
            freq_guided_no_robust. Defaults to hybrid_robust (best under
            real-world degradations per the ablation study).

    Returns:
        Detection result with verdict, confidence, heatmap, explanations.
    """
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    try:
        contents = await file.read()
        import io

        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    mgr = get_manager()
    try:
        result = mgr.predict(image, model_name=model)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return result


@app.post("/detect/compare")
async def detect_comparative(file: UploadFile = File(...)) -> dict:
    """Run detection with all models for comparison.

    Args:
        file: Uploaded image file.

    Returns:
        Comparative results from all three models.
    """
    try:
        contents = await file.read()
        import io

        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    mgr = get_manager()
    try:
        result = mgr.predict_comparative(image)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Comparison error: {e}")
    return result


@app.get("/dashboard/data")
async def dashboard_data() -> dict:
    """Get evaluation metrics for the dashboard.

    Returns:
        Cross-generator and ablation results for all models.
    """
    results_dir = PROJECT_ROOT / "results" / "metrics"

    data: dict = {
        "cross_generator": {},
        "ablation": {},
    }

    # Load cross-gen results for each model (all 5 variants)
    all_models = [
        "clip_probe", "hybrid", "hybrid_robust",
        "freq_guided_no_robust", "freq_guided",
    ]
    for model_name in all_models:
        path = results_dir / f"{model_name}_cross_gen.json"
        if path.exists():
            with open(path) as f:
                data["cross_generator"][model_name] = json.load(f)

    # Load ablation table
    ablation_path = PROJECT_ROOT / "results" / "tables" / "ablation_table.md"
    if ablation_path.exists():
        with open(ablation_path) as f:
            data["ablation"]["table_md"] = f.read()

    # Load training histories
    for model_name in all_models:
        path = results_dir / f"{model_name}_training.json"
        if path.exists():
            with open(path) as f:
                training = json.load(f)
                data.setdefault("training", {})[model_name] = {
                    "best_val_auc": training.get("best_val_auc"),
                    "total_epochs": training.get("total_epochs"),
                }

    # Load per-model robustness
    for model_name in all_models:
        path = results_dir / f"{model_name}_robustness.json"
        if path.exists():
            with open(path) as f:
                data.setdefault("robustness", {})[model_name] = json.load(f)

    # Load calibration (T, NLL, ECE per model)
    calib_path = results_dir / "calibration.json"
    if calib_path.exists():
        with open(calib_path) as f:
            data["calibration"] = json.load(f)

    return data


# Serve frontend — mount AFTER API routes so they take priority
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML."""
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
