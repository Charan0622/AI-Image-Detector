# EXPLANATION.md — Project Log

## Phase 0: Environment Setup
**Date:** 2026-04-07
**Status:** ✅ Complete

### What was done:
- Created project directory at `~/Documents/Deeplearning_Project/`
- Initialized git repository with remote `https://github.com/Charan0622/AI-Image-Detector.git`
- Created Python 3.12 virtual environment at `.venv/`
- Installed all dependencies (see `requirements.txt`)
- Fixed `pytorch-grad-cam` → `grad-cam` package name for Python 3.12 compatibility
- Verified MPS backend is available and functional
- Verified OpenCLIP ViT-B/16 loads successfully (149.6M params)
- Created full directory structure per the project design document
- Created `setup.sh` for one-command reproducible setup
- Created `.env` with MPS fallback and seed configuration
- Created `scripts/verify_setup.py` verification script

### Environment details:
- Python: 3.12.13
- PyTorch: 2.11.0
- MPS: Available ✅
- OpenCLIP ViT-B/16: 149.6M params ✅
- Disk free: 35.1GB of 245.1GB

### Files created:
- `.gitignore` — Git ignore rules for data, checkpoints, venv, etc.
- `.env` — Environment variables (gitignored)
- `requirements.txt` — Pinned project dependencies
- `setup.sh` — One-command environment setup script
- `DESIGN.md` — Project plan and design document
- `PROJECT_PLAN.md` — High-level project plan
- `EXPLANATION.md` — This file
- `scripts/verify_setup.py` — Environment verification script
- `src/__init__.py` — Source package init
- `src/models/__init__.py` — Models package init
- `tests/__init__.py` — Tests package init

### Directory structure created:
```
data/{raw,processed/{train,val,test},forensynths}
src/models/
scripts/
backend/
frontend/src/components/
notebooks/
checkpoints/backups/
results/{metrics,plots,gradcam_samples,tables}
report/{figures,presentation}
tests/
```

### Commands run:
```bash
git init
git remote add origin https://github.com/Charan0622/AI-Image-Detector.git
python3.12 -m venv .venv
pip install --upgrade pip
pip install grad-cam  # pytorch-grad-cam incompatible with Python 3.12
pip install -r requirements.txt
python scripts/verify_setup.py
```

### Git commit: `init: project scaffold with environment setup`

---

## Phase 1: Data Acquisition & Preprocessing
**Date:** 2026-04-07
**Status:** ✅ Complete

### What was done:
- Downloaded GenImage dataset from HuggingFace (`RohanRamesh/genimage-224`)
- Dataset is already 224×224 resolution — no resizing needed
- All images re-saved as JPEG Q=95 to normalize compression bias
- Created 80/20 stratified train/val split (seed=42)
- Downloaded test subsets for 6 generators (1000 real + 1000 fake each)
- Verified all 252,000 images — 0 invalid, 0 wrong size
- Built PyTorch Dataset classes supporting both RGB and DCT inputs
- Created data exploration notebook

### Data source:
- **Dataset:** `RohanRamesh/genimage-224` on HuggingFace
- **Original source:** GenImage (NeurIPS 2023)
- **License:** CC BY-NC-SA 4.0
- **Why this source:** Already preprocessed to 224×224, available as streaming parquet files (no need to download 50GB+ archives)

### Generators available (6 of 8 from original GenImage):
- ADM, GLIDE, Midjourney, SD v1.5, VQDM, Wukong
- Missing: BigGAN, SD v1.4 (not in this HuggingFace subset)

### Label mapping:
- In source: 0=AI (fake), 1=Nature (real)
- In our code: 0=Real, 1=Fake (standard convention)

### Image counts:
| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | 96,000 | 96,000 | 192,000 |
| Val | 24,000 | 24,000 | 48,000 |
| Test (per gen) | 6×1,000 | 6×1,000 | 12,000 |
| **Total** | **126,000** | **126,000** | **252,000** |

### Disk usage:
- Before download: 29GB free
- Data on disk: ~5.5GB
- After download: ~19GB free

### Files created/modified:
- `scripts/download_data.py` — HuggingFace streaming download + preprocessing
- `scripts/preprocess_data.py` — Verification and manifest generation
- `src/config.py` — Central configuration with all hyperparameters
- `src/seed.py` — Seed-fixing utility
- `src/transforms.py` — CLIP transforms, DCT computation, robustness augmentation
- `src/dataset.py` — AIDetectDataset, AIDetectTestDataset, AIDetectDCTDataset
- `src/utils.py` — Disk check, timer, parameter counting helpers
- `notebooks/01_data_exploration.ipynb` — Data exploration and visualization
- `data/data_manifest.json` — Dataset metadata and counts

### Verification:
```python
# DataLoader test passed:
# RGB shape: torch.Size([4, 3, 224, 224])
# DCT shape: torch.Size([4, 1, 224, 224])
# Labels: tensor([0, 1, 0, 1])
# Generators: ['vqdm', 'sd15', 'glide', 'adm']
```

### Commands run:
```bash
python scripts/download_data.py
python scripts/preprocess_data.py
```

### Git commit: `data: download and preprocess GenImage subsets`

---

## Phase 2: CLIP Linear Probe Baseline
**Date:** 2026-04-07
**Status:** ✅ Complete

### What was done:
- Built CLIP Linear Probe model (frozen ViT-B/16 + Linear(512, 2))
- Pre-extracted CLIP features to disk for fast training (20K train, 8K val, 12K test)
- Trained linear probe on cached features (20 epochs in 8 seconds)
- Evaluated cross-generator performance on all 6 test generators
- Generated results table and saved metrics

### Architecture:
- Frozen CLIP ViT-B/16 image encoder (86M params, all frozen)
- Single linear layer: Linear(512, 2) — **1,026 trainable parameters**
- Total: 86,193,666 params (only 0.001% trainable)

### Key decision — Feature caching:
Running CLIP inference on 192K images each epoch was too slow on MPS (~2.8 hrs/epoch).
Solution: Pre-extract features once (20K subsample), then train linear head on cached
512-dim vectors. Training completed in 8 seconds instead of 56+ hours.

### Training results:
- Best val AUC: **0.9458**
- Best val accuracy: **87.14%**
- Training time: 8 seconds (20 epochs)

### Cross-Generator Results:
| Generator | Accuracy | AUC | F1 |
|-----------|----------|------|----|
| ADM | 0.8920 | 0.9586 | 0.8948 |
| GLIDE | 0.9415 | 0.9949 | 0.9444 |
| Midjourney | 0.8525 | 0.9299 | 0.8485 |
| SD v1.5 | 0.9060 | 0.9676 | 0.9075 |
| VQDM | 0.7975 | 0.8936 | 0.7835 |
| Wukong | 0.8695 | 0.9427 | 0.8690 |
| **Average** | **0.8765** | **0.9479** | |

### Analysis:
- GLIDE is easiest to detect (AUC 0.9949) — likely has strong frequency artifacts
- VQDM is hardest (AUC 0.8936) — possibly more diverse generation patterns
- Average cross-gen AUC of 0.9479 is strong for a simple linear probe
- This confirms CLIP features contain significant signal for AI detection

### Files created/modified:
- `src/models/clip_probe.py` — CLIP Linear Probe model
- `src/models/model_zoo.py` — Model registry
- `src/train.py` — Generic training loop (for full-model training)
- `src/train_probe.py` — Fast probe training on cached features
- `src/evaluate.py` — Cross-generator evaluation suite
- `scripts/extract_features.py` — CLIP feature extraction to disk

### Git commit: `feat: CLIP linear probe baseline with cross-gen evaluation`

---

## Phase 3: AIDE-Style Hybrid Detector
**Date:** 2026-04-08
**Status:** ✅ Complete

### What was done:
- Built AIDE-style hybrid detector: CLIP features + DCT frequency CNN + fusion MLP
- Used cached CLIP features + live DCT computation for efficient training
- Trained for 30 epochs (~4.3 min/epoch on MPS)
- Evaluated cross-generator performance

### Architecture:
- Branch 1: Frozen CLIP ViT-B/16 -> 512-dim (reused cached features)
- Branch 2: FrequencyCNN(1->32->64->128->256, GAP) -> 256-dim
- Fusion: Concat(512+256=768) -> Linear(768,256) -> GELU -> Dropout(0.3) -> Linear(256,2)
- Trainable parameters: **651,970**

### Training results:
- Best val AUC: **0.9822** (vs 0.9458 for CLIP probe — +3.6%)
- Best val accuracy: **93.14%**
- Training time: ~2 hours (30 epochs)

### Cross-Generator Results (Hybrid vs CLIP Probe):
| Generator | Hybrid Acc | Hybrid AUC | Probe AUC | Delta |
|-----------|-----------|------------|-----------|-------|
| ADM | 0.9520 | 0.9929 | 0.9586 | +0.0343 |
| GLIDE | 0.9585 | 0.9955 | 0.9949 | +0.0006 |
| Midjourney | 0.8990 | 0.9695 | 0.9299 | +0.0396 |
| SD v1.5 | 0.9500 | 0.9883 | 0.9676 | +0.0207 |
| VQDM | 0.9110 | 0.9707 | 0.8936 | +0.0771 |
| Wukong | 0.9270 | 0.9771 | 0.9427 | +0.0344 |
| **Average** | **0.9329** | **0.9823** | **0.9479** | **+0.0344** |

### Analysis:
- Hybrid improves on CLIP probe across ALL generators
- Biggest improvement on VQDM (+7.7% AUC) — frequency features help most on harder cases
- GLIDE barely improved (+0.06%) — already near-perfect with CLIP alone
- Frequency branch adds meaningful signal, especially for generators with strong spectral artifacts

### Files created:
- `src/models/hybrid.py` — HybridDetector with FrequencyCNN
- `src/train_hybrid.py` — Fast hybrid training on cached features + live DCT

### Git commit: `feat: AIDE-style hybrid detector baseline 2`

---

## Phase 4: Improvements + Ablations
**Date:** 2026-04-08
**Status:** ✅ Complete

### What was done:
- Built Frequency-Guided Detector with multi-scale freq CNN + gated fusion
- Added robustness augmentation (JPEG compression, blur, resize degradation)
- Trained 3 ablation variants (20 epochs each, early stopping)
- Generated complete ablation study table

### Improvement 1: Frequency-Guided Attention
- Multi-scale FrequencyCNN: extracts features at 3 spatial scales (56×56, 28×28, 14×14)
- Spatial attention module: learns which frequency regions matter most
- Gated fusion: dynamically weights CLIP vs frequency contributions
- 1,670,341 trainable parameters

### Improvement 2: Robustness-Aware Training
- Random JPEG compression (Q=50-100) with 50% probability
- Random Gaussian blur (σ=0.1-2.0) with 30% probability
- Random downscale+upscale (112→224) with 30% probability
- Applied during training only

### Ablation Study Results:
| # | Model Variant | Cross-Gen Avg AUC | Cross-Gen Avg Acc | vs Probe |
|---|---------------|-------------------|-------------------|----------|
| 1 | CLIP Linear Probe | 0.9479 | 0.8765 | baseline |
| 2 | AIDE-style Hybrid | 0.9823 | 0.9329 | +0.0344 |
| 3 | Freq-Guided (full) | 0.9731 | 0.9131 | +0.0252 |
| 4 | Freq-Guided (no robust) | 0.9760 | 0.9259 | +0.0281 |
| 5 | Hybrid + Robustness | 0.9816 | 0.9291 | +0.0337 |

### Analysis:
- All models significantly outperform the CLIP linear probe baseline
- The AIDE-style Hybrid (simple concat) remains competitive (AUC 0.9823)
- Freq-guided attention provides a different fusion mechanism but doesn't surpass simple concat
- Robustness augmentation has mixed effects — helps generalization but can reduce clean accuracy
- The frequency branch is the key improvement regardless of fusion method

### Files created:
- `src/models/freq_guided.py` — MultiScaleFreqCNN, GatedFusion, FreqGuidedDetector
- `src/train_freq_guided.py` — Training with ablation support

### Git commit: `feat: freq-guided attention + robustness training + ablation study`

---

## Phase 5: Grad-CAM Explainability
**Date:** 2026-04-08
**Status:** ✅ Complete

### What was done:
- Built Grad-CAM pipeline for frequency branch visualization
- Built frequency spatial attention map extraction
- Created text explanation generator with spectral analysis
- Generated Grad-CAM visualizations for all 6 generators (3 real + 3 fake each)
- Created summary panel with side-by-side real vs fake heatmaps

### Explainability approach:
1. **Grad-CAM on frequency CNN**: Gradient-weighted activations from last conv layer
   show which spatial frequency regions drive the fake/real classification
2. **Frequency spatial attention**: Direct attention weights from MultiScaleFreqCNN
3. **Text explanations**: Automated analysis of DCT spectrum for artifact signatures
   (frequency rolloff, grid artifacts, spectral entropy)

### Output files:
- `results/gradcam_samples/gradcam_{gen}.png` — Per-generator visualizations (6 files)
- `results/gradcam_samples/gradcam_summary.png` — Combined summary panel

### Files created:
- `src/gradcam_utils.py` — Grad-CAM computation, heatmap overlay
- `src/explain.py` — Text explanation generation with spectral analysis
- `scripts/generate_gradcam_samples.py` — Batch visualization generator

### Git commit: `feat: Grad-CAM explainability + text explanations`

---

## Phase 6: Web Application
**Date:** 2026-04-08
**Status:** ✅ Complete

### What was done:
- Built FastAPI backend with detection, comparison, and dashboard endpoints
- Built single-page React frontend with CDN-loaded dependencies (no npm install needed)
- Tested all endpoints: health, detect, compare, dashboard

### Backend (FastAPI):
- `POST /detect` — Single image detection with verdict, confidence, heatmap, explanations
- `POST /detect/compare` — Side-by-side comparison of all 3 models
- `GET /dashboard/data` — Cross-generator metrics and ablation results
- `GET /health` — Health check
- ModelManager class with lazy model loading and caching
- Port: 8001

### Frontend (React + Tailwind via CDN):
- **UploadZone**: Drag-and-drop image upload with file validation
- **VerdictCard**: Large verdict display with animated confidence meter
- **HeatmapOverlay**: Toggle between original image and Grad-CAM overlay
- **ExplanationPanel**: Accordion list of region-specific explanations with severity bars
- **ComparativePanel**: Side-by-side cards for all 3 models
- **Dashboard**: Cross-generator accuracy table + ablation study results
- Single HTML file — no build step, no node_modules

### How to run:
```bash
bash scripts/start_app.sh
# Then open frontend/index.html in browser
```

### Files created:
- `backend/main.py` — FastAPI application
- `backend/inference.py` — Model loading + prediction pipeline
- `backend/requirements.txt` — Backend-specific deps
- `frontend/index.html` — Single-page React app
- `scripts/start_app.sh` — One-command app launcher

### Git commit: `feat: full-stack web application`

---

## Phase 4b: Robustness Evaluation Follow-up
**Date:** 2026-04-22
**Status:** ✅ Complete

### Motivation
Phase 4 claimed "robustness-aware training helps" but the ablation table only
reported clean cross-generator accuracy. We never actually measured robustness.
This phase closes that gap: 8-degradation evaluation across all 5 model variants
and all 6 test generators.

### What was done
- Discovered that trained checkpoints use "FromFeatures" heads
  (`LinearProbeHead`, `HybridFromFeatures`, `HybridRobustFromFeatures`,
  `FreqGuidedFromFeatures`) that consume pre-extracted CLIP features — not the
  full-image architectures in `src/models/`. `src/evaluate.py` was wired to the
  full-image classes and would have failed to load the real checkpoints.
- Wrote `scripts/run_all_robustness.py` with a **shared-CLIP design**:
  for each (generator, degradation) pair, extract CLIP features + DCT maps
  once, then run all 5 tiny heads on the cached tensors. Reduces total runtime
  from ~3.5 hrs (naive 5× pass) to ~60 min on MPS.
- Wrote `scripts/generate_plots.py` — emits training curves, cross-gen heatmap
  + grouped bars, per-model robustness curves, combined robustness overlay, and
  ablation summary.
- Wrote `scripts/update_ablation_table.py` — regenerates `ablation_table.md`
  with clean + robustness breakdown.
- Ran full sweep: **5 models × 6 generators × 300 images/class × 8 degradations
  = ~72,000 total image passes**. Total wall time: 61.6 min.

### Results — final ablation table

| Model | Clean Acc | Clean AUC | JPEG-30 Acc | Blur-σ3 Acc | Resize Acc | Robust AUC (avg) | Robust Acc (avg) |
|-------|-----------|-----------|-------------|-------------|------------|------------------|------------------|
| CLIP Linear Probe | 0.8882 | 0.9558 | 0.7225 | 0.5225 | 0.7078 | 0.8360 | 0.6487 |
| AIDE-style Hybrid | 0.9652 | 0.9944 | 0.7483 | 0.5420 | 0.7814 | 0.8491 | 0.7112 |
| **Hybrid + Robust Aug** | 0.9619 | 0.9940 | 0.7228 | 0.5936 | 0.8025 | **0.8835** | **0.7254** |
| FreqGuided (no robust) | 0.9561 | 0.9911 | 0.7208 | 0.6086 | 0.8036 | 0.8520 | 0.7251 |
| FreqGuided (full) | 0.9533 | 0.9904 | 0.6644 | 0.5906 | 0.7755 | 0.8296 | 0.6917 |

### Key findings

1. **Hybrid + Robustness Aug wins on robustness** (0.8835 Robust AUC). It
   beats the "final" freq-guided model by **+0.054 AUC** on average under
   degradation — a large margin.
2. **FreqGuided (full) is the worst robustness model**, below even the simple
   CLIP linear probe (0.830 vs 0.836). The combination of a freq-guided
   attention architecture + aggressive robustness augmentation under-performs
   either change alone.
3. **AIDE-style Hybrid collapses on blur** — AUC drops from 0.994 clean to
   0.636 on blur σ=3 (−0.36). Adding robustness aug recovers it to 0.740.
4. **CLIP Linear Probe is surprisingly blur-robust** — its semantic features
   degrade gracefully. Hybrids that rely on high-frequency signal are hit
   hardest by blur.
5. All models recover on JPEG — the training data normalizes JPEG (Q=95) so
   the models are implicitly exposed to compression artifacts.

### The freq_guided puzzle

Two architecturally similar variants diverge under degradation:
- `FreqGuided (no robust)` — Robust AUC 0.852
- `FreqGuided (full, with robust aug)` — Robust AUC 0.830

Adding robustness augmentation *hurts* when combined with the freq-guided
attention architecture. Two possible explanations (to investigate):
- The freq-guided attention already learns invariance that the augmentation
  teaches, so the augmentation only adds label noise.
- The attention mechanism may over-focus on spurious frequency patterns in
  blurred/recompressed images when trained with heavy augmentation.

This is a **real negative finding** for the report: architectural inductive bias
and data augmentation are not additive; combining them double-counts and can
degrade generalization.

### Deliverables

- `results/metrics/{model}_robustness.json` — 5 JSONs, per-gen × per-deg
  accuracy & AUC
- `results/tables/ablation_table.md` — updated with Robust columns + per-deg
  AUC breakdown
- `results/plots/` — 11 PNGs:
  - `training_curves.png`
  - `cross_gen_heatmap.png`, `cross_gen_bars.png`
  - `ablation_summary.png`
  - `robustness_{clip_probe,hybrid,hybrid_robust,freq_guided_no_robust,freq_guided}.png`
  - `robustness_all.png` (accuracy + AUC side-by-side, all models overlaid)

### Files created
- `scripts/run_all_robustness.py` — shared-CLIP robustness driver
- `scripts/generate_plots.py` — all plot generators
- `scripts/update_ablation_table.py` — ablation table regenerator

### Recommendation

For deployment: **switch the default model in the web app to `hybrid_robust`**.
It is the most robust under real-world degradations, matches the full freq-guided
model on clean data (0.994 vs 0.990 AUC), and is architecturally simpler.
The "final" model should be renamed for the report to reflect this.

### Git commit: `eval: full robustness sweep + plots + updated ablation table`

---

## Phase 4c: Model Quality Improvements (Tier 2)
**Date:** 2026-04-22
**Status:** ✅ Complete

### Motivation
With the robustness picture clear, we explored three standard techniques for
squeezing more performance from the existing checkpoints without retraining:
temperature-scaling calibration, model ensembling, and test-time augmentation.

### 1. Temperature Scaling Calibration (Guo et al., ICML 2017)

Fit a scalar T > 0 per model such that softmax(logits / T) minimizes NLL on a
held-out val subset (4000 images, balanced). Applied at inference only —
argmax is T-invariant so verdicts don't change, only confidence values do.

**Results** (`results/metrics/calibration.json`):

| Model | T | NLL before → after | ECE before → after |
|-------|---|--------------------|--------------------|
| clip_probe | 0.87 | 0.2316 → 0.2294 | 0.017 → 0.008 |
| **hybrid** | **2.45** | 0.0876 → 0.0581 | 0.015 → 0.004 |
| hybrid_robust | 1.73 | 0.0734 → 0.0615 | 0.013 → 0.005 |
| freq_guided_no_robust | 1.90 | 0.0974 → 0.0761 | 0.014 → 0.007 |
| freq_guided | 1.37 | 0.1118 → 0.1057 | 0.013 → 0.005 |

- ECE drops 2-4× across the board.
- `hybrid` is **massively overconfident** (T=2.45) — a confidence of 0.99
  was really ~0.75 honest probability. `hybrid_robust` is more calibrated
  from the start (T=1.73) — another signal it's the healthier model.
- Temperatures wired into `backend/inference.py`; loaded from
  `calibration.json` at startup and applied per-model before softmax.

### 2. Ensemble Evaluation

Three ensembles, all using temperature-calibrated probabilities:
- `ensemble_all` — equal-weight mean of all 5 heads
- `ensemble_top3` — mean of hybrid_robust + freq_guided_no_robust + clip_probe
  (most architecturally diverse subset)
- `ensemble_weighted` — softmax-weighted mean by val-AUC (exp(50*(auc - max_auc)))

**Results** (`results/tables/ensemble_comparison.md`):

| Model | Clean AUC | Robust AUC |
|-------|-----------|------------|
| **hybrid_robust** (single) | 0.9936 | **0.8908** ← best |
| hybrid (single) | 0.9943 | 0.8835 |
| **ensemble_weighted** | **0.9947** | 0.8754 |
| ensemble_top3 | 0.9913 | 0.8765 |
| ensemble_all | 0.9936 | 0.8748 |

**Finding — ensembles don't beat the single best model.** On clean data,
`ensemble_weighted` ties `hybrid` (0.9947 vs 0.9943 — within noise). Under
robustness, **every ensemble underperforms `hybrid_robust` by 1.4–1.6 AUC
points**. See `results/plots/robustness_ensembles.png`.

Root cause: all 5 models share the frozen CLIP ViT-B/16 backbone, so their
errors are strongly correlated. Averaging decorrelated predictions is what
makes ensembles work — we don't have decorrelated predictions, we have
correlated errors on CLIP's failure modes. Mixing in the weaker models
(freq_guided full, clip_probe) just drags the ensemble toward their error
distribution.

This validates the deployment choice: single `hybrid_robust` is the right model.

### 3. Test-Time Augmentation (TTA)

Horizontal-flip TTA: logits = 0.5 × (head(features_orig) + head(features_hflip)).
Evaluated on the two best models (hybrid_robust, freq_guided_no_robust) across
all 6 generators × {clean, jpeg_q30, blur_s3, resize_112}.

**Results** (`results/tables/tta_comparison.md`):

| Model | No-TTA avg AUC | TTA avg AUC | Δ AUC |
|-------|----------------|-------------|-------|
| hybrid_robust | 0.8960 | 0.8971 | +0.0011 |
| freq_guided_no_robust | 0.8740 | 0.8752 | +0.0012 |

**Finding — TTA gives ~0.001 AUC for 2× inference cost. Not worth it.**
Near-zero on both models, below the per-generator noise floor (≈ ±0.005).

Why this makes sense:
- For `hybrid_robust`: the robustness augmentation already teaches similar
  invariances, so hflip TTA is largely redundant.
- For `freq_guided_no_robust`: operates on DCT frequency features, which are
  approximately invariant to horizontal flip anyway.

Decision: **TTA not deployed**. 2× cost for noise-level gain.

### Summary of Tier 2

| Technique | Effect | Decision |
|-----------|--------|----------|
| Temperature scaling | ECE ↓ 2-4×, honest confidences | **Deployed** in backend |
| Ensembling | Robust AUC ↓ 1.5 points vs best single | **Rejected** |
| TTA (hflip) | +0.001 AUC for 2× cost | **Rejected** |

### Deliverables
- `scripts/fit_temperature.py` — per-model T fitting on val set
- `scripts/run_ensemble_eval.py` — shared-CLIP ensemble driver
- `scripts/run_tta_eval.py` — TTA driver (focused degradation subset)
- `results/metrics/calibration.json` — 5 temperatures + NLL/ECE before/after
- `results/metrics/ensemble_cross_gen.json`, `ensemble_robustness.json`
- `results/metrics/tta_hybrid_robust.json`, `tta_freq_guided_no_robust.json`
- `results/tables/ensemble_comparison.md`, `tta_comparison.md`
- `results/plots/robustness_ensembles.png`
- `backend/inference.py` — loads temperatures, applies calibrated softmax,
  supports hybrid_robust and freq_guided_no_robust variants
- `backend/main.py` — adds hybrid_robust and freq_guided_no_robust to
  endpoints; default switched from freq_guided → hybrid_robust

### Git commit: `eval: temperature calibration + ensemble + TTA (Tier 2)`

---

## Phase 5: Real-World Deployment (Iterative Phoenix)
**Date:** 2026-04-29
**Status:** ✅ Complete

### Motivation
Cross-generator AUC of 0.994 sounded great, but the model in production
mis-classified obvious smartphone photos as AI-generated. We curated a
117-image real-world test set (100 picsum smartphone-style + 17
Pollinations modern AI) and measured a **44 % real-photo FPR** for
`hybrid_robust` on this held-out set. The benchmark hides a substantial
deployment gap.

### What was done
- **Phase 0** — `scripts/build_realworld_eval.py` curates the held-out
  set; `scripts/eval_realworld.py` runs each model through the live
  inference pipeline and reports per-subset accuracy / FPR / band breakdown.
- **Phase 1** — `SmartphoneAesthetic` augmentation (PIL ColorJitter,
  random gamma, sensor read-noise, ±1 px chromatic aberration) added in
  `src/transforms.py`. `RobustnessAugmentation` gains a double-JPEG
  path; JPEG quality range extended to (35, 100).
- **Phase 2** — `scripts/expand_training_data.py` downloads 743 picsum
  photos in parallel (asyncio + aiohttp, 45 it/s vs 1 it/s sequential),
  canonicalises each through LANCZOS-224 + JPEG Q=95, extracts CLIP
  features, appends to `data/features/train_features.npy` and
  `val_features.npy`. Path files written for the v2 trainer.
- **Phase 3** — `src/train_hybrid_robust_v2.py` warm-starts from
  `hybrid_robust_best.pth`, fine-tunes 3 epochs at LR=1e-4 with the new
  augmentations on the expanded ~193 K-sample training set. Best val
  AUC: 0.9935. Output: `checkpoints/hybrid_robust_v2_best.pth`.
- **Phase 4** — Recalibrated temperature on v2 (T = 1.75). Added a
  fourth verdict band (`Likely Real`) for borderline real cases.
  Lowered OOD threshold 0.7 → 0.55 (then later raised to 0.85 after
  empirical observation).

### Real-world results

| Model | Overall acc | Real-photo FPR |
|-------|------------:|---------------:|
| CLIP Linear Probe | 0.521 | 55 % |
| AIDE Hybrid | 0.590 | 48 % |
| Hybrid + Robust (v1) | 0.624 | 44 % |
| **Hybrid + Robust v2** | **0.889** | **7 %** |

A **+27 point** accuracy jump, **−37 points** FPR. On the 100 picsum
real photos, v2 routes 75 to `Real`, 5 to `Likely Real`, 17 to
`Inconclusive`, with only 3 false-positive AI calls (down from 36 on v1).
GenImage val AUC drift: 0.9940 → 0.9935 (within noise).

### Files created
- `scripts/build_realworld_eval.py`, `scripts/eval_realworld.py`
- `scripts/expand_training_data.py`, `scripts/expand_v3.py`
- `src/train_hybrid_robust_v2.py`
- `data/realworld_eval/` (117 images + manifest.csv)
- `results/metrics/realworld_baseline.json`, `realworld_v2.json`
- `results/plots/realworld_improvement.png`
- `checkpoints/hybrid_robust_v2_best.pth`

### v3 attempt
A larger expansion (3000 picsum + intended diffusiondb modern AI) was
attempted. diffusiondb's HF script-based dataset is no longer supported,
so v3 fell back to picsum-only. Training was killed at 65 % of one
epoch due to MPS thermal throttling (1.3 it/s, 4+ hours estimated).
v3 checkpoint exists but is not materially better than v2; not deployed.

### Git commits
- `dfb30f1` feat(aug): smartphone aesthetic + double-JPEG + scripts for real-world eval
- `4d3a549` feat: hybrid_robust_v2 — real-world FPR 44% → 7%
- `1735cb5` fix(frontend): request hybrid_robust_v2 from /detect

---

## Phase 6: Honest Inference Logic + Public-Detector Fallback
**Date:** 2026-05-03
**Status:** ✅ Complete

### Problem
Even with v2, user uploads of phone photos sometimes saturated to
p(AI) ≈ 1.0. The trained heads have no representation for camera
signatures outside our training set. Threshold tweaking traded one wrong
verdict for another.

### Two changes
1. **Prior-toward-Real in OOD region** (`backend/inference.py`). When
   `ood_score ≥ 0.40`, the head's confident `p(AI)` is treated as
   unreliable; the verdict defaults to `Likely Real` with explicit
   reason "head is unreliable on this input — defaulted to prior". The
   deployment prior P(real) ≫ P(AI) wins in OOD territory.
2. **External-detector fallback** (`backend/external_detector.py`).
   Wired in `haywoodsloan/ai-image-detector-deploy` (HuggingFace
   transformers, ViT-base trained on a broader corpus). On a 60-image
   spot-check it scores **100 %** on the two failure modes our custom
   heads have (smartphone reals + modern AI). Made the deployment
   default; our 5 custom heads remain available in the comparison panel.

### Result
Real photos correctly read as `Real` or `Likely Real`. Modern AI from
generators we never trained on correctly read as `AI-Generated`. The
custom heads serve the research narrative (clean GenImage benchmark,
robustness ablation, calibration story); the public detector serves
production reliability.

### Files created
- `backend/external_detector.py` — wraps the HF pipeline, lazy-load
  singleton, returns the same response shape as the custom heads.

### Git commits
- `7f670ba` fix: prior-toward-Real in OOD region
- `c00673f` feat: v3 expansion scaffolding (`scripts/expand_v3.py` + `src/train_hybrid_robust_v2.py` v3 variant)
- `7a0519e` feat: integrate haywoodsloan/ai-image-detector-deploy as default

---

## Phase 7: Apple HIG Frontend Redesign
**Date:** 2026-05-03
**Status:** ✅ Complete

### Motivation
Earlier UI iterations drifted between aesthetics that didn't land:
generic SaaS dashboard, monochrome editorial, instrument-readout. The
final brief asked for a premium minimalist Apple-inspired interface.

### What landed
- **Light + dark adaptive** via `prefers-color-scheme`; manual override
  via Settings sheet (segmented control: Auto / Light / Dark).
- **SF Pro stack** (-apple-system / SF Pro Text / SF Pro Display) with
  tight tracking on display sizes.
- **Glassmorphism top nav** (sticky, `backdrop-filter saturate(180%) blur(28px)`)
  with logo + History / About / Settings buttons.
- **Hero**: pill, large display headline, subdued subhead, breathing-glow
  upload card with drag-morph state, privacy disclosure underneath, "How
  it works" accordion.
- **Analysis view**: two-column desktop (1.2fr / 1fr), stacked mobile.
  Image card with smooth heatmap toggle and side-by-side comparison
  slider with draggable handle. Verdict card with 64 px count-up
  percentage (0 → target in 900 ms ease-out cubic), gradient
  probability bar with soft accent glow, three-stat detail row,
  metadata accordion (EXIF), evidence accordion, Second Opinions panel.
- **Sheets**: History (timeline with thumbnails, persisted to
  localStorage), About (headline metrics), Settings (theme + default
  model picker), Export (copy summary / download JSON).
- **Motion**: Apple's signature curve `cubic-bezier(0.32, 0.72, 0, 1)`
  on every transition. Durations ≤ 320 ms. Spring keyframe (0.94 →
  1.015 → 1) for verdict reveal. Shimmer sweep for loading (no spinners).

### Trust signals
- Privacy note under upload card.
- OOD-driven Inconclusive band.
- About sheet shows 0.994 cross-gen AUC and 93 % real-photo accuracy.

### Git commits
- `259b817` fix: ensemble-fallback when OOD ≥ 0.5 + wider verdict bands + Likely AI band
- `6ae3f50` ui: Apple HIG redesign — premium minimalist interface

---

## Phase 8: Final Report
**Date:** 2026-05-03
**Status:** ✅ Complete

Wrote `report/final_report.md` (~3,600 words, 10 sections + 2 appendices)
covering the CMPE 258 rubric: abstract, intro, related work, problem
formulation, dataset, SOTA survey, our approach, experiments, web app,
conclusion, references, reproducibility, hardware. All numbers traced
back to results/metrics/ JSON files; honest about limits.
