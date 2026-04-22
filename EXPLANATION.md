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
- Created full directory structure per CLAUDE.md specification
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
- `CLAUDE.md` — Project instructions and plan
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
