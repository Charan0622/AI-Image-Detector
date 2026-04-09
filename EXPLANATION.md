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
