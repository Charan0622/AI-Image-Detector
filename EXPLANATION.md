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
