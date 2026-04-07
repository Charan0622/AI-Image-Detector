# CLAUDE.md — AI-Generated Image Detection Project

## ⚠️ ABSOLUTE RULES (NEVER VIOLATE)

```
1. NEVER delete any file or directory without explicit user confirmation. Ask first. Always.
2. NEVER modify files outside of ~/aidetect/ project directory.
3. NEVER install packages globally. Everything goes in the project venv.
4. NEVER overwrite a model checkpoint without backing up the previous one first.
5. NEVER run training without confirming disk space first (run `df -h ~` before any training).
6. AFTER every phase completion, update ~/aidetect/EXPLANATION.md with:
   - What was done (every file created/modified)
   - Why each decision was made
   - Any deviations from this plan and why
   - Metrics/results obtained
   - Exact commands run
   - Timestamp of completion
7. BEFORE starting any phase, read this entire file again to stay aligned.
8. ASK the user before proceeding to the next phase. Never auto-advance.
9. If ANY error occurs, log it in EXPLANATION.md and ask the user before retrying.
10. Keep terminal output clean. Use `tqdm` for progress bars in all loops.
11. Git commit after every phase with a meaningful message.
12. NEVER use `rm -rf`. Only use `rm` on specific files with user permission.
13. All Python files must have docstrings, type hints, and be formatted with black.
14. Fix random seeds EVERYWHERE: torch, numpy, random, PYTHONHASHSEED.
15. Run `df -h ~` before and after downloading anything.
```

---

## PROJECT OVERVIEW

**Goal:** Build an end-to-end AI-generated image detector that:
- Takes any image as input
- Outputs: Real/Fake verdict, confidence score, Grad-CAM heatmap showing which regions look AI, text explanation of why, and a comparative panel showing predictions from multiple models
- Generalizes across generators never seen during training
- Includes a full evaluation dashboard with cross-generator accuracy tables and robustness charts

**Hardware:** MacBook Air M5, 16GB RAM, 15GB free disk, MPS backend
**Compute Offload:** Google Colab free tier (T4 GPU) for heavy training
**Course:** CMPE 258 Deep Learning, Spring 2026, SJSU

---

## DIRECTORY STRUCTURE (Create This First)

```
~/aidetect/
├── .git/
├── .gitignore
├── CLAUDE.md                    # This file (copy here)
├── EXPLANATION.md               # Running log of everything done
├── README.md                    # Project documentation for reproducibility
├── requirements.txt             # Pinned dependencies
├── setup.sh                     # One-command environment setup
├── .env                         # Environment variables (gitignored)
│
├── data/                        # ALL data (gitignored)
│   ├── raw/                     # Downloaded originals
│   ├── processed/               # Resized 224x224, normalized JPEG Q=95
│   │   ├── train/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   ├── val/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── test/
│   │       ├── sdv14/
│   │       │   ├── real/
│   │       │   └── fake/
│   │       ├── sdv15/
│   │       ├── midjourney/
│   │       ├── adm/
│   │       ├── glide/
│   │       ├── biggan/
│   │       ├── wukong/
│   │       └── vqdm/
│   └── forensynths/             # Cross-dataset eval
│
├── src/
│   ├── __init__.py
│   ├── config.py                # All hyperparameters, paths, seeds in ONE place
│   ├── seed.py                  # Seed-fixing utility
│   ├── dataset.py               # Dataset classes + DataLoaders
│   ├── transforms.py            # All augmentations + DCT extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_probe.py        # Phase 2: CLIP linear probe baseline
│   │   ├── hybrid.py            # Phase 3: AIDE-style hybrid baseline
│   │   ├── freq_guided.py       # Phase 4: Final model with freq-guided attention
│   │   └── model_zoo.py         # Registry to load any model by name
│   ├── train.py                 # Training loop (generic, works for all models)
│   ├── evaluate.py              # Cross-generator evaluation + metrics
│   ├── gradcam_utils.py         # Grad-CAM / Attention Rollout for ViTs
│   ├── explain.py               # Text explanation generator
│   └── utils.py                 # Misc helpers (logging, timing, disk check)
│
├── scripts/
│   ├── download_data.py         # Dataset downloader with progress + disk checks
│   ├── preprocess_data.py       # Resize, normalize, split
│   ├── extract_features.py      # Pre-extract CLIP features to disk (saves RAM)
│   ├── run_ablations.py         # Run all ablation experiments
│   └── generate_tables.py       # Generate LaTeX/markdown results tables
│
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── inference.py             # Model loading + prediction pipeline
│   ├── gradcam_api.py           # Grad-CAM endpoint logic
│   ├── comparative.py           # Multi-model comparison endpoint
│   └── requirements.txt         # Backend-specific deps
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── App.jsx              # Main layout
│   │   ├── components/
│   │   │   ├── UploadZone.jsx       # Drag-and-drop image upload
│   │   │   ├── VerdictCard.jsx      # Real/Fake + confidence meter
│   │   │   ├── HeatmapOverlay.jsx   # Grad-CAM visualization
│   │   │   ├── ExplanationPanel.jsx # "Why it's AI" reasons
│   │   │   ├── ComparativePanel.jsx # Side-by-side baseline vs improved
│   │   │   └── Dashboard.jsx       # Evaluation dashboard
│   │   │       ├── AccuracyTable.jsx    # Cross-generator accuracy matrix
│   │   │       ├── RobustnessChart.jsx  # Degradation robustness curves
│   │   │       └── QualitativeGrid.jsx  # Side-by-side image comparisons
│   │   └── assets/
│   └── public/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_experiments.ipynb
│   ├── 03_ablation_studies.ipynb
│   └── colab_training.ipynb     # Upload this to Colab for heavy training
│
├── checkpoints/                 # Model weights (gitignored)
│   ├── clip_probe_best.pth
│   ├── hybrid_best.pth
│   ├── freq_guided_best.pth
│   └── backups/                 # Auto-backup before overwrite
│
├── results/
│   ├── metrics/                 # JSON files with all eval metrics
│   ├── plots/                   # Generated charts and figures
│   ├── gradcam_samples/         # Sample Grad-CAM visualizations
│   └── tables/                  # Generated markdown/LaTeX tables
│
├── report/
│   ├── final_report.md          # Or .tex if preferred
│   ├── figures/
│   └── presentation/            # Slides for class presentation
│
└── tests/
    ├── test_dataset.py          # Verify data loading
    ├── test_models.py           # Verify forward pass shapes
    └── test_inference.py        # Verify API endpoint
```

---

## PHASE 0: ENVIRONMENT SETUP

**Goal:** Create isolated venv, install all dependencies, verify MPS, initialize git repo.

### Step-by-step instructions:

```bash
# 0.1 — Create project directory
mkdir -p ~/aidetect
cd ~/aidetect

# 0.2 — Initialize git FIRST
git init
cat > .gitignore << 'EOF'
data/
checkpoints/
*.pth
*.pt
__pycache__/
*.pyc
.env
node_modules/
dist/
.DS_Store
*.egg-info/
wandb/
frontend/node_modules/
EOF
git add .gitignore
git commit -m "init: project scaffold with .gitignore"

# 0.3 — Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 0.4 — Create requirements.txt with PINNED versions
cat > requirements.txt << 'EOF'
torch>=2.4.0
torchvision>=0.19.0
torchaudio>=2.4.0
open-clip-torch>=2.26.0
timm>=1.0.0
numpy>=1.26.0
scipy>=1.13.0
Pillow>=10.3.0
scikit-learn>=1.5.0
matplotlib>=3.9.0
seaborn>=0.13.0
pandas>=2.2.0
tqdm>=4.66.0
pytorch-grad-cam>=1.5.0
fastapi>=0.111.0
uvicorn>=0.30.0
python-multipart>=0.0.9
aiofiles>=23.0.0
requests>=2.32.0
gdown>=5.1.0
black>=24.0.0
pytest>=8.0.0
EOF

# 0.5 — Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 0.6 — Create .env file
cat > .env << 'EOF'
PYTORCH_ENABLE_MPS_FALLBACK=1
PYTHONHASHSEED=42
PROJECT_ROOT=~/aidetect
EOF

# 0.7 — Create all directories
mkdir -p data/{raw,processed/{train/{real,fake},val/{real,fake},test/{sdv14,sdv15,midjourney,adm,glide,biggan,wukong,vqdm}},forensynths}
mkdir -p src/models
mkdir -p scripts
mkdir -p backend
mkdir -p frontend/src/components
mkdir -p notebooks
mkdir -p checkpoints/backups
mkdir -p results/{metrics,plots,gradcam_samples,tables}
mkdir -p report/{figures,presentation}
mkdir -p tests

# 0.8 — Create all __init__.py files
touch src/__init__.py src/models/__init__.py tests/__init__.py
```

### Verification script (run this and paste output):

```python
# save as scripts/verify_setup.py
"""Verify that the environment is correctly configured."""
import sys
import torch
import platform

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Platform: {platform.platform()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(2, 3, 224, 224, device=device)
    print(f"MPS tensor test: {x.shape} on {x.device} ✅")
else:
    print("⚠️ MPS not available — will fall back to CPU")

import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
print(f"OpenCLIP ViT-B/16 loaded ✅ — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Disk check
import shutil
total, used, free = shutil.disk_usage("/")
print(f"Disk: {free/1e9:.1f}GB free of {total/1e9:.1f}GB total")

print("\n🟢 Environment setup complete!")
```

### After Phase 0, update EXPLANATION.md:

```markdown
# EXPLANATION.md — Project Log

## Phase 0: Environment Setup
**Date:** [timestamp]
**Status:** ✅ Complete

### What was done:
- Created ~/aidetect/ project directory
- Initialized git repository
- Created Python 3.12 venv at ~/aidetect/.venv
- Installed all dependencies (see requirements.txt)
- Verified MPS backend is available
- Verified OpenCLIP ViT-B/16 loads successfully
- Created full directory structure

### Environment details:
- Python: [version]
- PyTorch: [version]
- MPS: [available/not]
- Disk free: [X]GB

### Files created:
[list every file]

### Git commit: `init: project scaffold with .gitignore`
```

### ✅ CHECKPOINT: Ask user "Phase 0 complete. Ready for Phase 1 (Data)?"

---

## PHASE 1: DATA ACQUISITION & PREPROCESSING

**Goal:** Download GenImage SD v1.4 subset + all test splits, preprocess to 224×224, normalize JPEG compression, create train/val split, verify data integrity.

### Critical constraints:
- **Disk budget:** Max 4GB for all data combined
- **Download only:** SD v1.4 train split (for training) + ALL generator val splits (for cross-gen eval)
- **Do NOT** download the full GenImage dataset

### Step 1.1 — Download script

Create `scripts/download_data.py`:

```python
"""
Download GenImage dataset subsets.

Downloads ONLY:
- SD v1.4 train split (for training our models)
- All 8 generator val/test splits (for cross-generator evaluation)
- ForenSynths test set (for cross-dataset evaluation)

Checks disk space before every download.
Never deletes anything without user confirmation.
"""
```

**Data sources (in priority order):**
1. Google Drive mirror of GenImage (check Harvard Dataverse page for link)
2. Hugging Face datasets (search for "genimage" — some community uploads exist)
3. Kaggle "tiny-genimage" as absolute fallback (~1GB)

**If full GenImage subsets are too large:**
- Download only 20,000 images per class (real/fake) for training
- All test images are small enough to keep fully

### Step 1.2 — Preprocessing script

Create `scripts/preprocess_data.py`:

```python
"""
Preprocess all downloaded images:
1. Resize to 224x224 (LANCZOS resampling)
2. Re-save as JPEG quality=95 (normalizes compression bias)
3. Create 80/20 train/val split from SD v1.4 training data
4. Organize into processed/ directory structure
5. Generate data_manifest.json with counts and checksums
6. Log everything to EXPLANATION.md
"""
```

**Preprocessing rules:**
- Use `Pillow` for all image operations
- LANCZOS resampling for resize (highest quality)
- JPEG Q=95 for both real and fake (removes compression bias — this is critical per the Unbiased GenImage paper)
- Random seed = 42 for train/val split
- Stratified split: maintain class balance
- Delete raw/ only after user confirms processed/ looks correct
- Generate a `data/data_manifest.json`:

```json
{
  "train": {"real": 16000, "fake": 16000},
  "val": {"real": 4000, "fake": 4000},
  "test": {
    "sdv14": {"real": 1000, "fake": 1000},
    "sdv15": {"real": 1000, "fake": 1000},
    "midjourney": {"real": 1000, "fake": 1000},
    ...
  },
  "image_size": [224, 224],
  "jpeg_quality": 95,
  "preprocessing_date": "...",
  "total_disk_usage_mb": "..."
}
```

### Step 1.3 — Dataset class

Create `src/dataset.py`:

```python
"""
PyTorch Dataset classes for AI-generated image detection.

Classes:
    - AIDetectDataset: Standard RGB dataset with labels
    - AIDetectDCTDataset: Returns both RGB and DCT frequency map
    - AIDetectFeatureDataset: Loads pre-extracted CLIP features from disk

All datasets return:
    - image: Tensor (3, 224, 224) or pre-extracted features
    - dct_map: Tensor (1, 224, 224) — DCT frequency representation
    - label: int (0=real, 1=fake)
    - metadata: dict with generator name, filename, etc.
"""
```

**DCT extraction (critical — this is the frequency branch input):**

```python
def compute_dct_map(image_pil):
    """
    Convert PIL image to 2D DCT spectral map.

    Steps:
    1. Convert to grayscale
    2. Apply 2D DCT (scipy.fftpack.dct)
    3. Take absolute value + log1p for dynamic range compression
    4. Normalize to [0, 1]
    5. Return as single-channel tensor

    Returns:
        Tensor of shape (1, 224, 224)
    """
```

### Step 1.4 — Data exploration notebook

Create `notebooks/01_data_exploration.ipynb`:
- Count images per class per generator
- Show sample images (real vs fake) from each generator
- Show sample DCT maps (real vs fake) — the frequency differences should be visible
- Histogram of image sizes before/after preprocessing
- Verify class balance

### Verification:
```python
# Run this to verify data pipeline
from src.dataset import AIDetectDCTDataset
from torch.utils.data import DataLoader

ds = AIDetectDCTDataset(split="train")
loader = DataLoader(ds, batch_size=4, shuffle=True)
batch = next(iter(loader))
print(f"RGB shape: {batch['image'].shape}")      # (4, 3, 224, 224)
print(f"DCT shape: {batch['dct_map'].shape}")     # (4, 1, 224, 224)
print(f"Labels: {batch['label']}")                 # tensor([0, 1, 1, 0])
print(f"Generators: {batch['metadata']['generator']}")
```

### Update EXPLANATION.md with:
- Exact download URLs used
- Image counts per split per generator
- Disk usage before and after
- Any issues encountered during download
- Sample image grid (save to results/plots/)

### Git commit: `data: download and preprocess GenImage subsets`

### ✅ CHECKPOINT: Ask user "Phase 1 complete. Data looks good? Ready for Phase 2 (Baselines)?"

---

## PHASE 2: BASELINE 1 — CLIP LINEAR PROBE

**Goal:** Train the simplest possible detector (frozen CLIP + 1 linear layer). This is SOTA candidate #1.

### Create `src/config.py`:

```python
"""
Central configuration file. ALL hyperparameters live here.
Nothing is hardcoded anywhere else in the codebase.
"""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # Paths
    project_root: Path = Path.home() / "aidetect"
    data_dir: Path = project_root / "data" / "processed"
    checkpoint_dir: Path = project_root / "checkpoints"
    results_dir: Path = project_root / "results"

    # Seeds (FIXED EVERYWHERE)
    seed: int = 42

    # Data
    image_size: int = 224
    num_workers: int = 4  # Reduce to 2 if RAM issues
    pin_memory: bool = False  # True for CUDA, False for MPS

    # CLIP
    clip_model_name: str = "ViT-B-16"
    clip_pretrained: str = "laion2b_s34b_b88k"
    clip_embed_dim: int = 512

    # Training — Linear Probe
    probe_batch_size: int = 32
    probe_lr: float = 1e-3
    probe_weight_decay: float = 1e-4
    probe_epochs: int = 20
    probe_scheduler: str = "cosine"

    # Training — Hybrid
    hybrid_batch_size: int = 16
    hybrid_lr: float = 5e-4
    hybrid_weight_decay: float = 1e-4
    hybrid_epochs: int = 30

    # Training — Freq-Guided (final model)
    final_batch_size: int = 16
    final_lr: float = 3e-4
    final_weight_decay: float = 1e-4
    final_epochs: int = 40

    # Frequency branch
    freq_branch_out_dim: int = 256
    freq_branch_type: str = "resnet18"  # or "custom_cnn"

    # Fusion
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.3

    # Robustness augmentation
    jpeg_q_range: tuple = (50, 100)
    blur_sigma_range: tuple = (0.1, 2.0)
    downscale_size: int = 112
    robustness_prob: float = 0.5

    # Evaluation
    test_generators: list = field(default_factory=lambda: [
        "sdv14", "sdv15", "midjourney", "adm",
        "glide", "biggan", "wukong", "vqdm"
    ])

    # Device
    @property
    def device(self):
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
```

### Create `src/seed.py`:

```python
"""Fix ALL random seeds for reproducibility."""
import os
import random
import numpy as np
import torch

def fix_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't have manual_seed_all but torch.manual_seed covers it
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Create `src/models/clip_probe.py`:

```python
"""
CLIP Linear Probe — Baseline 1 (SOTA Candidate #1)

Architecture:
    Frozen CLIP ViT-B/16 image encoder → 512-dim CLS token → Linear(512, 2)

This is the simplest possible detector. It tests whether CLIP's
pre-trained features already contain enough signal to detect AI images.

Published baselines (Cozzolino et al., CVPRW 2024) show this approach
achieves ~90% in-distribution and ~75-85% cross-generator AUC.
"""
import torch
import torch.nn as nn
import open_clip

class CLIPLinearProbe(nn.Module):
    def __init__(self, clip_model_name: str, clip_pretrained: str, num_classes: int = 2):
        super().__init__()
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.visual_encoder = clip_model.visual
        self.visual_encoder.eval()
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.visual_encoder(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for analysis)."""
        with torch.no_grad():
            return self.visual_encoder(x)
```

### Create `src/train.py`:

```python
"""
Generic training loop for all model variants.

Features:
- Works with any model that takes (image) or (image, dct_map) as input
- Early stopping on validation AUC
- Automatic checkpoint saving (with backup of previous best)
- Logs to console + results/metrics/
- Disk check before saving checkpoints
- tqdm progress bars

Usage:
    python -m src.train --model clip_probe --epochs 20
    python -m src.train --model hybrid --epochs 30
    python -m src.train --model freq_guided --epochs 40
"""
```

**Training loop must include:**
1. `fix_seeds()` at the very start
2. Disk space check before saving any checkpoint
3. Backup previous best checkpoint before overwriting
4. Log train_loss, train_acc, val_loss, val_acc, val_auc per epoch
5. Save metrics as JSON to `results/metrics/{model_name}_training.json`
6. Early stopping patience = 5 epochs on val_auc

### Create `src/evaluate.py`:

```python
"""
Cross-generator evaluation suite.

For each test generator:
- Accuracy
- AUC (Area Under ROC Curve)
- Precision, Recall, F1
- Confusion matrix

Also evaluates robustness:
- JPEG compression at Q=70, 50, 30
- Gaussian blur at sigma=1, 2, 3
- Resize 112→224 (simulating social media downscaling)

Outputs:
- results/metrics/{model_name}_cross_gen.json
- results/metrics/{model_name}_robustness.json
- results/tables/{model_name}_results.md (markdown table)
- results/plots/{model_name}_roc_curves.png
"""
```

### Expected output — results table:

```
| Generator   | Accuracy | AUC   | Precision | Recall | F1    |
|-------------|----------|-------|-----------|--------|-------|
| SD v1.4     | 0.XX     | 0.XX  | 0.XX      | 0.XX   | 0.XX  |
| SD v1.5     | 0.XX     | 0.XX  | 0.XX      | 0.XX   | 0.XX  |
| Midjourney  | 0.XX     | 0.XX  | 0.XX      | 0.XX   | 0.XX  |
| ADM         | 0.XX     | 0.XX  | 0.XX      | 0.XX   | 0.XX  |
| GLIDE       | 0.XX     | 0.XX  | 0.XX      | 0.XX   | 0.XX  |
| BigGAN      | 0.XX     | 0.XX  | 0.XX      | 0.XX   | 0.XX  |
| Wukong      | 0.XX     | 0.XX  | 0.XX      | 0.XX   | 0.XX  |
| VQDM        | 0.XX     | 0.XX  | 0.XX      | 0.XX   | 0.XX  |
| **Avg**     | **0.XX** |**0.XX**|          |        |       |
```

### Update EXPLANATION.md with:
- Full training log (loss curves, final metrics)
- Cross-generator results table
- Analysis: which generators are hardest? Why?
- Training time, peak memory usage
- Any MPS issues encountered

### Git commit: `feat: CLIP linear probe baseline with cross-gen evaluation`

### ✅ CHECKPOINT: Ask user "Baseline 1 complete. Here are the results: [table]. Ready for Phase 3?"

---

## PHASE 3: BASELINE 2 — AIDE-STYLE HYBRID

**Goal:** Build the two-branch hybrid detector (CLIP features + DCT frequency features). This is SOTA candidate #2.

### Create `src/transforms.py`:

```python
"""
All image transformations and augmentations.

Contains:
    - get_clip_transforms(): Standard CLIP preprocessing
    - get_dct_transform(): RGB → DCT spectral map conversion
    - get_robustness_augmentations(): JPEG/blur/resize degradations (Phase 4)
    - jpeg_compress_tensor(): Differentiable-ish JPEG simulation
"""
import scipy.fftpack as fft
import numpy as np
import torch

def compute_dct_map(image_np: np.ndarray) -> np.ndarray:
    """
    Convert RGB image (H, W, 3) to 2D DCT spectral map (H, W).

    Steps:
    1. Convert to grayscale via luminance formula: 0.299R + 0.587G + 0.114B
    2. Apply 2D DCT (type-II, orthonormalized)
    3. Take abs() + log1p() for dynamic range compression
    4. Min-max normalize to [0, 1]

    The resulting map highlights frequency-domain artifacts that
    differ between real photographs and AI-generated images.
    AI images typically show anomalous patterns in mid-to-high
    frequency bands due to upsampling operations in generators.
    """
```

### Create `src/models/hybrid.py`:

```python
"""
AIDE-Style Hybrid Detector — Baseline 2 (SOTA Candidate #2)

Architecture:
    Branch 1: Frozen CLIP ViT-B/16 → 512-dim
    Branch 2: Small CNN on DCT frequency map → 256-dim
    Fusion: Concatenate → MLP(768, 256, 2)

This follows the approach of AIDE (Yan et al., ICLR 2025) which showed
that combining semantic (CLIP) and frequency (DCT) features significantly
outperforms either branch alone.

Trainable parameters: ~5M (frequency CNN + fusion MLP only)
"""
import torch
import torch.nn as nn
import open_clip
from .components import FrequencyCNN

class HybridDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Branch 1: Frozen CLIP
        clip_model, _, _ = open_clip.create_model_and_transforms(
            config.clip_model_name, pretrained=config.clip_pretrained
        )
        self.clip_encoder = clip_model.visual
        self.clip_encoder.eval()
        for p in self.clip_encoder.parameters():
            p.requires_grad = False

        # Branch 2: Frequency CNN (trainable)
        self.freq_encoder = FrequencyCNN(
            in_channels=1,
            out_dim=config.freq_branch_out_dim
        )

        # Fusion head (trainable)
        fused_dim = config.clip_embed_dim + config.freq_branch_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, 2)
        )

    def forward(self, rgb: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            clip_feat = self.clip_encoder(rgb)       # (B, 512)
        freq_feat = self.freq_encoder(dct)            # (B, 256)
        fused = torch.cat([clip_feat, freq_feat], dim=1)  # (B, 768)
        return self.classifier(fused)

    def get_branch_features(self, rgb, dct):
        """Return individual branch outputs for analysis."""
        with torch.no_grad():
            clip_feat = self.clip_encoder(rgb)
        freq_feat = self.freq_encoder(dct)
        return clip_feat, freq_feat
```

### FrequencyCNN component:

```python
class FrequencyCNN(nn.Module):
    """
    Lightweight CNN for processing DCT frequency maps.

    Architecture: 4 conv blocks → global average pool → FC
    Input: (B, 1, 224, 224) DCT map
    Output: (B, out_dim) frequency features

    ~2M parameters — trains in minutes on MPS.
    """
    def __init__(self, in_channels=1, out_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 → 32
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # 112x112

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 56x56

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),  # 28x28

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)
```

### Training:
- Same training loop as Phase 2, but DataLoader now returns (rgb, dct, label)
- Batch size = 16 (two inputs means more memory)
- Epochs = 30
- Compare results against Phase 2 baseline

### Deliverable:
Updated results table now showing BOTH baselines side by side.

### Update EXPLANATION.md with:
- Hybrid model architecture details + parameter count
- Training curves comparison (probe vs hybrid)
- Cross-generator results comparison
- Analysis: does adding frequency features help? Where?

### Git commit: `feat: AIDE-style hybrid detector baseline 2`

### ✅ CHECKPOINT: Ask user "Both baselines complete. Comparative results: [table]. Ready for Phase 4 (Improvements)?"

---

## PHASE 4: IMPROVEMENTS + ABLATIONS

**Goal:** Implement 2 meaningful improvements, run full ablation study, prove they help.

### Improvement 1: Frequency-Guided Cross-Attention (Architectural Change)

Create `src/models/freq_guided.py`:

```python
"""
Frequency-Guided CLIP Detector — Final Model

KEY INNOVATION: Instead of simple concatenation, frequency features
GUIDE the model's attention over CLIP's spatial patch tokens.

Architecture:
    1. CLIP ViT-B/16 → extract intermediate patch tokens (B, 197, 512)
       (not just the CLS token — we need spatial information)
    2. DCT frequency map → FrequencyCNN → 256-dim summary
    3. Freq-Guided Cross-Attention: frequency features query CLIP patches
       to find which spatial regions have frequency anomalies
    4. Attended features + CLS token + freq features → classifier

This directly enables:
- "Which part of the image looks AI" (from attention weights)
- Better cross-generator generalization (frequency artifacts are generator-agnostic)

Trainable parameters: ~8M (freq CNN + attention + classifier)
"""

class FreqGuidedAttention(nn.Module):
    """
    Cross-attention where frequency features attend to CLIP spatial tokens.

    Input:
        clip_tokens: (B, num_patches, 512) — from ViT intermediate layer
        freq_feat: (B, 256) — from FrequencyCNN

    Output:
        attended: (B, 256) — frequency-guided spatial summary
        attn_weights: (B, 1, num_patches) — attention map (for Grad-CAM!)
    """
    def __init__(self, clip_dim=512, freq_dim=256, attn_dim=128):
        super().__init__()
        self.to_q = nn.Linear(freq_dim, attn_dim)
        self.to_k = nn.Linear(clip_dim, attn_dim)
        self.to_v = nn.Linear(clip_dim, attn_dim)
        self.scale = attn_dim ** -0.5
        self.out_proj = nn.Linear(attn_dim, 256)

    def forward(self, clip_tokens, freq_feat):
        q = self.to_q(freq_feat).unsqueeze(1)    # (B, 1, 128)
        k = self.to_k(clip_tokens)                 # (B, N, 128)
        v = self.to_v(clip_tokens)                 # (B, N, 128)

        attn_weights = torch.softmax(
            (q @ k.transpose(-1, -2)) * self.scale, dim=-1
        )  # (B, 1, N)

        attended = (attn_weights @ v).squeeze(1)   # (B, 128)
        return self.out_proj(attended), attn_weights


class FreqGuidedDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # CLIP backbone (frozen, but we hook into intermediate layers)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            config.clip_model_name, pretrained=config.clip_pretrained
        )
        self.clip_visual = clip_model.visual
        self.clip_visual.eval()
        for p in self.clip_visual.parameters():
            p.requires_grad = False

        # We need to extract INTERMEDIATE patch tokens, not just CLS
        # Hook into the last transformer block's output
        self._patch_tokens = None
        self._hook = self.clip_visual.transformer.resblocks[-1].register_forward_hook(
            lambda module, input, output: setattr(self, '_patch_tokens', output)
        )

        # Frequency branch
        self.freq_encoder = FrequencyCNN(in_channels=1, out_dim=256)

        # Cross-attention
        self.freq_attention = FreqGuidedAttention(
            clip_dim=512, freq_dim=256, attn_dim=128
        )

        # Final classifier
        # Input: CLS(512) + freq_attended(256) + freq_global(256) = 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(256, 2)
        )

    def forward(self, rgb, dct):
        # Get CLIP features
        with torch.no_grad():
            cls_token = self.clip_visual(rgb)        # (B, 512)
        patch_tokens = self._patch_tokens             # (B, 197, 512)
        # Remove CLS token from patches
        spatial_tokens = patch_tokens[:, 1:, :]       # (B, 196, 512)

        # Get frequency features
        freq_feat = self.freq_encoder(dct)             # (B, 256)

        # Cross-attention: frequency guides spatial attention
        attended, attn_weights = self.freq_attention(spatial_tokens, freq_feat)
        # attn_weights shape: (B, 1, 196) — this IS your spatial heatmap!

        # Fuse everything
        fused = torch.cat([cls_token, attended, freq_feat], dim=1)  # (B, 1024)
        logits = self.classifier(fused)

        return logits

    def get_attention_map(self, rgb, dct):
        """
        Get the frequency-guided attention map for visualization.
        Returns a (B, 14, 14) spatial attention map that can be
        overlaid on the original image to show "which parts look AI."
        """
        self.forward(rgb, dct)
        patch_tokens = self._patch_tokens
        spatial_tokens = patch_tokens[:, 1:, :]
        freq_feat = self.freq_encoder(dct)
        _, attn_weights = self.freq_attention(spatial_tokens, freq_feat)
        # Reshape from (B, 1, 196) to (B, 14, 14)
        return attn_weights.squeeze(1).reshape(-1, 14, 14)
```

### Improvement 2: Robustness-Aware Training (Training Strategy Change)

Create/update `src/transforms.py` to add:

```python
"""
Robustness augmentations that simulate real-world image degradations.

Applied ONLY during training to make the model robust to:
- Social media JPEG recompression (Q=50-100)
- Messaging app blur
- Screenshot/resize artifacts
- Platform-specific processing pipelines

These augmentations are applied BEFORE the standard CLIP preprocessing.
"""

class RobustnessAugmentation:
    def __init__(self, config):
        self.jpeg_q_range = config.jpeg_q_range
        self.blur_sigma_range = config.blur_sigma_range
        self.downscale_size = config.downscale_size
        self.prob = config.robustness_prob

    def __call__(self, image_pil):
        """Apply random degradations to a PIL image."""
        # 1. Random JPEG compression
        if random.random() < self.prob:
            q = random.randint(*self.jpeg_q_range)
            buffer = io.BytesIO()
            image_pil.save(buffer, format='JPEG', quality=q)
            buffer.seek(0)
            image_pil = Image.open(buffer)

        # 2. Random Gaussian blur
        if random.random() < self.prob * 0.6:
            sigma = random.uniform(*self.blur_sigma_range)
            image_pil = image_pil.filter(
                ImageFilter.GaussianBlur(radius=sigma)
            )

        # 3. Random downscale + upscale (social media simulation)
        if random.random() < self.prob * 0.6:
            w, h = image_pil.size
            small = image_pil.resize(
                (self.downscale_size, self.downscale_size),
                Image.LANCZOS
            )
            image_pil = small.resize((w, h), Image.LANCZOS)

        return image_pil
```

### Ablation Study

Create `scripts/run_ablations.py`:

```python
"""
Run complete ablation study. Trains and evaluates 7 model variants:

1. CLIP Linear Probe (Phase 2 baseline)
2. AIDE-style Hybrid (Phase 3 baseline)
3. Freq-Guided Attention only (Improvement 1 alone)
4. Hybrid + Robustness Aug only (Improvement 2 alone)
5. Full model (both improvements)
6. Full model WITHOUT freq branch (ablation: remove freq)
7. Full model WITHOUT robustness aug (ablation: remove aug)

For each variant, evaluates:
- In-distribution accuracy (SD v1.4)
- Cross-generator average AUC (all 8 generators)
- Robustness average AUC (JPEG Q=50, blur σ=2, resize 112)

Saves complete results to results/metrics/ablation_study.json
Generates markdown table to results/tables/ablation_table.md
"""
```

### Expected ablation table:

```
| # | Model Variant                         | SD v1.4 Acc | Cross-Gen AUC | Robust AUC | Δ vs Base |
|---|---------------------------------------|-------------|---------------|------------|-----------|
| 1 | CLIP Linear Probe                     | 0.XX        | 0.XX          | 0.XX       | —         |
| 2 | AIDE-style Hybrid                     | 0.XX        | 0.XX          | 0.XX       | +0.XX     |
| 3 | + Freq-Guided Attention               | 0.XX        | 0.XX          | 0.XX       | +0.XX     |
| 4 | + Robustness Augmentation             | 0.XX        | 0.XX          | 0.XX       | +0.XX     |
| 5 | Full Model (3 + 4)                    | 0.XX        | 0.XX          | 0.XX       | +0.XX     |
| 6 | Full − freq branch (ablation)         | 0.XX        | 0.XX          | 0.XX       | −0.XX     |
| 7 | Full − robustness aug (ablation)      | 0.XX        | 0.XX          | 0.XX       | −0.XX     |
```

**If running all 7 variants on MacBook is too slow:** Run variants 1-2 locally (you already have them from Phases 2-3). Run variants 3-7 on Google Colab. Create `notebooks/colab_training.ipynb` with cells that:
1. Mount Google Drive
2. Upload zipped dataset
3. Install deps
4. Run training for each variant
5. Download checkpoints to Drive

### Update EXPLANATION.md with:
- Full ablation results table
- Analysis of each improvement's contribution
- Training curves for all variants
- Any surprises or unexpected results

### Git commit: `feat: freq-guided attention + robustness training + full ablation study`

### ✅ CHECKPOINT: Ask user "Phase 4 complete. Ablation results: [table]. Ready for Phase 5 (Explainability)?"

---

## PHASE 5: GRAD-CAM EXPLAINABILITY

**Goal:** Generate visual explanations (heatmaps) showing which regions the model thinks look AI, plus text explanations of why.

### Create `src/gradcam_utils.py`:

```python
"""
Grad-CAM and Attention-based explainability for the detector.

Two complementary approaches:

1. Frequency-Guided Attention Map (from our model's cross-attention)
   - Directly available from FreqGuidedDetector.get_attention_map()
   - Shows which spatial regions have frequency anomalies
   - Resolution: 14×14 (one weight per ViT patch)

2. Grad-CAM on CLIP ViT layers
   - Uses pytorch-grad-cam library
   - Targets the last transformer block's LayerNorm
   - Provides gradient-weighted activation maps
   - Can be noisy on ViTs — use Attention Rollout as fallback

3. Combined heatmap
   - Element-wise product of (1) and (2)
   - Sharpens the explanation by combining both signals

Output:
    - Heatmap overlay on original image (PIL Image)
    - Raw attention weights (numpy array)
    - Top-K most suspicious patch coordinates
"""
```

### Create `src/explain.py`:

```python
"""
Generate text explanations for WHY an image is classified as AI-generated.

Approach:
1. Get the frequency-guided attention map (14×14)
2. Identify top-K most activated patches
3. For each high-activation patch:
   a. Crop the corresponding region from the original image
   b. Compute its DCT spectrum
   c. Match against known artifact signatures:
      - Spectral peaks at specific frequencies (grid artifacts from upsampling)
      - Unusually smooth frequency rolloff (diffusion model signature)
      - Periodic patterns in high-frequency bands (GAN fingerprints)
4. Also check global image statistics:
   - JPEG ghost analysis (inconsistent compression)
   - Color histogram anomalies
   - Edge coherence score

Returns a list of explanation strings like:
    - "High-frequency artifacts detected in the background region (top-right)"
    - "Unnatural texture smoothness in facial area"
    - "Spectral anomaly consistent with diffusion model upsampling"
    - "Color distribution inconsistent with natural photography"
"""
```

### Generate sample Grad-CAM visualizations:
- Pick 5 real images, 5 fake images from each generator
- Generate heatmap overlays for all
- Save to `results/gradcam_samples/`
- These will go in the report AND the web demo

### Update EXPLANATION.md with:
- Grad-CAM approach details
- Sample visualizations (reference file paths)
- Quality assessment: are the heatmaps meaningful?
- Fallback strategy if Grad-CAM on ViT is too noisy

### Git commit: `feat: Grad-CAM explainability + text explanations`

### ✅ CHECKPOINT: Ask user "Phase 5 complete. Here are sample heatmaps. Ready for Phase 6 (Web App)?"

---

## PHASE 6: WEB APPLICATION

**Goal:** Full-stack web app with: upload → verdict → heatmap → explanation → comparative panel → evaluation dashboard.

### Backend: `backend/main.py`

```python
"""
FastAPI backend for AI-Generated Image Detector.

Endpoints:
    POST /detect          — Single image detection
    POST /detect/compare  — Comparative detection (all models)
    GET  /dashboard/data  — Evaluation metrics for dashboard

Response schema for /detect:
{
    "verdict": "AI-Generated" | "Real",
    "confidence": 0.95,
    "heatmap_base64": "data:image/png;base64,...",
    "explanations": [
        {"region": "top-right", "reason": "High-frequency artifacts...", "severity": 0.8},
        {"region": "center", "reason": "Unnatural texture...", "severity": 0.6}
    ],
    "model_name": "freq_guided_v1",
    "inference_time_ms": 142
}

Response schema for /detect/compare:
{
    "models": [
        {"name": "CLIP Linear Probe", "verdict": "Real", "confidence": 0.52},
        {"name": "Hybrid Detector", "verdict": "AI-Generated", "confidence": 0.78},
        {"name": "Freq-Guided (Ours)", "verdict": "AI-Generated", "confidence": 0.95,
         "heatmap_base64": "...", "explanations": [...]}
    ]
}

Response schema for /dashboard/data:
{
    "cross_generator": {
        "generators": ["sdv14", ...],
        "models": {
            "clip_probe": {"accuracy": [...], "auc": [...]},
            "hybrid": {...},
            "freq_guided": {...}
        }
    },
    "robustness": {
        "degradations": ["jpeg_q70", "jpeg_q50", "blur_s2", "resize_112"],
        "models": { ... }
    },
    "ablation": { ... }
}
"""
```

### Frontend components:

**`UploadZone.jsx`** — Drag-and-drop area with file type validation (images only, max 10MB)

**`VerdictCard.jsx`** — Large verdict display with animated confidence meter (green→red gradient)

**`HeatmapOverlay.jsx`** — Original image with toggleable heatmap overlay, opacity slider

**`ExplanationPanel.jsx`** — Accordion list of explanation cards, each with:
- Region indicator (bounding box on the image)
- Reason text
- Severity bar

**`ComparativePanel.jsx`** — Side-by-side cards for each model:
- Model name + brief description
- Verdict + confidence
- Only the final model shows the heatmap (highlighting our improvement)

**`Dashboard.jsx`** — Tabbed layout:
- Tab 1: Cross-Generator Accuracy Table (color-coded: green=high, red=low)
- Tab 2: Robustness Charts (line charts, accuracy vs degradation level)
- Tab 3: Qualitative Grid (sample images with Grad-CAM from each model)
- Tab 4: Ablation Study Table

### Running the app:

```bash
# Terminal 1: Backend
cd ~/aidetect/backend
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd ~/aidetect/frontend
npm run dev  # Vite dev server on port 5173
```

### Disk budget check:
```
- Node modules: ~200MB
- Model checkpoints (3 models): ~1.2GB
- Everything else: negligible
```

### Update EXPLANATION.md with:
- All API endpoints with request/response examples
- Frontend component hierarchy
- Screenshots of each view
- Known limitations

### Git commit: `feat: full-stack web application with comparative panel + dashboard`

### ✅ CHECKPOINT: Ask user "Web app is running. Want to test it together before Phase 7?"

---

## PHASE 7: REPORT + PRESENTATION + VIDEO

**Goal:** Produce all deliverables required by the course.

### Final Report (`report/final_report.md` or `.pdf`):

**Structure (follow course rubric exactly):**

1. **Title + Abstract** (half page)
   - Problem, approach, key results in 150 words

2. **Introduction** (1 page)
   - Why AI image detection matters
   - Challenge of cross-generator generalization
   - Our contributions (frequency-guided attention, robustness training)

3. **Related Work** (1 page)
   - CNNSpot (Wang et al., CVPR 2020)
   - UnivFD (Ojha et al., CVPR 2023)
   - NPR (Tan et al., CVPR 2024)
   - AIDE (Yan et al., ICLR 2025)
   - C2P-CLIP (Tan et al., AAAI 2025)
   - Cozzolino et al. (CVPRW 2024)

4. **Problem Formulation** (0.5 page)
   - Task: binary classification (real vs AI-generated)
   - Input: RGB image of any resolution
   - Output: label, confidence, explanation, heatmap
   - Success criteria: >90% in-distribution, >80% cross-gen AUC
   - Constraints: must generalize to unseen generators

5. **Dataset** (1 page)
   - GenImage description + our subset strategy
   - Preprocessing pipeline
   - Bias mitigation (JPEG normalization)
   - ForenSynths for cross-dataset eval
   - Data statistics table

6. **SOTA Model Survey** (1 page)
   - Baseline 1: CLIP Linear Probe — architecture, results, analysis
   - Baseline 2: AIDE-style Hybrid — architecture, results, analysis
   - Comparative table

7. **Our Approach** (2 pages)
   - Architecture diagram (create using matplotlib or draw.io)
   - Frequency-Guided Cross-Attention mechanism explanation
   - Robustness-Aware Training strategy
   - Grad-CAM explainability pipeline

8. **Experiments** (2 pages)
   - Training details (hyperparameters, hardware, time)
   - Cross-generator results table (all 3 models × 8 generators)
   - Robustness results (accuracy vs degradation curves)
   - Full ablation study table
   - Qualitative examples (Grad-CAM visualizations)

9. **Web Application** (0.5 page)
   - Architecture (FastAPI + React)
   - Screenshots
   - Features demonstrated

10. **Conclusion + Future Work** (0.5 page)
    - What worked, what didn't
    - Limitations (dataset size, compute constraints)
    - Future: larger datasets, more generators, video detection

11. **References**

### Presentation slides:
- 10-12 slides for ~10 min presentation
- Create with Python-pptx or Keynote
- Slide flow: Problem → Why it matters → Our approach → Demo → Results → Conclusion

### Demo video (2-3 minutes):
- Screen record with QuickTime
- Show: upload real image → "Real" verdict → upload Midjourney image → "AI" verdict with heatmap → show comparative panel → switch to dashboard → show cross-gen table → show robustness charts
- Add voiceover explaining what's happening

### Update EXPLANATION.md with:
- Final project statistics (total lines of code, training hours, disk usage)
- Complete list of all files in the project
- Lessons learned
- Final thoughts

### Git commit: `docs: final report, presentation, and project documentation`

### ✅ CHECKPOINT: "All deliverables complete. Ready for submission?"

---

## PHASE 8: FINAL CLEANUP & SUBMISSION

### Checklist:

```
[ ] All code formatted with `black .`
[ ] All functions have docstrings
[ ] All files have module-level docstrings
[ ] README.md has complete setup instructions (someone else can reproduce)
[ ] requirements.txt is accurate (`pip freeze > requirements_full.txt` for reference)
[ ] .gitignore covers all data/checkpoint/node_module files
[ ] Data download instructions are in README (not the data itself)
[ ] EXPLANATION.md is complete with all phases documented
[ ] No hardcoded absolute paths (everything uses config.py)
[ ] Seeds are fixed in every script
[ ] All results are reproducible
[ ] Demo video is recorded and linked
[ ] Report is complete and formatted
[ ] Presentation slides are done
[ ] All git commits are clean with meaningful messages
```

---

## EMERGENCY FALLBACKS

| Problem | Solution |
|---|---|
| GenImage download is blocked/too slow | Use Kaggle "tiny-genimage" dataset (~1GB) or create your own mini-dataset using Stable Diffusion locally |
| MPS crashes during training | Reduce batch_size to 4, add `torch.mps.empty_cache()` after each batch, or just use CPU (slower but stable) |
| Disk fills up | `df -h ~` to check. Delete `data/raw/` after preprocessing. Compress checkpoints with gzip. |
| Grad-CAM on ViT produces garbage | Switch to Attention Rollout: average attention weights across all ViT layers |
| Cross-gen accuracy is terrible (<60%) | This is actually fine! Document it honestly. The course says no penalty for advanced methods that don't fully work. Focus on showing the ablations and your analysis. |
| Colab keeps disconnecting | Save every 3 epochs. Use `from google.colab import output; output.eval_js('google.colab.kernel.proxyPort(8080)')` to prevent timeout. |
| React setup fails on 15GB disk | Skip `npm install` and use a single HTML file with CDN-loaded React + Tailwind instead |
| Time crunch | Priority order: Phase 0-2 (minimum viable), Phase 4 (improvements), Phase 6 (demo), Phase 7 (report). Phases 3 and 5 can be simplified. |

---

## SCORING ALIGNMENT (30/30 TARGET)

| Requirement | How We Address It | Phase |
|---|---|---|
| Option 1: E2E DL app with training + SOTA | ✅ Full pipeline from data to deployment | All |
| A. Problem Formulation | ✅ Task, I/O, constraints, success criteria in config.py + report | 7 |
| B. Data Acquisition & Processing | ✅ GenImage subset + ForenSynths + preprocessing | 1 |
| C. 2+ SOTA candidates evaluated | ✅ CLIP probe + AIDE hybrid, both with cross-gen tables | 2, 3 |
| D. 2+ improvements with ablations | ✅ Freq-guided attention + robustness aug + 7-variant ablation | 4 |
| No tutorial copying | ✅ Novel fusion architecture | 4 |
| Open-source models only | ✅ OpenCLIP, timm, all open-source | All |
| Innovative/advanced approach | ✅ Cross-attention mechanism, frequency guidance | 4 |
| Proposal alignment | ✅ Same datasets, references, demo concept | All |
| Presentation milestone | ✅ Slides explicitly scheduled in Phase 7 | 7 |
| Code clarity (docstrings, typing) | ✅ Enforced in RULES: black formatting, docstrings, type hints | 8 |
| Documentation quality | ✅ README, EXPLANATION.md, report | 7, 8 |
| Comparative panel in demo | ✅ ComparativePanel.jsx shows all 3 models side-by-side | 6 |
| Evaluation dashboard | ✅ Dashboard.jsx with AccuracyTable, RobustnessChart, QualitativeGrid, AblationTable | 6 |
| Reproducibility | ✅ Fixed seeds, setup.sh, requirements.txt, README | 0, 8 |
| Demo video | ✅ Explicitly planned with shot list | 7 |

**Target: 30/30**

---

## FINAL NOTE TO CLAUDE CODE

When you start working on this project:

1. Read this ENTIRE file first.
2. Start with Phase 0. Do not skip ahead.
3. After each phase, update EXPLANATION.md BEFORE asking to proceed.
4. If you're unsure about anything, ASK the user. Don't guess.
5. Treat the user's 15GB disk like gold. Check `df -h ~` obsessively.
6. Every file you create must have a docstring explaining what it does.
7. Test everything before claiming it works.
8. When training, show progress bars and estimated time remaining.
9. If a training run will take >2 hours on MPS, suggest Colab instead.
10. Remember: the user's MacBook is their daily driver. Don't make it unusable.

**START PHASE 0 NOW. Ask for permission before Phase 1.**
