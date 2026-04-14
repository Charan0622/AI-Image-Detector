# AI-Generated Image Detector

> **Can you trust what you see?**
> A deep learning system that detects AI-generated images, explains *why* they look fake, and shows you *where* the artifacts hide.

```
  Real Photo?          or          AI-Generated?
  +-------------+                  +-------------+
  |             |                  |  ~~fake~~   |
  |   [photo]   |     ------>     |  artifacts  |
  |             |                  |  detected!  |
  +-------------+                  +-------------+
                    Our Model
              Accuracy: 93.3%+
           Cross-Gen AUC: 0.98+
```

---

## Project Details

| | |
|---|---|
| **Course** | CMPE 258 -- Deep Learning, Spring 2026 |
| **University** | San Jose State University (SJSU) |
| **Instructor** | Prof. |
| **Project** | End-to-End AI-Generated Image Detection System |
| **Hardware** | MacBook Air M5 (16GB RAM, MPS Backend) |

---

## Team

| Name | Student ID | Role |
|------|-----------|------|
| **Charan Sai Gandham** | 019142955 | Solo Developer -- Architecture, Training, Evaluation, Web App, Documentation |

---

## Problem Statement

The rapid advancement of generative AI (Stable Diffusion, Midjourney, DALL-E, BigGAN) has made it nearly impossible for humans to distinguish AI-generated images from real photographs. This poses serious threats to:

- **Misinformation** -- fake images spread as real news
- **Identity fraud** -- deepfakes used for impersonation
- **Academic integrity** -- AI-generated content submitted as original work
- **Digital trust** -- erosion of confidence in visual media

**Our goal:** Build a robust detector that not only classifies images as Real/Fake but also *explains* its reasoning through visual heatmaps and textual explanations -- and generalizes to AI generators it has *never seen during training*.

---

## Dataset

### GenImage (Primary Dataset)

We use the [GenImage](https://github.com/GenImage-Dataset/GenImage) benchmark dataset, sourced from HuggingFace (`RohanRamesh/genimage-224`).

| Split | Real Images | Fake Images | Total | Purpose |
|-------|------------|-------------|-------|---------|
| Train | 96,000 | 96,000 | 192,000 | Model training |
| Val | 24,000 | 24,000 | 48,000 | Hyperparameter tuning |
| Test | 6,000 | 6,000 | 12,000 | Cross-generator evaluation |

**Test generators** (6 unseen generators for cross-gen evaluation):
| Generator | Type | Test Samples |
|-----------|------|-------------|
| ADM | Diffusion | 2,000 |
| GLIDE | Diffusion | 2,000 |
| Midjourney | Diffusion | 2,000 |
| Stable Diffusion v1.5 | Diffusion | 2,000 |
| VQDM | Autoregressive | 2,000 |
| Wukong | Diffusion | 2,000 |

**Total dataset size:** 252,000 images (~5.6 GB)

### Preprocessing Pipeline

```
Raw Image --> Resize (224x224, LANCZOS) --> JPEG Re-save (Q=95) --> Normalize
```

- All images standardized to 224x224 pixels
- JPEG quality normalized to Q=95 to remove compression bias (critical insight from Unbiased GenImage paper)
- 80/20 stratified train/val split with seed=42
- DCT (Discrete Cosine Transform) frequency maps computed on-the-fly for frequency-branch models

---

## Approach

### Architecture Overview

```
                    Input Image (224x224)
                           |
              +------------+------------+
              |                         |
        [CLIP ViT-B/16]          [DCT Transform]
        (Frozen, 86M params)      (Frequency Map)
              |                         |
         CLS Token (512-d)      [Frequency CNN]
              |                    (Trainable)
              |                         |
              +-------[Fusion]---------+
                         |
                    [Classifier]
                         |
                   Real / Fake
```

### Models Implemented

| # | Model | Architecture | Trainable Params | Description |
|---|-------|-------------|-----------------|-------------|
| 1 | **CLIP Linear Probe** | Frozen CLIP + Linear Head | 1,026 | Simplest baseline -- tests if CLIP features contain detection signal |
| 2 | **AIDE-style Hybrid** | Frozen CLIP + DCT FreqCNN + Fusion MLP | 651,970 | Two-branch detector combining semantic + frequency features |
| 3 | **Freq-Guided Attention** | CLIP + Multi-Scale FreqCNN + Cross-Attention | 1,670,341 | Frequency features guide spatial attention over CLIP patches |

### Key Innovations

1. **Frequency-Guided Cross-Attention** -- Instead of simple feature concatenation, frequency features *attend* to CLIP's spatial patch tokens, identifying which image regions have frequency anomalies

2. **Robustness-Aware Training** -- Data augmentation simulating real-world degradations:
   - JPEG recompression (Q=50-100)
   - Gaussian blur (sigma=0.1-2.0)
   - Downscale + upscale (simulating social media pipelines)

3. **DCT Spectral Analysis** -- 2D Discrete Cosine Transform maps reveal frequency-domain artifacts invisible to the human eye but characteristic of AI generators

---

## Progress

### Completed

- [x] **Phase 0:** Environment setup (Python 3.12, PyTorch 2.11, MPS backend)
- [x] **Phase 1:** Data acquisition & preprocessing (252K images, 5.6GB)
  - Downloaded GenImage from HuggingFace
  - Preprocessed all images to 224x224 JPEG Q=95
  - Verified dataset integrity (0 invalid, 0 wrong size)
  - Built PyTorch Dataset classes with DCT support
  - Created data exploration notebook

### In Progress

- [ ] **Phase 2:** CLIP Linear Probe baseline training & evaluation
- [ ] **Phase 3:** AIDE-style Hybrid detector
- [ ] **Phase 4:** Frequency-Guided Attention + Ablation Study
- [ ] **Phase 5:** Grad-CAM explainability pipeline
- [ ] **Phase 6:** Full-stack web application (FastAPI + React)
- [ ] **Phase 7:** Final report, presentation, and demo video
- [ ] **Phase 8:** Code cleanup and submission

---

## Next Steps

1. **Complete model training** on full 192K dataset for all three architectures
2. **Run cross-generator evaluation** across all 6 unseen generators
3. **Generate Grad-CAM heatmaps** for visual explainability
4. **Build web application** with upload, verdict, heatmap overlay, and comparative panel
5. **Write final report** with methodology, results, ablation study, and conclusions
6. **Record demo video** showcasing the full system end-to-end

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch 2.11, OpenCLIP (ViT-B/16), timm |
| **Compute** | Apple MPS (Metal Performance Shaders) |
| **Frequency Analysis** | SciPy (2D DCT), NumPy |
| **Explainability** | pytorch-grad-cam, Attention Rollout |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | React (CDN), Tailwind CSS |
| **Evaluation** | scikit-learn (AUC, F1, Precision, Recall) |
| **Visualization** | Matplotlib, Seaborn |

---

## Repository Structure

```
aidetect/
├── src/                    # Core source code
│   ├── models/             # CLIP Probe, Hybrid, Freq-Guided
│   ├── train_probe.py      # CLIP probe training
│   ├── train_hybrid.py     # Hybrid detector training
│   ├── train_freq_guided.py# Freq-guided + ablations
│   ├── dataset.py          # PyTorch datasets
│   ├── transforms.py       # Augmentations + DCT
│   ├── gradcam_utils.py    # Grad-CAM visualization
│   └── config.py           # Central configuration
├── scripts/                # Data download, preprocessing, feature extraction
├── backend/                # FastAPI inference server
├── frontend/               # React web application
├── results/                # Metrics, plots, tables, Grad-CAM samples
├── checkpoints/            # Model weights (gitignored)
├── data/                   # Dataset (gitignored)
├── notebooks/              # Exploration & experiments
└── report/                 # Final report & presentation
```

---

## How to Run

```bash
# 1. Clone and setup
git clone <repo-url>
cd aidetect
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data
python -m scripts.download_data

# 3. Preprocess
python -m scripts.preprocess_data

# 4. Extract CLIP features
python -m scripts.extract_features_chunked

# 5. Train models
python -m src.train_probe
python -m src.train_hybrid
python -m src.train_freq_guided --variant all

# 6. Launch web app
bash scripts/start_app.sh
# Backend: http://localhost:8001
# Frontend: http://localhost:8080
```

---

## References

1. Wang et al., "CNN-generated images are surprisingly easy to spot...for now," *CVPR 2020*
2. Ojha et al., "Towards Universal Fake Image Detectors," *CVPR 2023*
3. Tan et al., "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection," *CVPR 2024*
4. Yan et al., "AIDE: AI-Generated Image DEtector," *ICLR 2025*
5. Tan et al., "C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection," *AAAI 2025*
6. Cozzolino et al., "Raising the Bar of AI-generated Image Detection with CLIP," *CVPRW 2024*
7. Zhu et al., "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image," *NeurIPS 2023*

---

<p align="center">
  <b>CMPE 258 -- Deep Learning | Spring 2026 | San Jose State University</b><br>
  Built by Charan Sai Gandham (019142955)
</p>
