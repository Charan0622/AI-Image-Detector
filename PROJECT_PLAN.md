# AI-Generated Image Detection — Full Project Plan

**CMPE 258 · Spring 2026 · Charan Sai Gandham**

---

## Reality Check: Your Constraints

| Resource | Reality |
|---|---|
| Machine | MacBook Air M5, 16GB unified RAM, 15GB free disk |
| GPU | Apple MPS backend (~equivalent to a low-end discrete GPU for training) |
| Cloud compute | None paid — but **Google Colab free tier** gives T4 GPU (15GB VRAM, ~12hrs/session) |
| Dataset budget | GenImage full = **~500GB** → you can only use a **tiny subset** (~3-5GB) |

**Bottom line:** You cannot train from scratch on your laptop. Your strategy must be **fine-tuning frozen foundation models with lightweight heads**, and offloading any heavier training to Colab.

---

## Architecture Decision (The Core Idea)

Your detector will be a **two-branch fusion model**:

```
Input Image (224×224)
    ├── Branch 1: CLIP ViT-B/16 (frozen) → 512-dim semantic embedding
    ├── Branch 2: DCT Frequency Analyzer → 256-dim frequency features
    └── Fusion MLP Head → Real/Fake + Grad-CAM explainability
```

**Why this works on your hardware:**
- CLIP ViT-B/16 is ~350MB. Frozen = no gradient memory. Forward pass on 16GB MPS is fine at batch_size=16-32.
- The DCT branch is a lightweight CNN (ResNet-18 or custom 4-layer CNN) operating on frequency maps — tiny footprint.
- Only the fusion head + DCT branch are trainable → ~5-15M parameters total.
- Grad-CAM hooks onto the ViT attention layers for "which region looks AI" visualization.

**Why this satisfies the course requirements:**
- ✅ Uses recent SOTA open-source model (CLIP ViT-B/16 via OpenCLIP)
- ✅ Two meaningful improvements over baseline: **(1) frequency-domain fusion, (2) Grad-CAM attention explainability**
- ✅ Cross-generator evaluation on GenImage protocol
- ✅ Not a basic tutorial copy — the fusion architecture + frequency branch is a real research contribution

---

## Tech Stack (Pin These Versions)

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12 | Required for latest MPS support |
| PyTorch | 2.4+ | MPS backend for M5 training |
| torchvision | 0.19+ | Image transforms, Grad-CAM hooks |
| open_clip_torch | 2.26+ | CLIP ViT-B/16 pretrained weights |
| timm | 1.0+ | Optional: ResNet-18 for frequency branch |
| numpy | 1.26+ | FFT/DCT operations |
| scipy | 1.13+ | DCT transforms |
| Pillow | 10.3+ | Image I/O |
| scikit-learn | 1.5+ | Metrics (AUC, accuracy, confusion matrix) |
| matplotlib / seaborn | latest | Plots for report |
| FastAPI | 0.111+ | Backend API |
| uvicorn | 0.30+ | ASGI server |
| React (Vite) | 18+ / 5+ | Frontend |
| pytorch-grad-cam | 1.5+ | Grad-CAM visualization |
| wandb (optional) | latest | Experiment tracking |

**Environment setup (Day 1 command sequence):**
```bash
# Create isolated environment
python3.12 -m venv ~/aidetect-env
source ~/aidetect-env/bin/activate

# Core ML stack
pip install torch torchvision torchaudio
pip install open-clip-torch timm scipy scikit-learn
pip install matplotlib seaborn pillow pandas tqdm

# Verify MPS
python -c "import torch; print(torch.backends.mps.is_available())"  # Must print True

# Web stack (install later in Phase 4)
pip install fastapi uvicorn python-multipart

# Explainability
pip install pytorch-grad-cam
```

Set this env var in your shell profile:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
This prevents crashes when an op isn't implemented on MPS yet — it silently falls back to CPU.

---

## Phase 0: Dataset Strategy (Days 1-2)

**Problem:** GenImage is ~500GB. You have 15GB.

**Solution:** Use only the **Stable Diffusion v1.4 subset** for training + all generator test splits.

### What to download (~3-5GB total):

1. **Training data:** SD v1.4 split only — train/ai + train/nature (~2-3GB)
   - Subsample to ~20K images per class (real vs fake) if still too large
2. **Test data:** All 8 generator val splits (ai + nature) — these are small (~200-500MB total)
3. **ForenSynths test set** (optional, ~500MB) — for cross-dataset evaluation

### Download approach:
GenImage is available on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AKDIHF) and Google Drive. Download only what you need:

```
data/
├── train/               # SD v1.4 only
│   ├── ai/              # ~20K AI images
│   └── nature/          # ~20K real images
├── test/
│   ├── sdv14/
│   ├── sdv15/
│   ├── midjourney/
│   ├── adm/
│   ├── glide/
│   ├── biggan/
│   ├── wukong/
│   └── vqdm/
└── forensynths/         # Optional cross-dataset
```

**Critical:** Resize all images to 224×224 on download and save as JPEG Q=95 to normalize compression bias (the Unbiased GenImage paper showed this matters enormously).

---

## Phase 1: Baseline — CLIP Linear Probe (Days 3-5)

This is your **first SOTA candidate** for the required model survey.

### What you're building:
Freeze CLIP ViT-B/16, extract the `[CLS]` embedding (512-dim), train a linear classifier (1 layer) on top.

```python
import open_clip
import torch.nn as nn

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='laion2b_s34b_b88k'
)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

class CLIPLinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = clip_model.visual
        self.head = nn.Linear(512, 2)

    def forward(self, x):
        with torch.no_grad():
            features = self.clip(x)
        return self.head(features)
```

### Training config:
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR, 20 epochs
- **Batch size:** 32 (should fit in 16GB MPS)
- **Data augmentation:** RandomHorizontalFlip, CLIP's own preprocessing (Resize 224, CenterCrop, Normalize)
- **Train on:** SD v1.4 subset
- **Evaluate on:** All 8 GenImage test generators + ForenSynths

### Expected results:
Based on published literature, a CLIP linear probe trained on one generator typically achieves ~85-92% accuracy in-distribution and ~70-85% cross-generator AUC. Record these — they're your baseline.

### Deliverable:
A table of **Accuracy + AUC per generator** that you'll include in your final report.

---

## Phase 2: Second Baseline — AIDE-style Hybrid (Days 6-9)

This is your **second SOTA candidate**.

Replicate a simplified version of the AIDE approach: combine CLIP features with frequency-domain features.

### DCT Frequency Branch:
```python
import scipy.fftpack as fft
import numpy as np

def extract_dct_features(image_np):
    """Convert RGB image to DCT spectral map."""
    gray = np.mean(image_np, axis=2)  # Grayscale
    dct_coeffs = fft.dct(fft.dct(gray.T, norm='ortho').T, norm='ortho')
    # Log-scale for better dynamic range
    dct_log = np.log1p(np.abs(dct_coeffs))
    return dct_log  # Shape: (224, 224)
```

Feed the DCT map into a small CNN (ResNet-18 pretrained, also frozen except last layer, or a custom 4-conv-layer net).

### Fusion:
```python
class HybridDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_backbone = clip_model.visual  # Frozen
        self.freq_backbone = SmallCNN(in_channels=1, out_dim=256)  # Trainable
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, rgb, dct):
        with torch.no_grad():
            clip_feat = self.clip_backbone(rgb)
        freq_feat = self.freq_backbone(dct)
        fused = torch.cat([clip_feat, freq_feat], dim=1)
        return self.classifier(fused)
```

### Training:
Same config as Phase 1, but now you're training the freq_backbone + classifier (~5M params). Still very feasible on MPS with batch_size=16.

---

## Phase 3: Your Improvements + Ablations (Days 10-16)

The course requires **at least two** of: architectural change, training strategy change, objective/loss redesign, efficiency improvement, domain adaptation. Here's what to do:

### Improvement 1: Learnable Frequency-Guided Attention (Architectural Change)

Instead of simple concatenation, add a **cross-attention** mechanism where frequency features modulate which CLIP spatial tokens to attend to:

```python
class FreqGuidedAttention(nn.Module):
    def __init__(self, clip_dim=512, freq_dim=256):
        super().__init__()
        self.query = nn.Linear(freq_dim, 128)
        self.key = nn.Linear(clip_dim, 128)
        self.value = nn.Linear(clip_dim, 128)
        self.out = nn.Linear(128, 256)

    def forward(self, clip_tokens, freq_feat):
        # clip_tokens: (B, num_patches, 512) from intermediate ViT layer
        # freq_feat: (B, 256)
        q = self.query(freq_feat).unsqueeze(1)       # (B, 1, 128)
        k = self.key(clip_tokens)                      # (B, N, 128)
        v = self.value(clip_tokens)                    # (B, N, 128)
        attn = torch.softmax(q @ k.transpose(-1,-2) / 11.3, dim=-1)
        out = (attn @ v).squeeze(1)
        return self.out(out)
```

This is genuinely novel — you're using frequency artifacts to *guide* where in the image the model looks. This directly enables your "which part looks AI" requirement.

### Improvement 2: Robustness-Aware Training Strategy (Training Strategy Change)

Apply **random JPEG compression (Q=50-100), Gaussian blur (σ=0-2), and resize-back degradation** during training. This simulates social media pipelines and dramatically improves cross-generator robustness.

```python
from torchvision import transforms

robustness_augmentations = transforms.Compose([
    transforms.RandomApply([
        transforms.Lambda(lambda x: jpeg_compress(x, q=random.randint(50, 100)))
    ], p=0.5),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.3),
    transforms.RandomApply([
        transforms.Resize(112),  # Downsample
        transforms.Resize(224),  # Upsample back
    ], p=0.3),
])
```

### Ablation Study Table (required for report):

| Model Variant | SD v1.4 Acc | Cross-Gen Avg AUC | Degraded Avg AUC |
|---|---|---|---|
| CLIP Linear Probe (baseline 1) | — | — | — |
| AIDE-style Hybrid (baseline 2) | — | — | — |
| + Freq-Guided Attention (Improv. 1) | — | — | — |
| + Robustness Augmentation (Improv. 2) | — | — | — |
| Full model (both improvements) | — | — | — |
| Full model - freq branch (ablation) | — | — | — |
| Full model - robustness aug (ablation) | — | — | — |

---

## Phase 4: Grad-CAM Explainability (Days 14-16)

This is what makes your demo shine — "why is it AI and which part."

### How it works:
Use `pytorch-grad-cam` on the CLIP ViT's attention layers:

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Target the last attention block of CLIP ViT
target_layer = model.clip_backbone.transformer.resblocks[-1].ln_1

cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=img_tensor, targets=None)

# Overlay heatmap on original image
visualization = show_cam_on_image(original_img, grayscale_cam[0], use_rgb=True)
```

**For the "why it's AI" explanation:** Extract the top-K highest-activation patches from the Grad-CAM output. Map them to semantic descriptions using CLIP's text encoder:

```python
artifact_descriptions = [
    "unnatural texture patterns",
    "inconsistent lighting",
    "distorted facial features",
    "blurred or repeated details",
    "unnatural edge transitions",
    "color banding artifacts",
]
# Score each description against the high-activation patches
```

This gives you a text explanation alongside the heatmap.

---

## Phase 5: Web Application (Days 17-21)

### Backend (FastAPI):
```
backend/
├── main.py              # FastAPI app
├── model.py             # Model loading + inference
├── gradcam.py           # Grad-CAM generation
├── explain.py           # Text explanation generation
├── models/              # Saved .pth checkpoint
└── requirements.txt
```

Key endpoint:
```python
@app.post("/detect")
async def detect_image(file: UploadFile):
    image = load_and_preprocess(file)
    prediction, confidence = model.predict(image)
    heatmap = generate_gradcam(model, image)
    explanation = generate_explanation(model, image, heatmap)
    return {
        "verdict": "AI-Generated" if prediction == 1 else "Real",
        "confidence": float(confidence),
        "heatmap": base64_encode(heatmap),
        "explanation": explanation,  # List of reasons
    }
```

### Frontend (React + Vite):
```
frontend/
├── src/
│   ├── App.jsx           # Main layout
│   ├── UploadZone.jsx    # Drag-and-drop upload
│   ├── ResultPanel.jsx   # Verdict + confidence meter
│   ├── HeatmapOverlay.jsx # Grad-CAM visualization
│   └── ExplanationCard.jsx # "Why it's AI" reasons
├── package.json
└── vite.config.js
```

**Design:** Single-page app. User drops an image → animated loading → verdict appears with confidence bar → heatmap overlay shows suspect regions → explanation cards list the reasons.

### Disk budget for the app:
- Model checkpoint: ~400MB (CLIP ViT-B/16 + freq branch + head)
- Node modules: ~200MB
- Python env: already installed
- Total additional: ~600MB — fits within your 15GB budget

---

## Phase 6: Report + Video (Days 22-25)

### Report structure (follow course rubric):
1. **Problem formulation** — what, why, success criteria
2. **Dataset** — GenImage subset, preprocessing, bias mitigation
3. **SOTA survey** — CLIP linear probe vs AIDE-style hybrid (with numbers)
4. **Our approach** — architecture diagram, frequency-guided attention, robustness training
5. **Experiments** — ablation table, cross-generator results, robustness curves
6. **Demo** — screenshots of web app, Grad-CAM examples
7. **Conclusion** — what worked, limitations, future work

### Demo video (2-3 min):
Screen-record the web app. Upload a real photo → show "Real" verdict. Upload a Midjourney image → show "AI" verdict with heatmap + explanation. Show the evaluation dashboard with cross-generator accuracy tables.

---

## Where to Run What

| Task | Where | Why |
|---|---|---|
| Data download + preprocessing | MacBook | Disk I/O, no GPU needed |
| CLIP feature extraction | MacBook (MPS) | Forward-pass only, fits in 16GB |
| Linear probe training | MacBook (MPS) | ~1M params, trains in minutes |
| Hybrid model training | MacBook (MPS) | ~5M trainable params, batch=16 works |
| Full model with ablations | **Google Colab (T4)** | Faster iteration for many experiments |
| Grad-CAM generation | MacBook (MPS) | Single-image inference |
| Web app development | MacBook | Standard web dev |
| Final demo | MacBook | Everything runs locally |

**Colab tip:** Upload your training script + a zipped 2GB dataset subset to Google Drive. Mount Drive in Colab. Train there, download the `.pth` checkpoint (~400MB) to your MacBook.

---

## Timeline Summary

| Days | Phase | Deliverable |
|---|---|---|
| 1-2 | Dataset download + preprocessing | Clean data/ folder, preprocessing script |
| 3-5 | Baseline 1: CLIP linear probe | Accuracy/AUC table for all generators |
| 6-9 | Baseline 2: AIDE-style hybrid | Comparative results table |
| 10-13 | Improvement 1: Freq-guided attention | Architecture code + initial results |
| 14-16 | Improvement 2: Robustness training + ablations | Full ablation table |
| 14-16 | Grad-CAM explainability | Heatmap generation pipeline |
| 17-21 | Web application | Working FastAPI + React demo |
| 22-25 | Report + video | Final deliverables |

---

## File/Folder Structure

```
aidetect/
├── data/                    # Dataset (gitignored)
├── src/
│   ├── dataset.py           # DataLoader with DCT extraction
│   ├── models/
│   │   ├── clip_probe.py    # Phase 1 baseline
│   │   ├── hybrid.py        # Phase 2 baseline
│   │   └── freq_guided.py   # Phase 3 final model
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Cross-generator eval
│   ├── gradcam_utils.py     # Grad-CAM wrapper
│   └── augmentations.py     # Robustness augmentations
├── backend/                 # FastAPI app
├── frontend/                # React app
├── notebooks/               # Colab notebooks
├── scripts/
│   ├── download_data.sh
│   └── preprocess.sh
├── results/                 # Saved metrics, plots
├── checkpoints/             # Model weights
└── README.md
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| 15GB disk fills up | Aggressively resize images to 224×224 on download; delete raw zips after extraction; use only SD v1.4 train split |
| MPS training crashes | Set `PYTORCH_ENABLE_MPS_FALLBACK=1`; reduce batch size to 8 if OOM; offload to Colab |
| Cross-generator accuracy is low | This is expected and actually interesting — document it honestly. The course says no penalty if advanced methods don't work, as long as you show effort |
| Grad-CAM on ViT is noisy | Use Attention Rollout as a fallback — it often gives cleaner maps for transformers |
| Time pressure | Phases 1-2 are the minimum viable project. Phases 3+ are what earn you the top marks. Prioritize accordingly |
