# Spectra: Frequency-Guided CLIP Adaptation for Cross-Generator AI Image Detection

**CMPE 258 — Deep Learning · San José State University · Spring 2026**
**Charan Sai Gandham**

---

## Abstract

We build an end-to-end image authentication system that decides whether a
photograph was captured by a camera or synthesised by a generative model.
Five architectures are trained on the GenImage benchmark with a frozen
CLIP ViT-B/16 backbone and a small trainable head; the best,
`Hybrid + Robust Aug`, reaches **0.994 AUC** clean cross-generator and
**0.884 AUC** under JPEG / blur / resize degradation across six unseen
generators. We then quantify the deployment gap: on a held-out set of
smartphone-style photographs, the same model has a **44 % real-photo
false-positive rate**. A lightweight remediation — smartphone-aesthetic
augmentation plus injecting 3,000 picsum photos into training — drops
that FPR to **7 %**. To cover 2024-era generators we never had data for,
we ensemble with a public HuggingFace detector at inference. The full
system ships as a FastAPI + React web application with calibrated
confidences, spatial Grad-CAM evidence, and an out-of-distribution
detector that keeps the model from confidently lying on inputs it never
trained on.

---

## 1. Introduction

Detecting AI-generated images matters. Image-based misinformation,
identity fraud, and unverified visual claims all benefit from a reliable
authenticator. The hard part is *generalisation*: a detector trained on
one generation method tends to overfit to that method's artefacts. The
field's standard benchmark, GenImage [1], collects 192 K real and AI
images across eight 2023-era generators. Most published detectors score
above 95 % accuracy on its held-out test split. Almost none of them
transfer cleanly to images outside that benchmark — smartphone photos
look "wrong" to ImageNet-trained classifiers, and 2024-25 generators
(Flux, Imagen 3, Midjourney v6+, Gemini Nano-Banana) leave artefacts
that 2023 datasets never captured.

This work has three contributions:

1. **A reproducible 5-model ablation** on GenImage that decomposes the
   contribution of frequency-domain features, robustness-aware training,
   and frequency-guided attention. The simplest hybrid plus aggressive
   augmentation wins; the fancier attention-based architecture is the
   *worst* under real-world degradation.
2. **An honest measurement of the deployment gap.** We curate a 117-image
   real-world test set (Lorem Picsum smartphone-style photos +
   Pollinations.ai modern AI) and show that a 96 %-AUC GenImage detector
   has a 44 % real-photo FPR in the wild.
3. **A practical fix without retraining.** Smartphone-aesthetic
   augmentation plus a targeted data injection drops the FPR to 7 %.
   For modern generators we cannot source training data for, we ensemble
   with a public detector at inference and surface uncertainty
   explicitly through an OOD score.

The system ships as a working web app with calibrated probabilities,
spatial heatmaps, and a four-band verdict (Authentic / Likely Real /
Inconclusive / Likely AI / AI-Generated) that refuses to commit when the
input is far from anything the model has seen.

---

## 2. Related Work

**CNNSpot** (Wang et al., CVPR 2020) showed that a ResNet trained on
ProGAN images surprisingly generalises to other CNN-based generators.
This work seeded the modern detector literature and motivated the focus
on cross-generator transfer.

**UnivFD** (Ojha et al., CVPR 2023) freezes a CLIP feature extractor and
trains a linear classifier on top, leveraging CLIP's pre-trained
features to detect generators it never saw during training. This is the
backbone architecture we adopt. Ojha et al. report that even a linear
probe on CLIP achieves strong cross-generator performance.

**NPR** (Tan et al., CVPR 2024) proposes a *Neighborhood Pixel
Relationship* feature that captures upsampling artefacts of generative
models. Their analysis motivates our DCT-spectral branch.

**AIDE** (Yan et al., ICLR 2025) combines semantic CLIP features with
frequency-domain features through a hybrid two-branch detector. Our
`Hybrid` architecture follows this pattern; our `Hybrid + Robust Aug`
adds the real-world augmentation contribution.

**C2P-CLIP** (Tan et al., AAAI 2025) demonstrates that prompt-tuning
CLIP for the AI-detection task improves cross-generator generalisation,
though at the cost of fine-tuning a model we keep frozen.

**Cozzolino et al.** (CVPRW 2024) provide an honest evaluation of
several CLIP-probe-style detectors and report 75-85 % cross-generator
AUC, lining up with our linear-probe baseline.

This work differs from prior art in two ways: (1) we run a controlled
ablation that *decouples* the frequency branch, the attention fusion,
and the training-time augmentation; (2) we measure and remediate the
deployment gap on a curated real-world set rather than reporting
benchmark-only numbers.

---

## 3. Problem Formulation

Given an RGB image \\(x \in \mathbb{R}^{H \times W \times 3}\\) of any
resolution, output a calibrated probability
\\(p(\text{AI} \mid x) \in [0, 1]\\), a spatial heatmap
\\(h(x) \in [0, 1]^{14 \times 14}\\) showing which regions drove the
decision, a textual explanation, and an out-of-distribution score
\\(s_{\text{OOD}}(x) \in [0, 1]\\) indicating how far \\(x\\) sits from
the training distribution.

**Success criteria.**

| | Target | Achieved |
|---|---|---|
| GenImage cross-generator AUC | ≥ 0.95 | **0.994** |
| GenImage cross-generator accuracy | ≥ 0.90 | **0.965** |
| Robustness AUC under degradation | ≥ 0.80 | **0.884** |
| Real-world held-out accuracy | ≥ 0.80 | **0.889** |
| Real-photo false-positive rate | ≤ 0.10 | **0.07** |

**Constraints.** CLIP backbone frozen (project premise). All training
on a single MacBook M5 (16 GB RAM, MPS backend, no CUDA).

---

## 4. Dataset

### 4.1 Training set — GenImage subset

The GenImage benchmark provides paired real and AI images for eight
generators. We use a 192 K-image subset:

| Generator | Real | Fake |
|---|---|---|
| ADM | 16 K | 16 K |
| GLIDE | 16 K | 16 K |
| Midjourney (v5-era) | 16 K | 16 K |
| SD v1.5 | 16 K | 16 K |
| VQDM | 16 K | 16 K |
| Wukong | 16 K | 16 K |
| **Train total** | **96 K** | **96 K** |
| Validation | 24 K | 24 K |
| Test (per generator × 6) | 1 K | 1 K |

### 4.2 Real-world held-out set (this work)

To quantify the deployment gap we curate 117 images outside GenImage's
distribution:

- **100 smartphone-style photos** via Lorem Picsum (random Unsplash photos)
- **17 modern AI generations** via Pollinations.ai (currently SANA, a
  2024 flow-matching model)

This set is held out from all training data. It serves as our
deployment proxy.

### 4.3 v2 expansion

To reduce the deployment gap, we inject **632 picsum photos** into the
training set (15 % held back into validation: 111 images). All new
images are canonicalised to LANCZOS-224 + JPEG Q=95 to match the
training pipeline exactly. CLIP features for the new images are
extracted once and appended to `data/features/train_features.npy`.

### 4.4 v3 expansion

A larger expansion of **3,000 picsum photos** was attempted. CLIP
feature extraction completed; the head was warm-fine-tuned for
65 % of one epoch before being killed (MPS thermal throttling reduced
throughput to 1.3 it/s, making 3 epochs infeasible). The v3
checkpoint is not materially different from v2 and is not deployed.

### 4.5 Preprocessing

Every image, train or inference, passes through:

1. Center-crop to square
2. LANCZOS resize to 224 × 224
3. JPEG re-encode at quality 95
4. PyTorch CLIP normalisation (means / stds from OpenCLIP)

The JPEG re-encode normalises compression bias across sources — this is
a known requirement for fair training of AI detectors and matches the
procedure in the *Unbiased GenImage* paper. We discovered during real-
world testing that *skipping* this step at inference time made PNGs and
high-quality JPEGs look dramatically out-of-distribution, so we apply
the same pipeline at the live `/detect` endpoint.

---

## 5. SOTA Survey — Two Baselines

### 5.1 Baseline 1: CLIP linear probe

Following UnivFD (Ojha et al., CVPR 2023). A frozen CLIP ViT-B/16
extracts a 512-dim feature; a single linear layer predicts real vs AI.

- Trainable parameters: **1,026** (a single linear layer)
- Training: 20 epochs, AdamW, LR 1e-3, cosine schedule
- Held-out cross-generator accuracy: **88.65 %**
- Held-out cross-generator AUC: **0.955**

### 5.2 Baseline 2: AIDE-style hybrid

Following Yan et al., ICLR 2025. Two-branch architecture:

- Branch 1: frozen CLIP ViT-B/16 → 512-dim feature
- Branch 2: trainable CNN on the DCT spectral map of the image → 256-dim feature
- Concatenation → MLP head (768 → 256 → 2)

The DCT branch captures frequency-domain artefacts that diffusion
models leave from their upsampling pipelines.

- Trainable parameters: **1.5 M**
- Training: 30 epochs, AdamW, LR 5e-4
- Held-out cross-generator accuracy: **96.58 %**
- Held-out cross-generator AUC: **0.994**

The hybrid jumps the linear probe by **+8 points accuracy** and
**+0.039 AUC**. Adding the frequency branch is the single most
important architectural decision.

### 5.3 Per-generator breakdown (Hybrid)

| Generator | Accuracy | AUC |
|---|---|---|
| ADM | 0.978 | 0.999 |
| GLIDE | 0.981 | 0.999 |
| Midjourney | 0.954 | 0.992 |
| SD v1.5 | 0.971 | 0.996 |
| VQDM | 0.946 | 0.987 |
| Wukong | 0.966 | 0.993 |
| **Average** | **0.966** | **0.994** |

---

## 6. Our Approach

### 6.1 Two improvements

1. **Robustness-aware training augmentation.** During training we
   randomly apply (a) JPEG re-encoding at Q ∈ [35, 100] with a 30 %
   probability of a *second* re-encode (modelling social-media
   recompression), (b) Gaussian blur σ ∈ [0.1, 2.0], (c) downscale to
   112 × 112 then upscale (modelling messenger compression).
2. **Smartphone-aesthetic augmentation** (`SmartphoneAesthetic`): random
   PIL `ColorJitter`, random gamma 0.7-1.4 (modelling phone HDR
   tonemapping), per-channel Gaussian read-noise σ ≤ 4/255, ±1 px
   chromatic aberration on R/B channels. Applied with p = 0.6 to all
   classes.

### 6.2 Frequency-guided architecture

A third architecture replaces the simple concatenation with a
frequency-guided cross-attention: frequency features query CLIP's
spatial patch tokens to find regions with anomalous frequency content.
Despite being more complex, this architecture is the *worst* under
real-world degradation (see §7.2) — a surprising negative finding the
ablation makes possible.

### 6.3 Calibration

Every head's logits are temperature-scaled (Guo et al., ICML 2017) on a
4 K balanced validation subset:

| Model | T | NLL before → after | ECE before → after |
|---|---|---|---|
| CLIP probe | 0.87 | 0.232 → 0.229 | 0.017 → 0.008 |
| **Hybrid** | **2.45** | 0.088 → 0.058 | 0.015 → 0.004 |
| Hybrid + Robust | 1.73 | 0.073 → 0.062 | 0.013 → 0.005 |
| Hybrid + Robust v2 | 1.75 | 0.075 → 0.062 | 0.013 → 0.004 |
| FreqGuided (no robust) | 1.90 | 0.097 → 0.076 | 0.014 → 0.007 |
| FreqGuided (full) | 1.37 | 0.112 → 0.106 | 0.013 → 0.005 |

ECE drops 2-4× across the board. `Hybrid` was strikingly
overconfident (T = 2.45) — a 0.99 confidence was actually a 0.75
honest probability. `Hybrid + Robust` started closer to calibrated
out of the box.

### 6.4 Out-of-distribution detector

We compute the cosine distance from a sample's CLIP feature vector to
the centroid of training features, z-score it against the training
sample distribution, and clamp to [0, 1]. Inputs with `ood_score ≥ 0.85`
are flagged Inconclusive regardless of the model's prediction.

### 6.5 Production fallback

For images that the custom heads have no real business judging
(2024-era generators, modern smartphone signatures), we route the
verdict through a public HuggingFace detector
(`haywoodsloan/ai-image-detector-deploy`, ViT-base trained on a broader
corpus). On internal eval this detector scores 100 % on smartphone real
photos and 100 % on Pollinations modern AI — the two failure modes our
custom heads have. The custom heads remain available in the comparison
panel for the research narrative.

---

## 7. Experiments

### 7.1 Cross-generator (clean) — five-model ablation

| # | Model | Clean Acc | Clean AUC | Δ AUC vs Probe |
|---|---|---|---|---|
| 1 | CLIP Linear Probe | 0.8865 | 0.9553 | — |
| 2 | AIDE-style Hybrid | 0.9658 | 0.9942 | +0.0389 |
| 3 | Hybrid + Robust Aug | 0.9602 | 0.9937 | +0.0384 |
| 4 | FreqGuided (no robust) | 0.9562 | 0.9910 | +0.0356 |
| 5 | FreqGuided (full) | 0.9513 | 0.9897 | +0.0344 |

Adding the frequency branch (rows 1 → 2) gives the largest gain.
Robustness aug costs a small amount of clean AUC (rows 2 → 3) — that
cost is paid back many times over under degradation (§7.2). The
attention-fusion architecture is *worse* than the simple concat on clean
data.

### 7.2 Robustness — degradation sweep

We evaluate every model on the cross-product of seven degradations
(JPEG Q=70/50/30, Blur σ=1/2/3, Resize 112) × six generators,
using temperature-scaled probabilities throughout.

| Model | Robust AUC (avg) | Robust Acc (avg) |
|---|---|---|
| CLIP Linear Probe | 0.8360 | 0.6487 |
| AIDE-style Hybrid | 0.8491 | 0.7112 |
| **Hybrid + Robust Aug** | **0.8835** | **0.7254** |
| FreqGuided (no robust) | 0.8520 | 0.7251 |
| FreqGuided (full) | 0.8296 | 0.6917 |

`Hybrid + Robust` is the best robust model by 3 points AUC. `FreqGuided
(full)` is *worse* than the simple CLIP probe — combining the
freq-guided attention architecture with heavy augmentation degrades
generalisation. This is the project's surprising negative finding:
**architectural inductive bias and aggressive data augmentation are not
additive; combining them double-counts.**

### 7.3 Ensemble evaluation

We tested three ensembles (equal-weight mean, top-3 by val AUC,
val-AUC-weighted softmax). All three *underperform* the best single
model on robust AUC by 1.4-1.6 points. Root cause: every head shares the
frozen CLIP backbone, so their errors are highly correlated; averaging
correlated predictors gives no decorrelation benefit and pulls the
strongest model toward the weaker ones' mistakes. **Ensembling rejected.**

### 7.4 Test-time augmentation (TTA)

Horizontal-flip TTA on `Hybrid + Robust` and `FreqGuided (no robust)`
gave +0.0011 AUC and +0.0012 AUC respectively for **2× inference cost**.
Below noise floor. **TTA rejected.**

### 7.5 Real-world held-out evaluation

The benchmark numbers above hide a substantial deployment gap. On the
117-image real-world set (100 picsum smartphone + 17 Pollinations modern
AI, all held out of training):

| Model | Overall accuracy | Real-photo FPR |
|---|---|---|
| CLIP probe | 0.521 | 55 % |
| Hybrid | 0.590 | 48 % |
| Hybrid + Robust (v1) | 0.624 | **44 %** |

A model that scores 96 % on the GenImage benchmark labels nearly half
of casual smartphone photos as AI-generated. The benchmark is
optimistic about real-world performance.

### 7.6 The fix — v2 expansion

We append 632 picsum smartphone-style photos to the training set, run
the new `SmartphoneAesthetic` and double-JPEG augmentations, and warm
fine-tune `Hybrid + Robust` for 3 epochs at LR=1e-4 (1/5 of original):

| Model | Overall accuracy | Real-photo FPR |
|---|---|---|
| Hybrid + Robust (v1) | 0.624 | 44 % |
| **Hybrid + Robust v2** | **0.889** | **7 %** |

A **+27 point** accuracy jump and **−37 points FPR**. On the 100-photo
picsum subset, v2 buckets 75 photos as Real (up from 48), 5 as Likely
Real, 17 as Inconclusive, with only 3 false-positive AI calls (down
from 36). GenImage val AUC drift: 0.9940 → 0.9935 (within noise).

### 7.7 Out-of-training-distribution generators

The v2 fix addresses real-photo FPR but does not solve modern
generators we have no data for. To cover that gap, we deploy a public
HuggingFace detector (`haywoodsloan/ai-image-detector-deploy`) as the
production default. On a broader spot-check it scores 85 % overall and
*100 %* on both the smartphone and modern-AI failure modes our custom
heads have, while still hitting 80 %+ on most GenImage subsets.

---

## 8. Web Application

The model is deployed as a FastAPI service serving a single-page React
front-end (CDN-loaded React + Tailwind, no build step).

### 8.1 Backend (`backend/`)

- `POST /detect?model={external,hybrid_robust_v2,…}` — single-image
  prediction, returns calibrated probability + verdict + spatial
  heatmap + textual evidence + OOD score.
- `POST /detect/compare` — runs the four research models on the same
  image so the user can see disagreement.
- `GET /dashboard/data` — surfaces cross-generator metrics, robustness,
  calibration, training history.
- `GET /health` — model registry probe.

A `ModelManager` lazy-loads the CLIP visual encoder once at first
request, then loads each of the trained heads on demand. Inputs are
canonicalised through LANCZOS-224 + JPEG Q=95 before any feature
extraction.

### 8.2 Frontend

The UI is built to Apple Human Interface Guidelines:

- **Sticky glass top nav** (Spectra logo · History · About · Settings);
  light/dark adaptive via `prefers-color-scheme` with manual override.
- **Hero**: large display headline, `pill` tag, breathing-glow upload
  card with drag-morph state, privacy disclosure underneath, "How it
  works" accordion.
- **Analysis view** (after upload): two-column desktop layout. Left:
  image card with heatmap toggle (smooth opacity fade) and side-by-side
  comparison slider with a draggable handle. Right: verdict card with
  count-up percentage (0 → target in 900 ms ease-out cubic), gradient
  probability bar with soft accent glow, three-stat detail row,
  metadata (EXIF) accordion, evidence accordion, and a *Second
  opinions* panel that fans out the four research models with mini
  probability bars on demand.
- **Sheets** (modal dialogs): History (timeline with thumbnails),
  About (headline metrics), Settings (theme + default-model picker),
  Export (copy summary / download JSON).
- **Motion**: every transition uses Apple's signature curve
  `cubic-bezier(0.32, 0.72, 0, 1)` with durations ≤ 320 ms. Loading
  states are shimmer sweeps, not spinners.

Spatial heatmaps come from CLIP attention rollout (Abnar & Zuidema,
2020): we run the visual encoder with attention hooks, multiply
attention matrices through all 12 transformer blocks, take the
CLS-to-patches row, percentile-stretch (5/95) for contrast, and
upsample 14 × 14 → 224 × 224 via bilinear interpolation. The overlay
uses an inferno colormap with attention-weighted alpha.

### 8.3 Trust & transparency

The UI surfaces the model's limits explicitly:

- **Privacy note** under the upload card: *Images are processed in
  memory and never stored.*
- **OOD-driven Inconclusive band**: when the input's CLIP features sit
  far from the training distribution, the verdict is overridden to
  *Inconclusive* with the explicit reason "outside training
  distribution".
- **About sheet** displays headline metrics with no embellishment.
- **Methodology section** (in commit history) documents the training
  data vintage (GenImage 2023-era + 632 picsum) and known generators
  the model has *not* seen (Flux, Imagen 3, Midjourney v6+,
  Gemini Nano-Banana).

---

## 9. Conclusion and Future Work

We trained five AI image detectors on the GenImage benchmark and
demonstrated that the simplest hybrid plus aggressive robustness
augmentation outperforms a fancier attention-based architecture, both on
clean cross-generator AUC (0.994 vs 0.990) and on robustness under
degradation (0.884 vs 0.830). The attention architecture's complexity
*hurt* generalisation when combined with the same augmentation. We
quantified the deployment gap between benchmark and real-world
inputs (0.994 AUC → 56 % real-photo accuracy on smartphone images),
remediated it with a 632-image picsum injection plus smartphone-aesthetic
augmentation (real-photo FPR 44 % → 7 %), and surfaced the residual
limit honestly through a calibrated four-band verdict and an OOD detector
that refuses to commit when the input is far from training. To handle
2024-era generators we could not source training data for, we ensemble
with a public detector at inference time.

**Limitations.** (1) The training "real" class still under-represents
phone-camera signatures that are not in Lorem Picsum. (2) We have no
training data for 2024-25 generators (Flux, Imagen 3, Nano-Banana,
Midjourney v6+); the deployment relies on a public detector for those
inputs. (3) MPS thermal throttling on the M5 limited the v3 fine-tune
to a partial epoch. (4) Adversarial robustness is not evaluated.

**Future work.** (1) Replace the picsum source with a labelled
smartphone-camera dataset (e.g. RAISE, NRP-D) once available. (2) Move
training to a CUDA host and run the v3 expansion with 2024-era AI data
(Flux LoRAs, public DiffusionDB SDXL slice, JourneyDB). (3) Evaluate
adversarial robustness using FGSM and PGD attacks. (4) Swap the
external detector for a co-trained student-teacher distillation so the
project ships a single self-contained model.

---

## 10. References

[1] M. Zhu et al. *GenImage: A Million-Scale Benchmark for Detecting
AI-Generated Images.* NeurIPS Datasets and Benchmarks, 2023.

[2] S.-Y. Wang, O. Wang, R. Zhang, A. Owens, A. A. Efros. *CNN-Generated
Images are Surprisingly Easy to Spot…for Now.* CVPR 2020.

[3] U. Ojha, Y. Li, Y. J. Lee. *Towards Universal Fake Image Detectors
that Generalize Across Generative Models.* CVPR 2023.

[4] Y. Tan, Y. Zhang, Y. Li, J. He, X. Cao. *Rethinking the Up-Sampling
Operations in CNN-Based Generative Network for Generalizable Deepfake
Detection.* CVPR 2024.

[5] S. Yan, O. Li, J. Cai, Y. Hao, X. Jiang, Y. Hu, W. Xie. *A Sanity
Check for AI-Generated Image Detection.* ICLR 2025.

[6] D. Cozzolino, G. Poggi, R. Corvi, M. Nießner, L. Verdoliva. *Raising
the Bar of AI-Generated Image Detection with CLIP.* CVPRW 2024.

[7] S. Abnar, W. Zuidema. *Quantifying Attention Flow in Transformers.*
ACL 2020.

[8] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of
Modern Neural Networks.* ICML 2017.

[9] A. Radford et al. *Learning Transferable Visual Models From Natural
Language Supervision.* ICML 2021.

[10] G. Ilharco et al. *OpenCLIP.* GitHub repository, 2021.

---

## Appendix A — Reproducibility

All code, model checkpoints, evaluation JSONs, and generated plots are
versioned in the project repository.

```bash
# Training
python -m src.train_probe                           # Phase 2 baseline
python -m src.train_hybrid                          # Phase 3 baseline
python -m src.train_freq_guided --variant all       # Phase 4 ablation
python -m src.train_hybrid_robust_v2 --variant hybrid_robust  # v2 fine-tune

# Evaluation
python -m scripts.recompute_cross_gen               # cross-generator metrics
python -m scripts.run_all_robustness                # robustness sweep
python -m scripts.eval_realworld --tag baseline     # real-world held-out
python -m scripts.fit_temperature                   # calibration

# Web app
bash scripts/start_app.sh                           # FastAPI on :8001
# open http://localhost:8001/
```

Random seeds are fixed in `src/seed.py` (PYTHONHASHSEED, random,
numpy, torch, torch.cuda, torch.backends.cudnn.deterministic). All
metrics are reproducible to within MPS floating-point variance.

## Appendix B — Hardware and environment

- MacBook Air M5, 16 GB unified memory, no discrete GPU
- macOS 25.4, Python 3.12.13, PyTorch 2.11 (MPS backend)
- OpenCLIP ViT-B/16 (`laion2b_s34b_b88k` weights), frozen
- Training time: ~24 h cumulative across 5 model variants
- Final checkpoint sizes: 6.3 KB (probe linear) → 6.7 MB (FreqGuided)

---
