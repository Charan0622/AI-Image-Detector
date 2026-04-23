# Test-Time Augmentation Results

TTA = mean logit of (original, horizontal-flip). Evaluated on 4 conditions:
clean + 3 hardest degradations (JPEG-30, Blur-σ3, Resize-112) across 6 generators.

| Model | Condition | No-TTA AUC (avg) | TTA AUC (avg) | Δ AUC |
|-------|-----------|------------------|---------------|-------|
| hybrid_robust | clean | 0.9936 | 0.9940 | +0.0004 |
| hybrid_robust | jpeg_q30 | 0.9035 | 0.9057 | +0.0022 |
| hybrid_robust | blur_s3 | 0.7569 | 0.7591 | +0.0022 |
| hybrid_robust | resize_112 | 0.9301 | 0.9297 | -0.0004 |
| **hybrid_robust** | **overall** | **0.8960** | **0.8971** | **+0.0011** |
| freq_guided_no_robust | clean | 0.9911 | 0.9916 | +0.0005 |
| freq_guided_no_robust | jpeg_q30 | 0.8762 | 0.8786 | +0.0024 |
| freq_guided_no_robust | blur_s3 | 0.7254 | 0.7267 | +0.0014 |
| freq_guided_no_robust | resize_112 | 0.9033 | 0.9040 | +0.0007 |
| **freq_guided_no_robust** | **overall** | **0.8740** | **0.8752** | **+0.0012** |
