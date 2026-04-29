# Ablation Study Results

Cross-gen = averaged across the 6 unseen test generators (adm, glide, midjourney, sd15, vqdm, wukong). Robust = averaged over 7 degradations (JPEG Q=70/50/30, Blur σ=1/2/3, Resize 112) × the same generators.

| # | Model | Clean Acc | Clean AUC | JPEG-30 Acc | Blur-σ3 Acc | Resize Acc | Robust AUC (avg) | Robust Acc (avg) | Δ AUC vs Probe |
|---|-------|-----------|-----------|-------------|-------------|------------|------------------|------------------|----------------|
| 1 | CLIP Linear Probe | 0.8865 | 0.9553 | 0.7225 | 0.5225 | 0.7078 | 0.8360 | 0.6487 | +0.0000 |
| 2 | AIDE-style Hybrid | 0.9658 | 0.9942 | 0.7483 | 0.5420 | 0.7814 | 0.8491 | 0.7112 | +0.0389 |
| 3 | Hybrid + Robust Aug | 0.9602 | 0.9937 | 0.7228 | 0.5936 | 0.8025 | 0.8835 | 0.7254 | +0.0384 |
| 4 | FreqGuided (no robust) | 0.9562 | 0.9910 | 0.7208 | 0.6086 | 0.8036 | 0.8520 | 0.7251 | +0.0356 |
| 5 | FreqGuided (full, our final) | 0.9513 | 0.9897 | 0.6644 | 0.5906 | 0.7755 | 0.8296 | 0.6917 | +0.0344 |

## Robustness Breakdown (AUC averaged across generators)

| Model | Clean | JPEG-70 | JPEG-50 | JPEG-30 | Blur-1 | Blur-2 | Blur-3 | Resize-112 |
|-------|-------|---------|---------|---------|--------|--------|--------|------------|
| CLIP Linear Probe | 0.9546 | 0.9101 | 0.8891 | 0.8463 | 0.8565 | 0.7705 | 0.7071 | 0.8726 |
| AIDE-style Hybrid | 0.9943 | 0.9661 | 0.9491 | 0.9042 | 0.8160 | 0.7658 | 0.6363 | 0.9061 |
| Hybrid + Robust Aug | 0.9936 | 0.9617 | 0.9471 | 0.9035 | 0.8771 | 0.8278 | 0.7397 | 0.9276 |
| FreqGuided (no robust) | 0.9911 | 0.9300 | 0.9124 | 0.8762 | 0.8124 | 0.8045 | 0.7253 | 0.9034 |
| FreqGuided (full, our final) | 0.9898 | 0.9140 | 0.9000 | 0.8608 | 0.8118 | 0.7710 | 0.6835 | 0.8664 |
