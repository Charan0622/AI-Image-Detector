# Ablation Study Results

| # | Model Variant | Cross-Gen Avg AUC | Cross-Gen Avg Acc | vs Probe |
|---|---------------|-------------------|-------------------|----------|
| 1 | CLIP Linear Probe | 0.9479 | 0.8765 | +0.0000 |
| 2 | AIDE-style Hybrid | 0.9823 | 0.9329 | +0.0344 |
| 3 | freq_guided | 0.9731 | 0.9131 | +0.0252 |
| 4 | freq_guided_no_robust | 0.9760 | 0.9259 | +0.0281 |
| 5 | hybrid_robust | 0.9816 | 0.9291 | +0.0337 |