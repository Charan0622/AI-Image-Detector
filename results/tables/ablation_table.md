# Ablation Study Results

| # | Model Variant | Cross-Gen Avg AUC | Cross-Gen Avg Acc | vs Probe |
|---|---------------|-------------------|-------------------|----------|
| 1 | CLIP Linear Probe | 0.9558 | 0.8882 | +0.0000 |
| 2 | AIDE-style Hybrid | 0.9944 | 0.9652 | +0.0385 |
| 3 | freq_guided | 0.9904 | 0.9533 | +0.0345 |
| 4 | freq_guided_no_robust | 0.9911 | 0.9561 | +0.0353 |
| 5 | hybrid_robust | 0.9940 | 0.9619 | +0.0381 |