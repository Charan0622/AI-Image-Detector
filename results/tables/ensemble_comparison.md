# Ensemble Comparison

All probabilities are temperature-calibrated from val set.
Clean AUC = mean across 6 generators; Robust AUC = mean over 7 degradations x 6 generators.

| Model | Clean AUC | Robust AUC |
|-------|-----------|------------|
| clip_probe | 0.9546 | 0.8360 |
| hybrid | 0.9943 | 0.8835 |
| hybrid_robust | 0.9936 | 0.8908 |
| freq_guided_no_robust | 0.9911 | 0.8520 |
| freq_guided | 0.9898 | 0.8298 |
| ensemble_all | 0.9936 | 0.8748 |
| ensemble_top3 | 0.9913 | 0.8765 |
| ensemble_weighted | 0.9947 | 0.8754 |
