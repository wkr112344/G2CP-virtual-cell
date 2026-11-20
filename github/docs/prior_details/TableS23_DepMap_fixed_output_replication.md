# Table S23. DepMap fixed-output evaluation-object replacement

| Category | Metric | vs CERES | vs Chronos | Δ_eval | 95% CI | Random baseline | Direction |
|---|---|---:|---:|---:|---|---:|---|
| Correlation | Mean per-cell-line Pearson | 0.8595 | 0.8226 | +0.0369 | [+0.0342, +0.0398] | — | higher |
| Correlation | Median per-cell-line Pearson | 0.8664 | 0.8307 | +0.0357 | [+0.0311, +0.0399] | — | higher |
| Correlation | Median per-cell-line Spearman | 0.8502 | 0.8263 | +0.0238 | [+0.0197, +0.0273] | — | higher |
| Correlation | Element-wise Pearson | 0.8503 | 0.8144 | +0.0359 | [+0.0331, +0.0388] | — | higher |
| Error | Mean RMSE | 0.2776 | 0.3301 | −0.0525 | [−0.0543, −0.0505] | — | lower |
| Top-k | Top-10 retention | 0.4298 | 0.3706 | +0.0592 | [+0.0438, +0.0757] | 0.0080 | higher |
| Top-k | Top-10 Jaccard | 0.2805 | 0.2315 | +0.0489 | [+0.0359, +0.0625] | — | higher |
| Top-k | Top-50 retention | 0.6602 | 0.6237 | +0.0365 | [+0.0299, +0.0432] | 0.0402 | higher |
| Top-k | Top-50 Jaccard | 0.4953 | 0.4557 | +0.0396 | [+0.0325, +0.0468] | — | higher |
| Top-k | Top-100 retention | 0.7095 | 0.6767 | +0.0329 | [+0.0276, +0.0382] | 0.0804 | higher |
| Top-k | Top-100 Jaccard | 0.5523 | 0.5134 | +0.0389 | [+0.0327, +0.0452] | — | higher |
| Top-k | Top-10% (k=124) retention | 0.7133 | 0.6790 | +0.0343 | [+0.0296, +0.0389] | 0.0997 | higher |
| Top-k | Top-10% (k=124) Jaccard | 0.5566 | 0.5160 | +0.0406 | [+0.0351, +0.0460] | — | higher |

Notes: Δ_eval = M(P_fixed, E_CERES) − M(P_fixed, E_Chronos). Both arms use the same frozen DeepDEP prediction matrix. DeepDEP was developed in a CERES-era target setting, so the CERES–Chronos difference reflects both reference-product choice and model–target matching. The table quantifies fixed-output performance and top-k recovery across two highly concordant released references. 95% intervals are paired cell-line cluster bootstrap percentile intervals (B=2000; seed=20260926). Random baselines are shown only for retention metrics.