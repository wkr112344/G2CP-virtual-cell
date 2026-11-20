# LINCS reference matrices (11,275 × 978)

This closes a reproduction gap: previously the paper's LINCS reference values were read on the fly from the public
level-5 products during analysis and were not distributed with the package; here we provide the **two reference
matrices written to disk**, so a reproducer does not need to download the 41 GB raw gctx.

| File | Content | Shape | md5 | Check |
|---|---|---|---|---|
| `Y_beta2020_11275x978.npy` | level5beta2020 (MODZ) reference values | 11,275 × 978, float32 | `32eac1ce…` | per-row Pearson against the beta-trained frozen prediction → drug-equal-weight **0.367987** (main text 0.3680) ✓ |
| `Y_dcic2021_11275x978.npy` | dcic2021 (Characteristic Direction) reference values | 11,275 × 978, float32 | `dece7c79…` | per-row Pearson against the same frozen prediction → drug-equal-weight **0.072386** (main text 0.0724) ✓ |

## Calibers (row-for-row identical to `P_frozen_multimetric_cohorts.py::pair_level_978()`)

- **Row order**: the (cell, drug) order of `../.../supplementary/frozen_multimetric_perrow_scores.csv` (11,275 rows)
- **Column order**: the `internal_col` of `lincs_index/93_gene_map_12328_to_978.json` (the 978 landmark genes)
- **Sources**: `g2cp_cache_beta_v2/y.npy` (level5beta2020), `g2cp_cache_2021/y.npy` (dcic2021)
- Export script: `build_lincs_reference_matrices.py` (this directory; it includes the two checks above and aborts if either fails)

## How a reproducer uses it

```python
import numpy as np, pandas as pd
T = np.load('Y_beta2020_11275x978.npy')                  # 11,275 × 978
P = np.load('<frozen_pred_matrix_beta_trained_11275x978.npy>')
r = [np.corrcoef(P[i], T[i])[0,1] for i in range(len(T))]
rows = pd.read_csv('frozen_multimetric_perrow_scores.csv')   # row-order reference
print(rows.assign(r=r).groupby('drug').r.mean().mean())       # ≈ 0.3680
```
