# DepMap raw artifacts · for the "shared-gene background control (residualized 2×2)" experiment

Generated 2026-09-29 (copy and checksum only; the raw data was not modified).

## Grid and row/column calibers (**read the matrices exactly as described here**)

- Row order of every 908 × 17393 matrix = `index/cell_order.txt`
- Column order = `index/target_genes.txt`, formatted as `SYMBOL (EntrezID)`, e.g. `A1BG (1)`
  → when matching against external gene names, strip the ` (id)` suffix first.
- Row order of `*_test.npy` = `index/test_cell_order.txt` (136 cells)
  → verified: **identical in order** to `test_cells` in `split.json`.
- The split comes from `index/split.json` (**seed 42**, locked during training, do not re-split):
  train 636 / val 136 / test 136.

## Directory

| Path | Content | Shape |
|---|---|---|
| `reference/Y_CERES.npy` | CERES reference | (908, 17393) float32 |
| `reference/Y_Chronos.npy` | Chronos reference | (908, 17393) float32 |
| `reference/common_observed_mask.npy` | common observed mask (NaN handling; do not impute) | (908, 17393) bool |
| `predictions_test/seed{0-4}_pred_ceres_arm_test.npy` | CERES-trained arm test predictions | (136, 17393) float32 |
| `predictions_test/seed{0-4}_pred_chronos_arm_test.npy` | Chronos-trained arm test predictions | (136, 17393) float32 |
| `first_predictor_DeepDEP/DeepDEP_predictor_final.csv` | first-predictor frozen predictions | 273 × 1244 |
| `first_predictor_DeepDEP/DeepDEP_predictor_strict.csv` | same, strict version | 272 × 1244 |

`index/depmap_index.json` provides a machine-readable index containing
`train_rows_in_cell_order` / `test_rows_in_cell_order` — to take the "training-cells-only gene mean"
just index Y with `train_rows_in_cell_order`.

## ⚠ Known constraint (affects whether a "fair comparison" is possible)

The first predictor (DeepDEP) and the second predictor (VAE-DeepDEP) **do not sit on the same grid**:

| Item | final | strict |
|---|---|---|
| shape | 273 × 1244 | 272 × 1244 |
| genes ∩ GH 17393 | 1244 | 1244 |
| cells ∩ all 908 | 272 | 271 |
| **cells ∩ test 136** | **43** | **43** |
| cells ∩ train 636 | 190 | 189 |

**Conclusion**: all 1244 genes fall inside the 17393 of GH, but only 43/136 cells fall in the test set.
To compare DeepDEP and the two training arms on equal footing, the evaluation is forced down to
**43 test cells × 1244 genes**.

This is a design-level trade-off with two options:
1. **Use only the second predictor for the main analysis** (136 test cells × 17393 genes, complete grid);
2. Keep DeepDEP as a supplementary contrast on the 43×1244 sub-grid, explicitly flagging the grid limitation.

## Original data locations (unmodified)

- Data: `C:\Users\wkr20\Desktop\depmap_second_corpus\work\DeepDEP_training_data\aligned_21Q2_908_3omics`
- Predictions: `C:\Users\wkr20\Desktop\depmap_second_corpus\GH_second_predictor\predictions`
- First predictor: `C:\Users\wkr20\Desktop\depmap_second_corpus\work\fixed_eval_input`

> Note on `MANIFEST.csv`: the `source_in_workspace` column records the authors' local directory layout at
> packaging time. Non-ASCII local directory names (the Chinese-named "figure source data" and
> "supplementary material" directories) were normalized to ASCII (`figure_source_data`, `supplementary`)
> so that every path in this package is plain ASCII; the mapping is 1:1.
