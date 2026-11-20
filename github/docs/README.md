# DepMap positive control: expression-PCA → multi-output Ridge

**Question.** Once the gene-level background has been residualized away, is there any model that still
retains a detectable predictive signal? If only models with real cell-state information survive, then
"the frozen model goes to zero after residualization" cannot be explained as "residualization mechanically
kills all signal".

All computations use the same **Figure 6 split** (`split.json`, seed 42: train 636 / val 136 / test 136 cells),
with DepMap 21Q2 data: `X_TPM.npy` (908×19177), `Y_CERES/Y_Chronos.npy` (908×17393), `common_observed_mask`.

## Objects (4 × 2 evaluation references)

| Object | Definition |
|---|---|
| gene-mean baseline | prediction = per-gene mean over the training set (the same vector for every cell) |
| **expression-PCA→Ridge (positive control)** | X → training-set standardization → PCA(100) → multi-output Ridge, with training targets CERES / Chronos respectively |
| permuted-label Ridge (negative control) | same pipeline, but the row labels of the training Y are shuffled |
| frozen G2CP (current model) | existing 5-seed averaged predictions (`depmap_analysis_ready_20260929`) |

`alpha` is chosen by **5-fold CV on the training set only** over {0.1, 1, 10, 100, 1000} (the test set never
participates); all three arms select 1000.

## Three calibers (key definitions)

- `raw`: per-cell Pearson(pred, truth) across genes.
- `shift`: pred and truth are **both reduced by the per-gene training-set mean** (original design). Note this
  is a **per-gene** constant (c_g), not a global constant, so
  `cov(x−c, y−c) = cov(x,y) − cov(x,c) − cov(y,c) + var(c) ≠ cov(x,y)` —
  it genuinely changes the across-gene correlation, removing exactly the essentiality background shared across genes.
- `z`: `shift` plus division by the per-gene training-set standard deviation (an extra scale normalization).
- `across-cell`: per-gene Pearson across the test cells (invariant to per-gene shift/scale).

## Results (median, 5000× cell bootstrap CI)

| Object | raw | shift (residualized) | z (residualized + scale) | across-cell |
|---|---|---|---|---|
| gene-mean baseline | +0.9338 | **0.0000** | 0.0000 | n/a (constant predictor) |
| permuted Ridge | +0.9272 | **−0.0001** [−0.0082, +0.0074] | −0.0015 | +0.0015 |
| **expression Ridge (CERES)** | +0.9345 | **+0.2164** [+0.1951, +0.2321] ✓ | **+0.1432** ✓ | **+0.1233** ✓ |
| **expression Ridge (Chronos)** | +0.9434 | **+0.2771** [+0.2568, +0.3063] ✓ | **+0.1833** ✓ | **+0.1621** ✓ |
| frozen G2CP (CERES) | +0.9336 | −0.0012 [−0.0096, +0.0069] ✗ | +0.0013 | +0.0003 |
| frozen G2CP (Chronos) | +0.9389 | −0.0027 ✗ | +0.0040 | +0.0055 |

## 2×2 (training target × evaluation reference), interaction = (CC−CH)−(HC−HH)

| Caliber | expression Ridge | frozen G2CP | permuted |
|---|---|---|---|
| raw | +0.0620 [+0.0587, +0.0654] ✓ | +0.0573 [+0.0549, +0.0597] ✓ | 0 |
| **shift (residualized)** | **+0.3057** [+0.2731, +0.3408] ✓ | **−0.0003** [−0.0172, +0.0167] ✗ | 0 |
| z (residualized + scale) | **+0.2101** [+0.1885, +0.2347] ✓ | +0.0094 ✗ | 0 |

## Conclusions (closing the loop for reviewers)

- **Residualization does not mechanically kill all signal**: a genuine predictor still scores +0.216 / +0.277
  after residualization, with the CI excluding zero.
- **The pipeline itself does not manufacture signal**: permuted Ridge ≈ 0 and the gene-mean baseline is exactly 0.
- **The frozen model's product-matching interaction is pure gene-level background**: raw +0.0573 → −0.0003 after
  residualization (zeroed), whereas the expression Ridge interaction is actually **amplified** to +0.3057 by
  residualization.
  → The two components of reference-product sensitivity are thus fully separated: the frozen model's share comes
  from the shared gene-mean structure, the Ridge's share from genuine cell-state predictive power.

## Files

```
scripts/positive_control_expression_ridge.py   main experiment (all metrics + bootstrap)
scripts/bootstrap_interaction.py              5000× cell bootstrap for the interaction
scripts/make_panelAB_figure.py                Panel A/B figure
scripts/make_panelC_figure.py                 Panel C (2×2) figure
results/model_metric_summary.csv              4 objects × 2 references × all metrics
results/per_cell_metrics.csv                  per-cell raw/shift/zres
results/per_gene_across_cell.csv              per-gene across-cell (17,393 genes)
results/positive_control_summary.json         summary + bootstrap
results/interaction_bootstrap.json            interaction CI (three calibers × three objects)
figures/panel_AB_positive_control.png         Panel A/B
figures/fig_panelC_2x2_heatmap.png            Panel C
```

## Known boundaries

- The permuted Ridge has a median across-cell of +0.0015 with CI [+0.0001, +0.0034], which technically excludes
  zero — but it is two orders of magnitude smaller than the Ridge's +0.1233, i.e. numerical-noise level; we report
  it as it stands.
- The across-cell metric requires that a gene has ≥30 observed test cells; the gene-mean baseline has no
  across-cell value (constant predictor).
- Ridge is a linear predictor, so the conclusion is limited to "linearly readable cell-state information".
