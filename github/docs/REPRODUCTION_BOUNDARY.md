# Reproduction boundary

This file states exactly what can and cannot be recomputed from the files in this archive.
It is written so a reviewer can tell, before running anything, which numbers are
independently verifiable and which are delivered as summaries.

## A · Exactly recomputable from files in this archive

| Domain | What | Source files |
|---|---|---|
| LINCS | three metrics × three weightings, all drug- and cell-weighted cells | `S1_weighting_and_metrics.csv`, `source_data/frozen_prediction_matrices/frozen_perrow_three_metrics.csv` (11,275 rows × Pearson/Spearman/cosine × two references) |
| LINCS | leave-one-cell-out | `S2_leave_one_cell_out.csv` |
| LINCS | 2×2 interaction, D_beta / D_dcic | `S20` → superseded by `S23_lincs_crossed_bootstrap.csv` (raw and gene-mean-removed) |
| LINCS | global rank Spearman / Kendall | `S21_rank_concordance_summary.csv` |
| LINCS | top-k retention, Jaccard, random expectation | `S3`, `S22_topk_random_expectation.csv` |
| LINCS | within-reference list stability | `S17`, `S17b`, `S17c` (+ per-pair detail, see B) |
| LINCS | disease shortlist boundary structure | `S19_disease_shortlist_boundary.csv` |
| LINCS | per-drug coverage and scores | `S_pearson_by_drug.csv`, `S_spearman_by_drug.csv`, `S_cosine_by_drug.csv`, `TableS_drug_coverage_matrix_drugs.csv` |
| LINCS | per-cell scores | `S_*_by_cell.csv` |
| DepMap | Table 4 raw four cells and the gene-mean baseline | `S4`, `S5_depmap_raw_residualized_2x2.csv` |
| DepMap | **per-cell interaction (previously summary-only)** | `source_data/depmap_background_control/task5_interaction_per_cell_seed.csv` (680 rows = 5 seeds × 136 cells, raw and residualized) |
| DepMap | gene-wise cross-cell distribution | `S14_depmap_positive_control_genewise_across_cell.csv` (173,930 rows) |
| DepMap | positive-control model metrics | `S13` |
| DepMap | 272-cell consistency | `S16_depmap_272cell_consistency_and_prediction_means.csv` |
| GTEx | 2×2 four cells, 189 donor medians, donor bootstrap, 68-tissue sensitivity | `S8`, `S10`, `S11_gtex_donor_bootstrap.csv` (5,000 per-row replicates), `S12` |

## B · Recomputable, but note which aggregation order

**DepMap interaction point estimates.** The per-cell values are in
`task5_interaction_per_cell_seed.csv`, so 0.056317 (raw) and −0.000467 (residualized)
can be reproduced exactly. The aggregation order matters and is not the obvious one:

```python
piv = df.pivot(index='seed', columns='cell', values='raw_interaction').sort_index(axis=1)
estimate = np.median(piv.to_numpy(), axis=1).mean()   # per-cell median, then mean over seeds
```

Flattening to 680 values and taking one median gives 0.05632 for raw but −0.00054 for
residualized — the raw point is median-insensitive, the residualized one is not, so only
the order above reproduces the reported figure. `S6_depmap_interaction_bootstrap_summary.csv`
is the authoritative record; `depmap_background_control/REPORT.md` reports a third quantity
(−0.000100), the mean of the 5,000 bootstrap replicates, which is a different estimand.

**Bootstrap intervals.** `S9` (10,000 rows), `S11` (5,000 rows) and
`task6_interaction_bootstrap_distributions.csv` (5,000 rows × 2) ship every replicate, so
those intervals can be recomputed from the replicates rather than trusted from the CI column.
The within-reference top-k intervals (`S17`, `S17b`, `S17c`) previously shipped only
mean + CI + round count; per-pair detail is now provided in
`source_data/lincs_index/within_reference_topk_stability_replicates.csv`
(B = 200 resamples, C(200,2) = 19,900 pairs per k, seed 20260930).

## C · Not recomputable from this archive — summary values only

| Item | What is delivered | What is missing |
|---|---|---|
| **Figure 2D**, seven published models (CIGER / DeepCE / MultiDCP / PertDiT / PRnet / TranSiGen / XPert; delta 0.011–0.0967) | `source_data/figure_source_data/TableS_seven_models.csv` — 7 rows of summary values with a permutation p and a bootstrap CI | the per-(cell, drug) predictions of the seven models. The upstream screening result retained only the aggregated form, so the seven-model panel cannot be recomputed, only checked for internal consistency. |
| **Figure 4D**, disease and pathway gene-set panels (Liver / Thyroid / APL / MYC / EMT / Hypoxia; Spearman 0.4135–0.6167) | `source_data/figure_source_data/TableS_disease_signatures_summary.csv` — 10 rows of Spearman, top-10 Jaccard and n_drugs | the per-drug scores behind those panels, and the gene lists themselves. MSigDB C2:CGP is licence-restricted and cannot be redistributed. |
| **G2CP frozen predictions** | `frozen_pred_matrix_{beta,dcic}_trained_11275x978.npy` (44 MB each) and the per-row metric table | — *this item is in fact delivered*; see the note in the reproduction map. |
| **Raw LINCS / DepMap matrices** | `inst_id` lists and deterministic rebuild paths | the level-5 and DepMap primary matrices themselves (public but large) — see `DATA_SOURCES.md` for download and subset-build commands. |

For the two rows marked as summary-only, the archive delivers **the results, not the
pipeline that produced them**. They support auxiliary comparisons; the paper's primary
quantities (LINCS delta 0.2956, DepMap interaction, GTEx 0.07096) are all in class A.