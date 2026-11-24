# Reproduction boundary

This file states exactly what can and cannot be recomputed from the files in this archive.
It is written so a reviewer can tell, before running anything, which numbers are
independently verifiable and which are delivered as summaries. Every path below is relative
to `zenodo/` unless it starts with `analysis/` (relative to `github/`) or names a top-level
file such as `DATA_SOURCES.md`.

## A · Exactly recomputable from files in this archive

| Domain | What | Source files |
|---|---|---|
| LINCS | three metrics × three weightings, all drug- and cell-weighted cells | `S1_weighting_and_metrics.csv`, `source_data/frozen_prediction_matrices/frozen_perrow_three_metrics.csv` (11,275 rows × Pearson/Spearman/cosine × two references) |
| LINCS | leave-one-cell-out | `S2_leave_one_cell_out.csv` |
| LINCS | 2×2 interaction, D_beta / D_dcic | `S23_lincs_crossed_bootstrap.csv` is the authoritative record (20 rows, raw and gene-mean-removed). Regenerate it with `analysis/stability_checks/S21_lincs_crossed_bootstrap.py` (seed 20260930, 2,000 rounds); all 20 rows return to float64 round-off. `S20_lincs_crossed_2x2_model_comparison.csv` is a scratch variant from a different RNG consumption order: it returns 0.2657–0.2887 for the raw interaction, **not** the published 0.2646–0.2889 |
| LINCS | global rank Spearman / Kendall | `S21_rank_concordance_summary.csv` |
| LINCS | top-k retention, Jaccard, random expectation | `S3`, `S22_topk_random_expectation.csv` |
| LINCS | within-reference list stability | `S17`, `S17b`, `S17c` (+ per-pair detail, see B; S17/S17c use 2,000 independent pairs, S17b uses 19,900 pairs from 200 rankings) |
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
`S21` likewise ships every replicate indirectly: `S23_lincs_crossed_bootstrap.csv` carries
`boot_mean`, `sd` and both percentile bounds for all 20 rows, and the per-replicate draws are
regenerated bit-for-bit by `analysis/stability_checks/S21_lincs_crossed_bootstrap.py`.

The within-reference top-k intervals (`S17`, `S17b`, `S17c`) were computed under **two
different pairing schemes**, and the shipped per-pair detail covers only the second:

- `S17` (full 2,037-drug set) and `S17c` (within-matrix Spearman): 2,000 independent bootstrap
  pairs, i.e. two fresh row resamples per round, from `analysis/stability_checks/_patch_A_B.py`
  section A. Overlap means can differ from the shipped values by ≤0.25 counts because the
  archived driver fixes seed 20260930 and the original run's seed was not recorded; every
  percentile bound is identical.
- `S17b` (1,303 drugs with ≥3 evaluation records): 200 bootstrap rankings per matrix scored
  over all C(200,2) = 19,900 unordered pairs, from `_within_ref_stability.py` (`B = 200`).
  Every value is reproduced exactly from
  `source_data/lincs_index/within_reference_topk_stability_replicates.csv`
  (`scenario = V2_ge3_cells`, `reference = beta`; 398,000 rows = 19,900 pairs × 5 k × 2
  references × 2 scenarios).

## C · Not recomputable from this archive — summary values only

| Item | What is delivered | What is missing |
|---|---|---|
| **Figure 2D**, seven published models (CIGER / DeepCE / MultiDCP / PertDiT / PRnet / TranSiGen / XPert; delta 0.011–0.0967) | `source_data/figure_source_data/TableS_seven_models.csv` — 7 rows of summary values with a permutation p and a bootstrap CI | the per-(cell, drug) predictions of the seven models. The upstream screening result retained only the aggregated form, so the seven-model panel cannot be recomputed, only checked for internal consistency. |
| **Figure 4D**, disease and pathway gene-set panels (Liver / Thyroid / APL / MYC / EMT / Hypoxia; Spearman 0.4135–0.6167) | the three disease contexts are **fully recomputable**: `source_data/figure_source_data/TableS_disease_perdrug_scored.csv` (4,197 rows = 3 signatures × 1,399 drugs, with `score_A_beta_reversal`, `score_B_dcic_reversal`, `rank_A`, `rank_B`) plus `S19_disease_shortlist_boundary.csv` and `TableS_disease_top10_shortlist.csv`. The three pathway contexts are summary-only: `TableS_disease_signatures_summary.csv` (10 rows of Spearman, top-10 Jaccard, n_drugs) | the per-drug pathway-suppression scores for MYC / EMT / Hypoxia, and the MSigDB Hallmark and C2:CGP gene lists themselves (licence-restricted, not redistributable). |
| **G2CP frozen predictions** | `frozen_pred_matrix_{beta,dcic}_trained_11275x978.npy` (44 MB each) and the per-row metric table | — *this item is in fact delivered*; see the note in the reproduction map. |
| **Raw LINCS / DepMap matrices** | `inst_id` lists and deterministic rebuild paths | the level-5 and DepMap primary matrices themselves (public but large) — see `DATA_SOURCES.md` for download and subset-build commands. |

For the rows marked as summary-only, the archive delivers **the results, not the pipeline that
produced them**. They support auxiliary comparisons; the paper's primary quantities (LINCS
delta 0.2956, DepMap interaction, GTEx 0.07096) are all in class A.