# Conclusion → script → input → expected output

| Paper conclusion | Script | Input | Expected output (checksum value) |
|---|---|---|---|
| LINCS main Δ 0.2956 (95% CI 0.2918–0.2993) | `github/analysis/stability_checks/_audit_numbers.py` or as-supplied | `zenodo/source_data/frozen_prediction_matrices/frozen_perrow_three_metrics.csv` | same value as Table S1 |
| The three metrics agree in direction across matrices | same as above | same as above | Table S1 (Pearson/Spearman/cosine × drug/cell line) |
| Training-state reversal D_beta = +0.1975, D_dcic = −0.0810, I = 0.2785 | `github/analysis/stability_checks/_patch_A_B.py` | `supplementary/frozen_multimetric_perrow_scores.csv` | Table 3, S25 |
| Crossed 2×2 reversal holds 2000/2000 | `github/analysis/stability_checks/_patch_A_B.py` | same | S25 (direction_probability = 1.0) |
| Crossed CI for the main Δ, 0.2857–0.3130 | `github/analysis/stability_checks/_crossed_bootstrap_and_disease.py` | frozen per-row scores | S25 |
| Rank Spearman 0.5675 / Kendall τ 0.4110 | `github/analysis/stability_checks/_audit_numbers.py` + `_within_ref_stability.py` | per-drug ranks | S18 |
| top-10 across matrices 3/10; same-reference 7.93 / 7.04 | `github/analysis/stability_checks/_within_ref_stability.py`, `_patch_A_B.py` | per-row scores (B = 2000 independent pairs) | S17, S17b, S17c |
| top-k vs random expectation | `github/analysis/stability_checks/_audit_numbers.py` | n_drugs = 2,037 | S22 |
| Disease shortlist boundary 0.14%–10.85%, band 11–13 | `github/analysis/stability_checks/_crossed_bootstrap_and_disease.py` | `figure_source_data/TableS_disease_perdrug_scored.csv` | `detail/S19_disease_shortlist_boundary.csv` |
| DepMap interaction 0.05632 → −0.00047 | `github/analysis/audit_and_redraw.py` (canonical generator of S6/S9; `depmap_background_control.py` uses a different bootstrap and is not used for these tables) | `zenodo/source_data/depmap_analysis_ready_*` | Table 5, S6 |
| Gene-mean baseline ≈ 0.90–0.94 | same | same | S4 |
| Positive control 0.2164 / 0.2771 / 0.1233 / 0.1621 | `github/analysis/positive_control_expression_ridge.py` (needs `X_TPM.npy`; rebuild with `github/analysis/rebuild_X_TPM.py`) | `depmap_positive_control` (+ see DATA_SOURCES for the X_TPM rebuild) | S13–S15 |
| DepMap 272-cell consistency 0.9431/0.9467 and prediction means 0.8595/0.8226 | `github/analysis/stability_checks/_recompute_depmap272.py` | `figure_source_data/DepMap_{CERES,Chronos}_272x1244.csv` + `DeepDEP_predictor_strict.csv` | S16 |
| GTEx interaction 0.07096 / 0.06187 (median of medians) | `github/analysis/07_bootstrap.py` (step `07` of the GTEx sequence `00`–`09` in the same directory) | `zenodo/source_data/gtex_reference_sensitivity` (per-sample scores) | S8, S10–S12 |
| Figures 1–7 and single panels | `github/analysis/figure_build/*.py` | `zenodo/source_data/figure_source_data` | same-named files in `zenodo/figures/` |

> Permutation control: `S28_permutation_scheme_comparison.csv` holds all three schemes — A (positional), B (within-cell-line), and C (within-cell-line restricted to strata that admit a complete drug-identity derangement; 6,800 of 6,892 records, 92 single-record strata excluded) — with observed values, null means, 95% intervals, empirical P values and record counts. `build_S28_v2.py` produced this table and audited every permutation: across 2,000 permutations per scheme no record kept its original drug and no fallback construction was needed, recorded in `S28_permutation_constraint_check.csv`. The driver is `github/analysis/supplementary_rebuild_S23_S28/build_S28_v2.py`, with `config_paths.py` and a
> `README.md` that lists the directory layout, the external inputs that are fetched rather than redistributed (the XPert SDST h5ad, the two per-pair LINCS caches, the benchmark prediction folders),
> and the commands for S27–S29. Point `GIGASCIENCE_ROOT` at a directory holding `source_data/` (the archived
> tree), `external/` and `work/`, then run the script; it re-derives the 1,343-drug set, replays the three
> schemes and rewrites `S28_permutation_scheme_comparison.csv`. Replayed on 24 Nov 2026 from the archived
> inputs: all 18 rows match the shipped table exactly (observed Pearson, null mean, both interval ends,
> empirical P, drug count, B; maximum difference 0), and scheme C again keeps 6,800 records over 1,342 drugs
> after excluding the 92 single-record groups. The shipped CSV additionally carries `n_records` and
> `n_excluded_groups` bookkeeping columns that the current script revision does not emit.

> `S31` remains the unrestricted substitution analysis; it was computed with an earlier release of the substitution pipeline, so its observed values differ from scheme A in the fourth decimal place (0.61990 versus 0.61972 for SDST/XPert) and it is labelled as a historical comparison. Table 4 reports results calculated from the original public predictions and is the main result (SDST/XPert 0.61990, beta2020/XPert 0.03249, with their original bootstrap CIs). The permutation analyses used XPert predictions regenerated with an SDPA attention fallback; observed scores and permutation null distributions from that implementation are reported separately per scheme (A/B: 0.61972 / 0.03257; C: 0.61997 / 0.03250). Table 4 uses 1,343 drugs; scheme C excludes singleton strata and evaluates 1,342 drugs over 6,800 model records, so the two are different analysis sets. The replayed point estimates are not paired with the original-prediction CIs.

> Checking convention: all key numbers use tolerance 5×10⁻⁴; interval endpoints 1×10⁻³. The one-by-one comparison
> script is `github/analysis/stability_checks/_audit_numbers.py`, driven by `EXPECTED_OUTPUTS.json`.

> Supplementary renumbering in the submitted package. This snapshot keeps the original file names used during
> analysis; the submission package removes duplicated tables and closes the numbering gaps. The mapping was
> verified on 24 Nov 2026 by matching file contents (numeric columns rounded to six decimals, row count and
> header set) rather than by filename, so each package identifier below points at the file that produced it.
> Identifiers not listed are unchanged between snapshot and package.

| Package S | Snapshot file in `zenodo/supplementary/` |
|---|---|
| S1 | `S1_weighting_and_metrics.csv` |
| S2 | `S2_leave_one_cell_out.csv` |
| S3 | `S3_topk_performance_ranking.csv` (identical content to `S22_topk_random_expectation.csv`; the duplicate is dropped) |
| S4 | `S4_depmap_train_gene_mean_baseline.csv` |
| S5 | `S5_depmap_raw_residualized_2x2.csv` |
| S6 | `S6_depmap_interaction_bootstrap_summary.csv` |
| S7 | `S7_deepdep_common_grid_summary.csv` |
| S8 | `S8_gtex_reference_product_2x2.csv` |
| S9 | `S9_depmap_crossed_bootstrap.csv` |
| S10 | `S10_gtex_donor_medians.csv` |
| S11 | `S11_gtex_donor_bootstrap.csv` |
| S12 | `S12_gtex_tissue_sensitivity.csv` |
| S13 | `S13_depmap_positive_control_model_metrics.csv` |
| S14 | `S14_depmap_positive_control_genewise_across_cell.csv` |
| S15 | `S15_depmap_positive_control_interaction_bootstrap.csv` |
| S16 | `S16_depmap_272cell_consistency_and_prediction_means.csv` |
| S17 | `S17_within_reference_rank_stability.csv` — prediction-accuracy shortlist stability, from 2,000 independent bootstrap pairs |
| S17b | `S17b_coverage_ge3_rank_stability.csv` (identical content to `S18_coverage_ge3_rank_stability.csv`; the duplicate is dropped) |
| S17c | `S17c_within_reference_spearman.csv` — 2,000 independent within-matrix Spearman values (0.8580 beta2020, 0.8246 dcic2021) |
| S18 | `S21_rank_concordance_summary.csv` — cross-matrix rank concordance (Spearman 0.5675, Kendall τ 0.4110) |
| S19 | `S25_disease_within_reference_stability.csv` — disease-reversal rankings, 2,000 replicates with 11,175 non-independent pairs taken from the first 150 |
| S20 | `S24_lincs_perdrug_raw_vs_residualized.csv` |
| S21 | `S23_lincs_crossed_bootstrap.csv` |
| S22 | `S27_disease_signature_gene_counts.csv` |
| S23 | `S26_signature_pair_summary.csv` |
| S24 | `S29_paired_advantage_bootstrap.csv` |
| S25 | `S27_sdst_beta2020_condition_audit.csv` |
| S26 | `S31_unrestricted_substitution.csv` |
| S27 | `S33_matrix_swap_reporting_survey.csv` |
| S28 | `S34_deepdep_top10_overlap.csv` |
| S29 | `S35_depmap_controlled_within_gene.csv` |

> Detail files in the package follow the same mapping: package `detail/S17b_*` = snapshot `S17b_coverage_ge3_rank_stability.csv`
> (same content as `S18_coverage_ge3_rank_stability.csv`); `detail/S17c_*` = snapshot `S17c_within_reference_spearman.csv`;
> `detail/S18_*` = snapshot `S19_disease_shortlist_boundary.csv`;
> `detail/S19_context_panel_*` are generated in-package and have no snapshot predecessor;
> `detail/S21_*` = snapshot `S25_direction_check.csv` and `S25_inside_outside_verdict.csv`;
> `detail/S23_*` and `detail/S23b_*` = snapshot `S26_signature_pair_audit.csv`, `S26b_*.csv`;
> `detail/S26_*` = snapshot `S28_matched_vs_mispaired_pearson_historical_schemeA.csv`,
> `S28_per_row_pearson.csv`, `S28_permutation_scheme_comparison.csv`,
> `S28_permutation_constraint_check.csv`, `S28_xpert_transigen_matched_summary.csv`.
