# Conclusion → script → input → expected output

| Paper conclusion | Script | Input | Expected output (checksum value) |
|---|---|---|---|
| LINCS main Δ 0.2956 (95% CI 0.2918–0.2993) | `github/analysis/stability_checks/_audit_numbers.py` or as-supplied | `zenodo/source_data/frozen_prediction_matrices/frozen_perrow_three_metrics.csv` | same value as Table S1 |
| The three metrics agree in direction across matrices | same as above | same as above | Table S1 (Pearson/Spearman/cosine × drug/cell line) |
| Training-state reversal D_beta = +0.1975, D_dcic = −0.0810, I = 0.2785 | `_patch_A_B.py` | `supplementary/frozen_multimetric_perrow_scores.csv` | Table 3, S20 |
| Crossed 2×2 reversal holds 2000/2000 | `_patch_A_B.py` | same | S20 (direction_probability = 1.0) |
| Crossed CI for the main Δ, 0.2855–0.3128 | `_crossed_bootstrap_and_disease.py` | frozen per-row scores | S18 |
| Rank Spearman 0.5675 / Kendall τ 0.4110 | `_audit_numbers.py` + `_within_ref_stability.py` | per-drug ranks | S21 |
| top-10 across matrices 3/10; same-reference 7.93 / 7.04 | `_within_ref_stability.py`, `_patch_A_B.py` | per-row scores (B = 2000 independent pairs) | S17, S17b, S17c |
| top-k vs random expectation | `_audit_numbers.py` | n_drugs = 2,037 | S22 |
| Disease shortlist boundary 0.14%–10.85%, band 11–13 | `_crossed_bootstrap_and_disease.py` | `figure_source_data/TableS_disease_perdrug_scored.csv` | S19 |
| DepMap interaction 0.05632 → −0.00047 | `as_supplied/depmap_background_control.py` | `source_data/depmap_analysis_ready_*` | Table 4, S6 |
| Gene-mean baseline ≈ 0.90–0.94 | same | same | S4 |
| Positive control 0.2164 / 0.2771 / 0.1233 / 0.1621 | `positive_control_expression_ridge.py` | `depmap_positive_control` (+ see DATA_SOURCES for the X_TPM rebuild) | S13–S15 |
| DepMap 272-cell consistency 0.9431/0.9467 and prediction means 0.8595/0.8226 | `_recompute_depmap272.py` | `figure_source_data/DepMap_{CERES,Chronos}_272x1244.csv` + `DeepDEP_predictor_strict.csv` | S16 |
| GTEx interaction 0.07096 / 0.06187 (median of medians) | `gtex_reference_sensitivity/scripts/07_bootstrap.py` | `gtex_reference_sensitivity` (per-sample scores) | S8, S10–S12 |
| Figures 1–7 and single panels | `github/analysis/figure_build/*.py` | `zenodo/source_data/figure_source_data` | same-named files in `zenodo/figures/` |

> Checking convention: all key numbers use tolerance 5×10⁻⁴; interval endpoints 1×10⁻³. The one-by-one comparison
> script is `github/analysis/stability_checks/_audit_numbers.py`, driven by `EXPECTED_OUTPUTS.json`.
