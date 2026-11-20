# Supplementary material index (GigaScience submission clean-up, v5)

Main text: **"How differently generated evaluation matrices shape biomedical model performance, model comparison,
drug ranking and candidate selection: a cross-ecosystem analysis of LINCS, DepMap and GTEx"**

This submission package keeps only the files that directly correspond to the current main-text claims. Earlier
exploratory material such as LayerDiag, CPI, cancer/non-cancer stratification, the genetics side, and old
baseline/threshold sweeps has been removed from the local submission package, to avoid a reviewer mistaking
historical analyses for part of the current paper's evidence chain. The frozen Zenodo version of record still
preserves the complete, checkable artifacts corresponding to the main-text numbers.

## Core files

| Number | File | Current use |
|---|---|---|
| S13 | `NoteS13_method_and_prior_details.md` | fixed-output / 2×2 statistical calibers, training-exposure boundary, and dcic2021 scoring-object provenance; S13.7 is the hop-by-hop checksum chain |
| S14 | `NoteS14_robustness_and_readings.md` | LINCS alignment/transformation checks, ranking/shortlist robustness, interpretation boundaries, and the matched-2×2 note |
| S19 | `TableS19_drug_identity_audit.*` | drug identity and ECFP4 duplication audit |
| S20 | `TableS20_absolute_topk_screening_rank.csv` | underlying table of cross-reference performance / disease-signature ranking for 1,399 drugs |
| S21 | `TableS21_L1000_provenance_audit.md` | targeted reporting/provenance audit of 12 L1000 studies |
| S21a | `TableS21a_representative_object_tracking_cases.md` | representative score-to-object tracking cases |
| S22 | `TableS22_strict_two_sided_absent_per_drug.csv` | strict-unseen / fingerprint-clean per-drug results |
| S23 | `TableS23_DepMap_fixed_output_replication.*` + `FigS5_DepMap_fixed_output_replication.*` | DepMap CERES/Chronos fixed-output cross-ecosystem stress test |
| S24 | `TableS24_seven_predictor_fixed_output.csv` | per-model two-arm PCC, ΔPCC and bootstrap 95% CI for the fixed-output re-evaluation of seven published predictors (arm A = each predictor's own SDST reference, arm B = the LINCS dcic2021/CD product; both on the 978 LINCS landmark genes) |
| S25 | `NoteS25_training_evaluation_2x2.md` + `TableS25_*` | main-text 11,275-pair matched training-product × evaluation-product 2×2 and the E_common = 15,990 sensitivity |
| S26 | `NoteS26_gene_caliber_and_seven_model_arms.md` | 978 vs 963 gene-caliber explanation (two alignment layers of the same gene axis, not a dimensionality contradiction) + definition of the two arms of the seven-model re-evaluation (including the Figure 3 caption fix) |

The fixed-output re-evaluation of the seven published predictors is shown in main-text Figure 3 and Supplementary
Table S24; per-model two-arm PCC, ΔPCC and bootstrap 95% CI are in `TableS24_seven_predictor_fixed_output.csv`, and
the arm definitions and caliber notes are in `NoteS26_gene_caliber_and_seven_model_arms.md`. The corresponding
prediction matrices, intermediate results, and checkable artifacts are in the frozen Zenodo version of record; the
local submission package does not duplicate the large matrices.

## Interpretation boundaries

1. This study estimates **processed reference product replacement sensitivity**; it does not attribute the LINCS
   difference uniquely to any one preprocessing step.
2. The matched 2×2 shows that the reference contrast is closely tied to training–evaluation target matching; the full
   symmetric 2×2 currently covers only the main predictor.
3. DepMap is a cross-data-ecosystem reproduction of the phenomenon, not a direct replication of the LINCS
   `+0.2956` effect size.
4. strict-unseen / fingerprint-clean covers only the currently traceable fine-tuning training exposure; the full
   training composition of the earlier base checkpoint was not preserved.

## Data and code

- Frozen version of record: Zenodo DOI `10.5281/zenodo.22720615`
- Development code: `https://github.com/wkr112344/G2CP-virtual-cell`
- SHA256, MD5, and build chain of the dcic2021 processed reference: S13.7
- DepMap fixed-output results: S23
- LINCS matched 2×2: S25
