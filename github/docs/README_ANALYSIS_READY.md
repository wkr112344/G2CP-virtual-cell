# DepMap analysis-ready artifacts (transfer-friendly)

Generated 2026-09-29 from `depmap_raw_artifacts_20260929/` (source untouched).

## WHY THIS PACKAGE IS SMALLER
The full package is 207 MB and does not survive IM transfer (float32 deflates
to ~93% of raw, so compression cannot help). This package keeps everything the
pre-registered analysis needs and drops what it does not:

  KEPT   reference matrices at the 136 TEST cells      (scoring endpoints)
  KEPT   train-only gene mean, pre-computed            (gene-level background)
  KEPT   all 10 prediction matrices (5 seeds x 2 arms) (cross-seed criterion)
  DROPPED the 908-row full reference matrices
         -> Y_CERES / Y_Chronos / mask at val cells and at train cells
            (train rows are NOT needed per-row: only their gene mean is)

If you need the full 908-row Y matrices, they are on the local disk at
`Claw\depmap_raw_artifacts_20260929\reference\` and can be shipped separately.

## LAYOUT
index/                    cell_order(908) target_genes(17393) split.json(seed42)
                          test_cell_order(136) depmap_index.json manifest.json
reference_test/           Y_CERES_test136.npy  Y_Chronos_test136.npy
                          common_observed_mask_test136.npy        (136, 17393)
train_gene_mean/          train_genemean_Y_CERES.npy  train_genemean_Y_Chronos.npy
                          train_observed_count.npy                (17393,)
predictions_test/         seed{0-4}_pred_{ceres,chronos}_arm_test.npy x10 (136, 17393)
first_predictor_DeepDEP/  _intersection_stats.json mapping_audit.json
verify_alignment.py       self-check (run it first)

## ALIGNMENT RULES (do not skip)
1. 908-row matrices: row order = index/cell_order.txt; column order =
   index/target_genes.txt, entries are `SYMBOL (EntrezID)` -> strip the ` (id)`
   suffix before matching against external gene symbols.
2. *_test*.npy row order = index/test_cell_order.txt, verified identical to
   split.json test_cells (index/depmap_index.json:
   `test_pred_row_order_equals_split_test_cells = true`).
3. Split is locked at seed 42: train 636 / val 136 / test 136. The gene mean
   here uses ONLY the 636 train rows (no val, no test).
4. NaN handling: use common_observed_mask; do NOT impute. The train gene mean
   is NaN-aware (nanmean over observed entries only) and train_observed_count.npy
   gives, per gene, how many train cells were observed.

## WHAT THE GENE MEAN IS FOR
Pre-registered contrast: residualize BOTH prediction and reference by the
train-only gene mean, then recompute
  (a) fixed-output score,
  (b) per-gene across-cell score,
  (c) training x evaluation 2x2 interaction.
Primary criterion is written down before looking at results: does the
residualized interaction keep its sign, stay stable across the 5 seeds, and
have a bootstrap CI excluding 0.

## GRID CONSTRAINT (affects the design)
First predictor DeepDEP sits on a different grid: 273 x 1244 (strict 272 x 1244).
All 1244 genes are inside the 17393, but only 43 of 136 test cells overlap.
Including DeepDEP shrinks the evaluation to 43 test cells x 1244 genes.
Recommended: main analysis on the full 136 x 17393 grid (second predictor only);
DeepDEP as a supplementary contrast with the grid limitation stated.
