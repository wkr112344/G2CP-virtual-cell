FROZEN PREDICTION MATRICES - LINCS controlled reference-product comparison
=============================================================================
Generated: 2026-09-29
Repro script: gigascience_v5/experiments/P_frozen_multimetric_cohorts.py
(faithful to prior archive p4c_run_A_2x2.py)

CONTENT
-------
frozen_pred_matrix_beta_trained_11275x978.npy   beta-trained G2CP predictions
frozen_pred_matrix_dcic_trained_11275x978.npy   dcic-trained G2CP predictions
frozen_perrow_three_metrics.csv                 row -> (cell, drug) map + per-row scores

SHAPE / DTYPE
-------------
(n_rows=11275, n_cols=978) float32
These are the ORIGINAL frozen prediction matrices that the local submission
package previously lacked (Note S14.4).

ROW ORDER
---------
Rows correspond to P* = the 11,275 common (cell line, drug) eval pairs, exactly
the same order as the 'row' column in frozen_perrow_three_metrics.csv
(columns 'cell' and 'drug' give the identity of each row).
P* is defined by the p4c/step55 design: C*=64 common cell lines, D*=20,370 common
drugs, H=2,037 held-out drugs (RandomState(0) last 10%), pairs present in both
reference caches.

COLUMN ORDER
------------
978 LINCS landmark genes, fixed column map internal_col from
_scratch/v31_mvpa/93_gene_map_12328_to_978.json (md5 28caecbb433fea49415aeb2ea1b34fa1)
i.e. prediction = full 12,328-space output sliced as M[:, internal_col].

MODELS (all other factors identical between the two arms)
---------------------------------------------------------
beta_trained : assets/g2cp_v7_beta_ft.pt
dcic_trained : assets/g2cp_v7_dcic_ft.pt
(same base g2cp_full_cpi_v7.pt, same adaptation protocol, only the training
 target differs - this is the controlled 2x2 design)

REFERENCE ARMS USED FOR SCORING
-------------------------------
A = level5beta2020 (MODZ)   <- data/g2cp_cache_beta_v2, sliced to 978
B = dcic2021 (CD)           <- data/g2cp_cache_2021,    sliced to 978

VERIFICATION MARKERS (please check before reuse)
------------------------------------------------
Row-wise Pearson, per-drug mean, drug-clustered bootstrap (2000, seed 12345):
  beta_trained vs level5beta2020 = 0.3680 [0.3627, 0.3733]
  beta_trained vs dcic2021       = 0.0724 [0.0683, 0.0766]
  dcic_trained vs level5beta2020 = 0.1705 [0.1652, 0.1757]
  dcic_trained vs dcic2021       = 0.1534 [0.1489, 0.1582]
If a recomputation does not reproduce these, the matrices are being mis-read.

NOTE: these matrices are the inputs behind the three-metric readings
(Pearson / Spearman / cosine) and the non-overlapping cohort replications;
see deliverables_independent_cohort_experiments_20260929.zip for those tables.
