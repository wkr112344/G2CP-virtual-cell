#!/usr/bin/env bash
# Reproduction entry point: recompute all analyses in order and check the key numbers.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/6] DepMap shared-gene background control (Table 4, S4-S6)"
python github/analysis/depmap_background_control.py --data zenodo/source_data

echo "[2/6] DepMap positive control (S13-S15)"
python github/analysis/positive_control_expression_ridge.py --data zenodo/source_data/depmap_positive_control

echo "[3/6] GTEx reference matching (S8, S10-S12)"
for s in 06_score_2x2 07_bootstrap; do python github/analysis/${s}.py; done

echo "[4/6] LINCS rank stability / crossed bootstrap (S17-S22)"
python github/analysis/stability_checks/_within_ref_stability.py
python github/analysis/stability_checks/_crossed_bootstrap_and_disease.py
python github/analysis/stability_checks/_patch_A_B.py

echo "[5/6] DepMap 272-cell consistency (S16)"
python github/analysis/stability_checks/_recompute_depmap272.py

echo "[6/6] Check the key numbers"
python github/analysis/stability_checks/_audit_numbers.py --expected EXPECTED_OUTPUTS.json

echo "All done: if the last step reports no FAIL, every key number in the paper has been reproduced."
