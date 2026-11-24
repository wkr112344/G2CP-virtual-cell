#!/usr/bin/env bash
# Reproduction entry point: recompute the supported analyses in order and check the key numbers.
#
# Usage:
#   ./run_all.sh                         # verify archived outputs only (no raw rebuild needed)
#   ./run_all.sh --data /path/to/raw     # additionally rebuild DepMap raw scores from pkg1-4
#
# Everything this script reads is under the repository, except the optional raw DepMap
# matrices (large, rebuilt from public sources; see DATA_SOURCES.md). Paths are resolved
# relative to this script so a fresh checkout works without edits.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
DATA=""
MANUSCRIPT="$HERE/../GigaScience_submission_20261122/01_main_text/GigaScience_MainText_v1.2_20261123.docx"
[ -f "$MANUSCRIPT" ] || MANUSCRIPT="$(find "$HERE" -iname '*MainText*.docx' | head -1)"
EXPECTED="$HERE/EXPECTED_OUTPUTS.json"
AUDIT="$HERE/github/analysis/stability_checks/_audit_numbers.py"
PY="${PYTHON:-python}"
[ -f "$MANUSCRIPT" ] || MANUSCRIPT=""   # Part B coverage is skipped when no DOCX is present

# Parse optional --data (raw DepMap pkg root). All other args are ignored.
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="${2:-}"; shift 2 ;;
    *) shift ;;
  esac
done

echo "[1/6] Recompute Supp. Tables S5/S6/S9 from per-cell scores + crossed bootstrap"
# audit_and_redraw.py reads source_data/ and writes supplementary/ under github/analysis/.
# It is the canonical generator of S6/S9 (crossed bootstrap, 5-seed x 136-cell); the older
# depmap_background_control.py uses a different bootstrap and is NOT used for these tables.
$PY github/analysis/audit_and_redraw.py

if [ -n "$DATA" ]; then
  echo "[2/6] (optional) Rebuild raw DepMap task tables from pkg1-4 under $DATA"
  $PY github/analysis/depmap_background_control.py --data "$DATA"
else
  echo "[2/6] Skip raw DepMap rebuild (no --data). Archived task tables in source_data/ are used."
fi

echo "[3/6] DepMap positive control (S13-S15)"
XTPM_DIR="${G2CP_ALIGNED_DIR:-$HERE/zenodo/source_data/depmap_raw_artifacts_20260929/reference}"
if [ -f "$XTPM_DIR/X_TPM.npy" ]; then
  $PY github/analysis/positive_control_expression_ridge.py
else
  echo "  SKIP: X_TPM.npy is not in $XTPM_DIR."
  echo "  It is the only input not shipped with the archive (908 x 19,177 expression matrix)."
  echo "  Rebuild it from the public DepMap 21Q2 expression file, then rerun this step:"
  echo "    $PY github/analysis/rebuild_X_TPM.py /path/to/OmicsExpressionProteinCodingGenesTPMLogp1.csv --out <dir>"
  echo "    export G2CP_ALIGNED_DIR=<dir>"
  echo "  The archived S13-S15 tables are still verified against EXPECTED_OUTPUTS.json in step 6."
fi

echo "[4/6] GTEx reference matching (S8, S10-S12)"
GTEX="$HERE/zenodo/source_data/gtex_reference_sensitivity"
if [ -f "$GTEX/data_processed/rnaseqc.npy" ]; then
  $PY github/analysis/06_score_2x2.py
else
  echo "  SKIP 06_score_2x2.py: the per-sample matrices (data_processed/rnaseqc.npy, rsem.npy) are the"
  echo "  5.9 GB x2 GTEx downloads and are not archived. The per-sample scores it produces are archived in"
  echo "  results/per_sample_scores.csv, and step 1 already regenerates S8/S10-S12 from them."
fi
$PY github/analysis/07_bootstrap.py

echo "[5/6] LINCS rank stability / crossed bootstrap / DepMap 272 consistency"
for s in _within_ref_stability _crossed_bootstrap_and_disease _patch_A_B _recompute_depmap272; do
  $PY "github/analysis/stability_checks/${s}.py"
done

echo "[6/6] Check every key number against EXPECTED_OUTPUTS.json"
AUDIT_ARGS="--expected $EXPECTED"
if [ -n "$MANUSCRIPT" ]; then
  AUDIT_ARGS="$AUDIT_ARGS --manuscript $MANUSCRIPT"
else
  echo "  NOTE: no main-text DOCX next to the repository, so Part B (number coverage in the manuscript)"
  echo "  is skipped. Copy GigaScience_MainText_*.docx to ../GigaScience_submission_20261122/01_main_text/"
  echo "  or pass it explicitly to _audit_numbers.py --manuscript to enable that section."
fi
$PY "$AUDIT" $AUDIT_ARGS

echo "All done: '0 mismatched' above means every key number matches the expected-value config."
