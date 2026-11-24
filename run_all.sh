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
$PY github/analysis/positive_control_expression_ridge.py

echo "[4/6] GTEx reference matching (S8, S10-S12)"
for s in 06_score_2x2 07_bootstrap; do $PY github/analysis/${s}.py; done

echo "[5/6] LINCS rank stability / crossed bootstrap / DepMap 272 consistency"
for s in _within_ref_stability _crossed_bootstrap_and_disease _patch_A_B _recompute_depmap272; do
  $PY "github/analysis/stability_checks/${s}.py"
done

echo "[6/6] Check every key number against EXPECTED_OUTPUTS.json"
AUDIT_ARGS="--expected $EXPECTED"
[ -n "$MANUSCRIPT" ] && AUDIT_ARGS="$AUDIT_ARGS --manuscript $MANUSCRIPT"
$PY "$AUDIT" $AUDIT_ARGS

echo "All done: '0 mismatched' above means every key number matches the expected-value config."
