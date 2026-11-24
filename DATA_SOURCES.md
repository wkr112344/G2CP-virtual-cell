# External data sources and rebuild paths

| Data | Version / identifier | How to obtain | Local copy dated | Included here | License |
|---|---|---|---|---|---|
| LINCS L1000 level-3/level-5 (beta2020) | GSE92742; MODZ level-5 | GEO / CLUE.io | 2026-09-11 | No (41 GB gctx) | Public; see GEO terms |
| LINCS dcic2021 (Characteristic Direction) | SigCom LINCS / DCIC 2021 | sigcom.lincs.mssm.edu / clue.io | 2026-09-11 | No | Public |
| LINCS evaluation grid, 11,275 rows | `source_data/lincs_index/eval11275_trt_inst_ids.txt` | Deterministic rebuild from the inst_id list | 2026-09-11 | Yes (inst_id lists) | — |
| LINCS 11,275×978 reference matrices | `source_data/lincs_reference_matrices/` | Exported from the public level-5 products (see below) | 2026-09-11 | **Yes** (~88 MB, float32) | Derived; see GEO/MSigDB-side terms of the source products |
| DepMap 21Q2 Public v2 dependency matrices (CERES/Chronos) | 21Q2 Public v2 | depmap.org/portal/download | 2026-09-06 | Derived subsets on Zenodo | CC-BY 4.0 |
| DepMap 21Q2 expression matrix (positive-control input) | `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | depmap.org | 2026-09-06 | **No (must be rebuilt; see below)** | CC-BY 4.0 |
| DeepDEP published predictions | Original paper supplement | Original paper | 2026-09-06 | Yes (`source_data/.../first_predictor_DeepDEP`) | Per original paper |
| GTEx v11 RNASeQC v2.4.3 / RSEM v1.3.3 | 2025-08-22 release | gtexportal.org | 2026-09-24 | Derived data in package; raw 5.9 GB ×2 not included | GTEx terms |
| MSigDB C2:CGP (ACEVEDO_LIVER… / RODRIGUES_THYROID… / CASORELLI_APL…) | v2023.2 | msigdb.org (registration required) | 2026-09-24 | **No (license restriction)** | MSigDB terms; whole-database redistribution prohibited |

## The one input that is rebuilt rather than shipped

**`X_TPM.npy` for the DepMap positive control** (908 × 19,177, float32, ~69 MB) is not redistributed because it is a
direct projection of the public DepMap 21Q2 expression file onto the analysed cell lines. The rebuild is scripted and
deterministic:

```
python github/analysis/rebuild_X_TPM.py /path/to/OmicsExpressionProteinCodingGenesTPMLogp1.csv --out <dir>
export G2CP_ALIGNED_DIR=<dir>
./run_all.sh
```

The rule, which needs no author-side judgement: read the 908 identifiers in
`source_data/depmap_analysis_ready_20260929/index/cell_order.txt`, select those rows from the expression file in that
order, keep every gene column in the file's own order, cast to float32. The script prints the resulting shape and a
sha256, so a rebuilt matrix can be compared directly between reviewers. `Y_CERES.npy`, `Y_Chronos.npy`,
`common_observed_mask.npy`, `index/split.json` and the frozen test predictions all ship with the archive, so the only
missing piece is this projection.

The matrix is certified numerically rather than by a hash recorded here: after the rebuild,
`positive_control_expression_ridge.py` must reproduce the archived S13–S15 values (for example `ridge_CERES` evaluated
on CERES: `cell_raw_median` 0.9345, `cell_zres_median` 0.1432), and the audit step of `run_all.sh` reports any
mismatch against `EXPECTED_OUTPUTS.json`. `run_all.sh` skips this step with an explicit message when `X_TPM.npy` is
absent, so a fresh checkout still completes the archived-output verification end to end.

## Resolved gap (kept here for the record)

**LINCS 11,275×978 reference matrices** were previously read on the fly from the 41 GB gctx during analysis and not
shipped. They are now exported and included under `source_data/lincs_reference_matrices/`
(`Y_beta2020_11275x978.npy`, `Y_dcic2021_11275x978.npy`), together with the export script and the `inst_id` lists,
so a reviewer can either use them directly (skipping the 41 GB download) or rebuild them deterministically from the
public level-5 products. Checksums are given in `source_data/lincs_reference_matrices/README.md` and in the
per-file `.json` sidecars.
