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

## One item the authors still need to add (needed for bit-for-bit reproduction)

**`X_TPM.npy` for the DepMap positive control** (908×19,177, ~69 MB): this file is no longer in the workspace.
To rebuild it: from DepMap 21Q2 `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, select the 908 cell lines given in
`cell_order.txt`, keep the 19,177 protein-coding genes, write as float32 `X_TPM.npy`, and place it on Zenodo
(with a rebuild script + checksum). Otherwise the positive control can only be "rerun to the same result" and the
input matrix itself cannot be checked.

## Resolved gap (kept here for the record)

**LINCS 11,275×978 reference matrices** were previously read on the fly from the 41 GB gctx during analysis and not
shipped. They are now exported and included under `source_data/lincs_reference_matrices/`
(`Y_beta2020_11275x978.npy`, `Y_dcic2021_11275x978.npy`), together with the export script and the `inst_id` lists,
so a reviewer can either use them directly (skipping the 41 GB download) or rebuild them deterministically from the
public level-5 products. Checksums are given in `source_data/lincs_reference_matrices/README.md` and in the
per-file `.json` sidecars.
