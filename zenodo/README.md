# Reproduction materials — how to use this archive

This archive is the data/artefact half of the reproduction package for the paper

> **How differently generated evaluation matrices shape biomedical model performance, model comparison,
> drug ranking and candidate selection: a cross-ecosystem analysis of LINCS, DepMap and GTEx**

The package is split over three Zenodo records that belong to the same study (each record's *Related works*
panel links to the others). The **code**, the reproduction map, the expected values and the run script are in the
GitHub repository:

<https://github.com/wkr112344/VirtualCellWorkbench/tree/gigascience-refproduct-repro>

## Layout of this archive

```
source_data/    derived inputs and outputs
                frozen_prediction_matrices/   frozen per-row predictions and per-row metric table
                lincs_index/                  LINCS inst_id lists + 12,328->978 gene map
                lincs_reference_matrices/     the two 11,275x978 reference matrices (+ export script)
                depmap_analysis_ready_*/      DepMap grids, splits, 5-seed predictions, train gene means
                depmap_raw_artifacts_*/       DepMap reference matrices and first-predictor outputs
                depmap_positive_control/      expression-PCA -> Ridge positive control
                gtex_reference_sensitivity/   GTEx per-sample scores, metadata, bootstrap results
                figure_source_data/           per-panel source tables behind Figures 1-6
figures/        final main figures (PDF + PNG) and Supplementary Figure S1
supplementary/  Supplementary Tables S1-S22 (CSV) and the supplementary document
manuscript/     manuscript copy
checksums/      MD5 and SHA256 manifests
```

## Verifying the download

```bash
cd <extracted archive root>
md5sum -c checksums/MD5SUMS_zenodo.txt
shasum -a 256 -c checksums/SHA256SUMS_zenodo.txt
```

Every line must report `OK`. Paths in these two manifests are relative to this archive's root.
`checksums/MD5SUMS_github.txt` and `checksums/SHA256SUMS_github.txt` cover the repository files
(package-root documents and `github/`); run those from the root of a fresh clone.

## Where to start

1. `source_data/frozen_prediction_matrices/` holds the per-row metric table that backs the paper's headline
   numbers (Tables S1-S3, S17-S22) - it can be re-analysed without any large download.
2. `source_data/lincs_reference_matrices/` closes the LINCS gap: the two reference matrices are shipped, so the
   41 GB level-5 products are not needed unless you want to rebuild them from scratch.
3. `source_data/gtex_reference_sensitivity/` contains the per-sample scores behind Figure 6 / Tables S8-S12.

## Not redistributed here (public, but large or licence-restricted)

LINCS L1000 raw and level-5 products, the DepMap 21Q2 raw matrices, GTEx v11 raw expression, and the three MSigDB
C2:CGP disease signatures. Download and rebuild paths are documented in `DATA_SOURCES.md` in the repository.
One input still has to be rebuilt by the user: the DepMap positive-control matrix `X_TPM.npy` (908 x 19,177),
following `cell_order.txt` and the instructions in `DATA_SOURCES.md`.
