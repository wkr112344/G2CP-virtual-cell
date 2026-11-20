# GTEx v11 reference-sensitivity: pre-registered analysis framework

Data (all md5-verified):

- `paths.gene_tpm_gz` — RNASeQC gene TPM, 74,628 × 19,788 (GCT)
- `paths.transcripts_tpm_gz` — RSEM transcript TPM, 385,659 × 19,788 (with gene_id)
- Both files are from the same release (2025-08-22); the sample column order is identical file-to-file.

## Run order

```bash
PY=C:/Users/wkr20/miniconda3/envs/dpb311/python.exe
$PY scripts/00_inspect_headers.py          # header/dimension check (seconds; already run)
$PY scripts/01_check_samples.py            # annotation join + donor parsing (seconds; already run)
$PY scripts/02_aggregate_rsem_to_gene.py   # transcripts -> genes, about 5 GB RAM, ~10-20 min
$PY scripts/03_align_products.py           # common gene space -> memmap (19788 × ~n_genes)
$PY scripts/04_make_splits.py              # donor 80/20 (seed 20260929)
$PY scripts/05_build_predictors.py         # per-product tissue-mean predictor
$PY scripts/06_score_2x2.py                # per-sample 2x2 scores
$PY scripts/07_bootstrap.py                # donor bootstrap 5000
$PY scripts/08_make_figures.py             # two figures
```

## Pre-registered statistical definitions (locked before running; calibers unchanged after results)

- **Space**: log2(TPM+1), all common genes (`primary_gene_set=all`; the protein-coding subset is kept as a
  sensitivity analysis and requires a gene_biotype_file in the config).
- **Predictor**: tissue-mean (`SMTSD` fine-grained tissue, train-only means, with a global-mean fallback for tissues
  missing from training). One predictor each for RNASeQC and RSEM (transcripts summed by gene_id) — this is the only
  factor allowed to change; the tissue definition and the train/test split are identical across the two arms.
- **Score**: per test sample, across-gene Pearson (primary) / Spearman (secondary), between the prediction vector and
  the sample vector.
- **2×2**: predictor ∈ {RNASeQC-trained, RSEM-trained} × evaluation reference ∈ {RNASeQC, RSEM}.
- **Main criterion**: interaction = (rRR − rRE) − (rER − rEE).
  Significance = whether the 95% CI from 5000 donor bootstraps excludes 0 (and the direction is consistent across
  splits/metrics). Bootstrap unit = test donor (resample donors, not samples).

## Pre-committed interpretations of the two outcomes (not changed after results)

- **Interaction clear (CI excludes zero)**:
  "GTEx provides an independent transcriptomic ecosystem in which benchmark conclusions
  remain sensitive to the processed reference product."
- **Interaction ≈ 0**:
  "GTEx serves as a negative-control ecosystem, showing that reference sensitivity is
  context-dependent rather than inevitable."

## Known boundaries (honest list)

- RSEM transcript-sum ≠ RNASeQC gene quantification itself (the quantifier difference is exactly the object of study,
  but it means the difference between the two arms includes both the quantifier and the aggregation layer; for now it
  is interpreted jointly as pre-registered).
- Neither GCT nor RSEM carries biotype, so protein-coding filtering needs an external gencode attributes file.
- The tissue-mean predictor is the simplest predictor, so the conclusion is limited to "reference sensitivity under a
  weak predictor".
- The comparison with the LINCS/DepMap branches is conceptual (both are reference-product sensitivity), not a
  numerical pooling.
