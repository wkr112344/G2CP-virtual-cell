# G2CP — A 162-Cell-Line Genome-Wide Virtual Cell Platform

Independent reimplementation and large-scale extension of **UniPert-G2CP** (Li et al., *Cell*, 2026), a unified framework for predicting transcriptomic responses to genetic and chemical perturbations.

[![Preprint](https://img.shields.io/badge/openRxiv-preprint-blue)](https://doi.org/10.64898/2026.08.10.743942)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-green.svg)](https://www.python.org/)

## Overview

This repository provides:

- **162 cell lines × 32,039 compounds** — 4× compounds and 32× cell lines vs. the original paper (5 lines, 7,860 compounds)
- **12,328 genome-wide gene output** vs. original landmark genes
- **Deployed interactive API** for real-time perturbation prediction
- **Full evaluation suite** with held-out, self-consistency, and benchmark-consistent metrics
- **Qualitative case demonstrations** on literature-supported pairs: dexamethasone→TSC22D3/NFKBIA/FKBP5 (recapitulated), bortezomib→BAG3/DNAJB1/HSPA1A (recapitulated), CD36→PPARG/CEBPA (absent), metformin→SLC7A5 (partially recapitulated)

### Key Metrics

| Quantity | Value |
|---|---|
| Cell lines | 162 |
| Compounds | 32,039 |
| Genome-wide gene output | 12,328 |

Performance metrics (genetic/novel-drug PCC, directional accuracy, CPI enrichment, mechanism-clustering SMD) are reported in the [manuscript](./biorxiv_submission.html) together with their aggregation conventions, release-controlled evaluation sets, and confidence intervals — they are convention- and release-dependent and should not be read as single face-value numbers.

## Quick Start

### Requirements

- Python 3.12+, PyTorch 2.x with CUDA
- NVIDIA GPU ≥ 4 GB VRAM (RTX 3050 Ti tested)
- ESM-2 (8M) for protein embedding anchoring

### Installation

```bash
pip install torch numpy scipy scikit-learn rdkit-python fair-esm flask anndata
```

### Running the Prediction API

```bash
python serve_g2cp.py --port 8766
```

API endpoints:

- `POST /predict` — Predict drug effect (by name or SMILES)
- `POST /gene` — Predict gene knockout effect
- `GET /health` — Service status
- `GET /cells` — List supported cell lines
- `GET /genes` — List supported genes
- `GET /drugs` — List supported drugs

### Example

```python
import requests

# Predict metformin effect on HEPG2
r = requests.post("http://127.0.0.1:8766/predict", json={
    "drug": "metformin",
    "cell_name": "HEPG2"
})
data = r.json()
print("Top upregulated genes:", [(g['g'], g['v']) for g in data['up'][:5]])
print("Top downregulated genes:", [(g['g'], g['v']) for g in data['down'][:5]])

# Predict CD36 knockout in ASC cells
r = requests.post("http://127.0.0.1:8766/gene", json={
    "gene": "CD36",
    "cell_name": "ASC"
})
data = r.json()
```

## Repository Structure

```
.
├── train_g2cp_full.py         # Full training pipeline
├── train_g2cp_contrast.py     # Network definition & contrastive loss
├── serve_g2cp.py              # Flask prediction API
├── unipret/
│   └── compound_encoder.py    # SMILES → ECFP4 fingerprint encoding
├── eval_full.py               # Held-out perturbation evaluation
├── eval_smd.py                # Mechanism-clustering SMD evaluation
├── eval_holdout_similarity.py # Novel drug hold-out evaluation
├── _exam2_align.py            # Alignment ρ evaluation
├── _exam3_ef.py               # CPI enrichment factor evaluation
├── _eval_dir_acc.py           # Directional accuracy evaluation
├── _reproduce_paper_smd.py    # Paper-consistent SMD replication
├── _reproduce_paper_rho.py    # Paper-consistent ρ replication
├── index.html                 # Interactive web frontend
├── chain.html                 # Drug→target→pathway explanation tool
├── biorxiv_submission.html    # Preprint (print-ready)
└── data/
    ├── dataset.json           # 218 drugs, 20 targets, 262 CPI pairs
    └── lincs_meta/            # LINCS metadata & evaluation files
```

## Data Availability

Training data sources (public):
- LINCS L1000: GSE92742 (level 5, moderated Z-scores)
- DepMap CRISPR: 23Q2 release, CERES scores
- ChEMBL 37 (CPI annotations)
- GEO: GSE61302, GSE11917, GSE17579, GSE20986, GSE21413, GSE25941, GSE33622, GSE50378, GSE50397, GSE53751, GSE62914, GSE22886, GSE60235 (13 series)

**Model weights**, cache-reconstruction scripts, and all result files are archived on Zenodo: [https://doi.org/10.5281/zenodo.22720615](https://doi.org/10.5281/zenodo.22720615) (CC-BY-4.0). A distinction that matters for reuse: the full upstream LINCS L1000 level-5 matrices (~7.5 GB) are **not distributed** and must be regenerated from public sources (GSE92742, the LINCS level-5 beta release, and the DCIC 2021 distribution) with the included cache-reconstruction scripts. The matrices actually used in this study are distributed: `zenodo/source_data/lincs_reference_matrices/` and `frozen_prediction_matrices/` contain the 11,275 × 978 aligned reference and frozen prediction matrices (`.npy`), so every result in the paper can be rescored without rebuilding the upstream matrices. See `REPRODUCTION_BOUNDARY.md` for an item-by-item statement of what is exactly recomputable versus summary-only.

The original study's open resources: [GitHub](https://github.com/lynn-1998/UniPert-G2CP_reproduce) | [Zenodo](https://zenodo.org/records/20355906)

## Citation

```bibtex
@article{wei2026g2cp,
  title={Cancer cell line heterogeneity imposes a primary bottleneck for virtual perturbation screening at scale},
  author={Zhan, Lijing and Qi, Canyang and Wei, Kairui},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.08.10.743942}
}
```

## License

Code in this repository is licensed under the MIT License; model weights, data, and figures are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Author

**Kairui Wei** — School of Clinical Medicine, Xinjiang Medical University, Urumqi 830011, China

---

## Reproducibility package (GigaScience manuscript)

The complete reproducibility bundle for the evaluation-matrix study ships inside this
repository, alongside the G2CP source tree:

- `REPRODUCTION_MAP.md` — each paper conclusion mapped to the script, input, and expected output
- `MANIFEST.csv` — 369 entries listing every packaged file, its origin, and its role
- `github/analysis/` — analysis and figure-build scripts, including `stability_checks/_audit_numbers.py`
- `zenodo/source_data/` — frozen prediction matrices (11,275 × 978), reference matrices, and index files
- `zenodo/supplementary/` — the 29 numbered supplementary tables S1–S29 plus detail files used by the submitted package (this snapshot retains the original analysis-era file names; `REPRODUCTION_MAP.md` maps package identifiers to snapshot files)
- `run_all.sh` — reproduction driver; `requirements_freeze.txt` pins the archived Python 3.11 environment

To reproduce: run `./run_all.sh` from a fresh checkout. It rebuilds S5/S6/S9 from the archived per-cell scores, checks all key numbers against `EXPECTED_OUTPUTS.json`, and optionally rebuilds the raw DepMap tables with `--data <pkg-root>` (raw pkg1–4 matrices are large and rebuilt from public sources; see `DATA_SOURCES.md`). The checker resolves all paths relative to the script and accepts `--expected`, `--data`, and `--manuscript`.

Checking convention: key numbers tolerate 5×10⁻⁴, interval endpoints 1×10⁻³. Compare one by one with
`github/analysis/stability_checks/_audit_numbers.py`, driven by `EXPECTED_OUTPUTS.json`. A verification run reported 74 checks, 0 mismatched.
