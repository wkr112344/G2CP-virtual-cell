# G2CP — A 162-Cell-Line Genome-Wide Virtual Cell Platform

Independent reimplementation and large-scale extension of **UniPert-G2CP** (Li et al., *Cell*, 2026), a unified framework for predicting transcriptomic responses to genetic and chemical perturbations.

[![Preprint](https://img.shields.io/badge/bioRxiv-preprint-blue)](https://www.biorxiv.org/)
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

| Metric | Value |
|---|---|
| Genetic perturbation PCC (held-out) | 0.442 |
| Novel drug PCC (held-out) | 0.3047 |
| Directional accuracy (top-5% genes) | 73.8% |
| CPI enrichment factor (top 0.5%) | 139 (training-set ranking consistency; 109 strict hold-out) |
| Mechanism-clustering SMD | 1.636 (reference comparison; ECFP4 Tanimoto baseline 1.613) |
| Cell lines | 162 |
| Compounds | 32,039 |

See the [preprint](./biorxiv_submission.html) for full details.

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
- sciPlex3, ChEMBL 37
- GEO: GSE61302, GSE22886, GSE60235, and additional series

**Model weights** and **preprocessed datasets** are available on Zenodo/Figshare (DOI to be assigned).

The original study's open resources: [GitHub](https://github.com/lynn-1998/UniPert-G2CP_reproduce) | [Zenodo](https://zenodo.org/records/20355906)

## Citation

```bibtex
@article{wei2025g2cp,
  title={Reproducing and Scaling UniPert-G2CP: A 162-Cell-Line Genome-Wide Platform for Perturbation-to-Phenotype Prediction},
  author={Wei, Kairui},
  journal={bioRxiv},
  year={2026},
  doi={TBD}
}
```

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Author

**Kairui Wei** — School of Clinical Medicine, Xinjiang Medical University, Urumqi 830011, China
