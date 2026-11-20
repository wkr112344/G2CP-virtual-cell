# Supplementary Material S13: scoring objects, statistical calibers, and provenance details

This file retains only the content that directly corresponds to the current GigaScience manuscript. The analysis
goal is to compare how published processed reference products affect benchmark readings; it no longer contains
earlier LayerDiag, CPI, genetics-side, or other material that has left the current manuscript's main line.

## S13.1 Fixed-output reference-product replacement contrast

The main analysis defines `Δ_ref = M(P_fixed, E_A) − M(P_fixed, E_B)`. `P_fixed` is the same frozen prediction,
`E_A/E_B` are two published processed reference products, and `M` is a fixed scoring protocol. The commonly
evaluable set, the held-out unit, the aggregation, and the metrics are all held constant.

## S13.2 Matched training-product × evaluation-product 2×2

The main-text 2×2 uses exactly the same 11,275 common `(cell line, drug)` evaluation pairs as the main result.
The two prediction states are beta-trained and dcic2021-trained, scored on the beta and dcic2021 references
respectively. The difference-in-differences interaction is defined as `I = (M_BB − M_BD) − (M_DB − M_DD)`;
the full four-cell result is in S25.

## S13.3 Cluster bootstrap

For the main analysis, the two arms, the paired difference, and the 2×2 interaction all resample held-out
**drugs** as the cluster unit. The main-text comparison uses B = 2,000; the 2×2 likewise uses a compound-cluster
bootstrap over the 2,037 held-out drugs.

## S13.4 Commonly evaluable set

The main analysis uses 64 common cell lines and 11,275 common `(cell line, drug)` pairs. The larger
`E_common = 15,990` is used only as a 2×2 sensitivity analysis, to avoid conflating absolute values computed on
different common sets.

## S13.5 Training-exposure sensitivity

The strict-unseen and fingerprint-clean analyses cover only the currently traceable fine-tuning training exposure.
The full training composition of the earlier base checkpoint was not preserved, so these subset results cannot rule
out all historical training exposure; they are used only to test whether the currently traceable direct drug
exposure, or an exactly identical ECFP4 fingerprint, is sufficient to explain the main difference. Per-drug results
are in S22; the identity audit is in S19.

## S13.6 Principle for identifying scoring objects

This work records source/accession, processing level, named release/build, and the processed object that actually
enters scoring as distinct layers. The scoring object behind each main-text number is locked by immutable files,
a manifest, and checksums. The 12-item L1000 reporting/provenance audit is in S21; representative object tracking
is in S21a.

## S13.7 Full provenance chain of the `dcic2021` processed reference

This note strings the dcic2021 evaluation object used by the main-text comparison arm (`data/g2cp_cache_2021/`)
into a hop-by-hop, checkable chain from "upstream distribution" to "the matrix that actually enters scoring", and
gives a per-file checksum. The goal is that "the exact file being scored" no longer depends on prose in the paper,
but is uniquely locked down by checksums.

## Provenance chain (top-down)

1. **Upstream distribution (SigCom / DCIC 2021)**
   - Original distributed file: `cp_coeff_mat.gctx` (the CD coefficient matrix of LINCS-sigs-2021; CD =
     Characteristic Direction, a processing family different from the β main source MODZ used here).
   - Source URL: `https://lincs-dcic.s3.amazonaws.com/LINCS-sigs-2021/gctx/cd-coefficient/cp_coeff_mat.gctx`
     (taken from this project's `Downloads/G2CP_push/download_level5.py`).
   - Local copy: `Desktop/cp_coeff_mat.gctx`, 36,084,518,760 B, MD5
     `12ee07d08c0368e77914113b3681eb39` (computed 2026-09-27).
   - Acquisition date: not on record (stated as such).

2. **`step44a_build_pairs_2021.py`** — build the (drug × cell line) pair-mean matrix from the gctx
   - Filtering: cell lines from the v7-162 vocabulary (bidirectional alias mapping), drugs from the v7 32,039 vocabulary.
   - Output: `rows_gctxorder.npy` (156,931 × 12,327) + `pairs_meta.npz` (cell_name / cell_idx / drug_id / drug_idx).

3. **`step44b_map_genes.py`** — map v7 HVG (12,328) ↔ gctx columns
   - Through HGNC canonicalization (sources `hgnc_symbols.tsv`, `geneinfo_beta.txt`), map the 12,328 v7 HVGs onto
     the gctx's 12,327 columns.
   - Output: `gene_map.json` (length 12,328; gctx column index or −1).

4. **`step44c_assemble_2021.py`** — assemble the final cache
   - Output: `y.npy` (156,931 × 12,328, in HVG order) + `meta.npz` (`kind/key/cell/gene_vocab/drug_vocab/cl_names/hvg/col2row`,
     inheriting the base vocabulary indices) + `drug_fps.npy` (copied from cache_full, aligned to the ckpt drug vocabulary).

5. **The frozen evaluation object (the matrix that actually enters scoring)**
   - `data/g2cp_cache_2021/y.npy` — the "evaluation object" of the dcic2021 arm.
   - The main-text comparison arm's held-out PCC on this object = 0.0724 (controlled 11,275-pair table).

## Per-file checksums (`results/dcic2021_cache_checksums_20260925.json`, computed 2026-09-25)

| File | MD5 | Size (B) | Note |
|---|---|---|---|
| `y.npy` | `be8c09dc312f0ee7ba4c10a1cb0979dd` | 7,738,581,600 | SHA256 `351362fca7e0326b9594fbb92bf9bfae171534d63ff171073922d97767faa9c4` (the evaluation object itself) |
| `rows_gctxorder.npy` | `88cecb25e51c99e89034ad8f2641e858` | 7,737,953,876 | step-2 intermediate |
| `pairs_meta.npz` | `6b9fdafac216ab77a008b525d770badb` | 2,690,208 | pair metadata |
| `gene_map.json` | `d4fbe9bd879834bc2cdd9cabe605bd48` | 75,177 | step-3 output |
| `meta.npz` | `15eb3c8e45a04ab3f0ddb9c0bbb189bc` | 2,169,832 | vocabularies / indices |
| `drug_fps.npy` | `302f2eec4ad58dcc215c2b21a43dfc2f` | 262,463,616 | drug fingerprints |
| `sigidx_maxDose.npz` | `9828af88cc3907f7a8dc47f860fb03d3` | 3,206,308 | signature index |
| `sigidx_sigALL.npz` | `bf90676af82bbf24c5a0aa4f2f4e8d4f` | 14,531,684 | signature index |
| `sigidx_sigT24.npz` | `a8ef5ff5b755f769e7065ed2d97e5095` | 9,945,060 | signature index |

## Symmetric comparison with the β main-source evaluation object (supporting the main-text claim "only the file changes, everything else stays the same")

| Hop | β main source (level5beta2020) | dcic2021 |
|---|---|---|
| Upstream file | `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx` (GEO GSE92742, Level 5 COMPZ MODZ) | `cp_coeff_mat.gctx` (LINCS-sigs-2021 CD coefficients) |
| Build script | `step58b_beta_build_v2.py` (+ siginfo / cellinfo / geneinfo) | `step44a/b/c_*` |
| Evaluation object | `data/g2cp_cache_beta_v2/y.npy` | `data/g2cp_cache_2021/y.npy` |
| Held-out PCC (controlled 11,275 pairs) | **0.3680** | **0.0724** |
| The only layer changed | — | processing family (MODZ → CD) + release version (2020 → 2021) |

> Both arms share the same base `g2cp_full_cpi_v7.pt`, the same vocabularies (cl_names v7-162, drug_vocab v7 32,039),
> the same held-out split (last 10%, 2,037 drugs), and the same 11,275-pair evaluation table; see main text 3.2,
> 4.1.1–4.1.4 and Supplementary Material S14.6.
