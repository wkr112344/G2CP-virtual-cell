# Supplementary S23–S28 (and the p = 0.0090 statistic) — rebuild package

Everything here is path-parameterised: no script contains a machine-specific absolute path.
Set one environment variable (or edit `config_paths.py`) and the scripts read from your own copy.

## 1. Directory layout expected

```
<GIGASCIENCE_ROOT>/
├── source_data/                                  # from the Zenodo record (see §2)
│   └── lincs_index/
│       └── 93_gene_map_12328_to_978.json         # 12,328 → 978 landmark axis
├── external/                                     # third-party data you fetch yourself (see §3)
│   ├── drug-perturbation-benchmark/
│   │   ├── data/XPert/processed_data/l1000_sdst_78453.h5ad
│   │   └── results/main_benchmark/{xpert,transigen}/fold_{0..4}/predictions/
│   │         ├── test_predictions.npy
│   │         ├── test_ground_truth.npy
│   │         └── test_sample_ids.csv
│   ├── g2cp_cache_beta_v2/                       # {pairs_meta.npz, y.npy}  (beta2020 / MODZ)
│   ├── g2cp_cache_2021/                          # {pairs_meta.npz, y.npy}  (dcic2021 / CD)
│   └── msigdb/c2.cgp.v2023.2.Hs.symbols.gmt      # for S25 gene counts
└── work/                                         # scripts write outputs here
```

```bash
# bash
export GIGASCIENCE_ROOT=/path/to/gigascience_repro

# Windows cmd
set GIGASCIENCE_ROOT=D:\gigascience_repro

# Windows PowerShell
$env:GIGASCIENCE_ROOT = "D:\gigascience_repro"
```

Check your setup first:

```bash
python config_paths.py     # prints every resolved path
```

## 2. Where `source_data/` comes from

`sources:source_data/lincs_index/` and `sources:source_data/lincs_reference_matrices/` are distributed
in the Zenodo record for this paper. Download the record, unzip, and put the `source_data/` folder
directly under `GIGASCIENCE_ROOT`.

## 3. Where the `external/` data comes from

| Folder | Description | How to obtain | Fixed version |
|---|---|---|---|
| `drug-perturbation-benchmark/` | Bai et al. benchmark repo — XPert / TranSiGen **test-set predictions** used here | GitHub `https://github.com/baijinming97/drug-perturbation-benchmark`; predictions archived at Zenodo `https://zenodo.org/records/20081274` | see §5 (commit SHA) |
| `drug-perturbation-benchmark/data/XPert/processed_data/l1000_sdst_78453.h5ad` | LINCS SDST processed expression (XPert) | same repo / Zenodo record above | as above |
| `g2cp_cache_beta_v2/`, `g2cp_cache_2021/` | per-(cell, drug) 978-gene reference vectors with `pairs_meta.npz` | rebuilt deterministically from LINCS level-5 public products (GSE92742 for beta2020; SigCom LINCS / DCIC for dcic2021) — see `DATA_SOURCES.md` in the repo root | beta2020; dcic2021 |
| `msigdb/c2.cgp.v2023.2.Hs.symbols.gmt` | MSigDB C2:CGP, for S25 UP/DN gene counts | msigdb.org (registration required) — cannot be redistributed | v2023.2 |
| `siginfo_beta.txt` | beta2020 (GSE92742) level-5 signature metadata, for the S27 condition / well-token audit | GEO `GSE92742_Broad_LINCS_sig_info.txt.gz`, unzip to `external/siginfo_beta.txt` (~465 MB); or pass `--siginfo /path/to/siginfo_beta.txt` | GSE92742 |

## 4. Commands

```bash
cd github/analysis/supplementary_rebuild_S23_S28

# (a) per-row three-reference scores — produced by the main analysis, reused here
python ../../analysis/stability_checks/score_three_refs.py     # or supply your own CSV

# (b) S29 / p = 0.0090 paired bootstrap  → S29_paired_advantage_bootstrap.csv
python build_S29_paired_bootstrap.py --by "$GIGASCIENCE_ROOT/work/score_three_refs_by_drug.csv"

# (c) S28 random-mis-pairing control     → S28_permutation_scheme_comparison.csv
python build_S28_v2.py

# (d) S27 condition / well-token audit    → S27_sdst_beta2020_condition_audit.csv
python build_S27_condition_audit.py
```

Expected runtimes (authors' machine, CPU only): (b) ~1 min, (c) ~6 min for all three schemes
(B = 2000; `S28_B` overrides the replicate count), (d) ~2 min.

## 5. Software versions and code revision

Recorded from the authors' environment. **The `drug-perturbation-benchmark` commit SHA must be pinned
from the archive you actually downloaded; do not write "latest".** Fill in the actual SHA before
submission:

| Item | Version |
|---|---|
| Python | 3.11 (the environment providing `h5py`); scripts also run under 3.13 with `h5py`, `numpy`, `pandas`, `scipy`, `openpyxl` |
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| h5py | 3.16.0 |
| scipy | (per `pip freeze` of the runtime env) |
| `drug-perturbation-benchmark` | **commit SHA: `<fill in>`** — `git -C external/drug-perturbation-benchmark rev-parse HEAD` |
| Zenodo XPert/TranSiGen predictions | record `20081274` |
| MSigDB C2:CGP | v2023.2 (human, symbols) |

Sections of the paper this package supports: S27 (condition audit), S28 (random-mis-pairing
negative control), S29 / main-text p = 0.0090 (paired bootstrap on the 1,343-drug subset).

## 6. What the three S28 permutation schemes are

* `A_positional_old` — records are grouped **by drug**, then mismatched **positionally**
  (row-to-row) across drug groups. Because the groups do not cover identical cell lines, this
  also changes the cell line. Kept only as a method comparison.
* `B_within_cell` — the cell line (and fold) are **held fixed**; within a `(cell, model, fold)`
  group, a row is swapped to a *different drug*'s 978-gene reference vector. Only the drug pairing
  changes. The 92 single-record groups have no other drug to swap with and therefore keep their
  original record; they are counted in `n_single_record_groups`.
* `C_within_cell_permutable_only` — scheme B with those 92 single-record groups removed from both
  the correctly paired and the permuted evaluation (6,800 of 6,892 records, 1,342 drugs).
  **This is the scheme cited in the main text.**

In all schemes the model prediction is untouched and the drug-equal-weight aggregation is identical
(fold-inner drug mean → cross-fold mean). Under all three schemes the six observed (correctly
matched) values exceed the 95% upper bound of the corresponding null, with empirical P = 0.0005,
the smallest value attainable with B = 2,000.

Every permutation in B and C is audited: within each permuted stratum a record must receive another
record's response whose drug identifier differs, the draw is rejected otherwise, and the final
assignment is re-checked position by position. Across 2,000 permutations per scheme 13.6 million
positions were checked, none kept its original drug, and the deterministic fallback construction was
never needed; the counts are written to `S28_permutation_constraint_check.csv`.

The drug subset is identical to main-text Table 4: after removing drug names affected by the
integer-ID collision (`pert_idx → pert_id` one-to-many) the benchmark has **1,343** drugs
(1,391 − 204). `build_S28_v2.py` self-checks the six correctly matched Pearson values against
Table 4; the frozen XPert predictions replay to within 2×10⁻⁴ of the table (attention fallback),
and the script reports the comparison instead of aborting.

## 7. The two-sided empirical P value, and its resolution floor

`build_S29_paired_bootstrap.py` computes

```
p = min(1, 2 × min(#{d ≤ 0}, #{d > 0}) / B),   B = 2000
```

This is a **bootstrap empirical P value**, not a permutation-test P value. With `B = 2000` the
empirical resolution is `2 / B = 0.001`; **when the smaller tail count is 0 the script reports the
floor `0.001`, never exactly `0`** — a bootstrap empirical P value cannot be 0.

| Advantage difference | Observed | 95% range | `#{d ≤ 0}` | `#{d > 0}` | Two-sided P |
|---|---|---|---|---|---|
| SDST − MODZ | +0.01453 | +0.01077–+0.01819 | 0 | 2000 | 0.001 |
| SDST − CD | +0.01156 | +0.00836–+0.01447 | 0 | 2000 | 0.001 |
| MODZ − CD | −0.00298 | −0.00510–−0.00088 | 1991 | 9 | 0.0090 |

The `p = 0.0090` quoted in the main text is `2 × 9 / 2000` and reproduces exactly. The paired
bootstrap uses the **same drug-index resample for both sides** of each comparison, and **each pair
resets its RNG** (`seed = 20261011`); chaining the RNG across pairs yields 0.011 instead of 0.0090.
`p00090_calculation_chain.py` additionally emits the `n_boot_le0` / `n_boot_gt0` tail counts so the
reported P value can be checked against its landing point.
