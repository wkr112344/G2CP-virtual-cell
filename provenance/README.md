# G2CP beta-trained / dcic-trained training-state provenance

Raw records behind the Methods sentence about how the two G2CP training states were built. These were
kept outside the earlier submission packages and are collected here so a reviewer can check the claim
directly. Nothing here changes a result; it documents the split and the fine-tuning configuration.

## What the main text claims, and which file supports it

| Methods claim | Evidence in this directory |
|---|---|
| "using random seed 0, 10% of perturbations were held out by drug identity" | `drug_holdout_beta2020.json` and `drug_holdout_dcic2021.json`: `split_unit = "drug"`, `id_space = "drug_id (BRD-…)"`, `seed = 0`, `recipe` uses `numpy.random.RandomState(0).permutation(n_universe)` then takes the last 10% (`perm[int(round(0.9*n_universe)):]`). Same implementation in `controlled_swap_source_sensitivity.py` lines 83–88: `perts = sorted(Dstar); n_te = int(len(perts)*0.9); perm = np.random.RandomState(0).permutation(len(perts)); H = set(perts[i] for i in perm[n_te:])`. |
| "all records of those held-out drugs were excluded from training across doses and cell lines" | Both `drug_holdout_*.json`: "test 行 = 该版本中 drug_id ∈ held 的全部行" (all rows of a held-out drug are test rows, at every dose and in every cell line). `n_train` 141,457 (beta2020) / 140,937 (dcic2021); `n_test` 16,065 / 15,994. |
| "the held-out drug list was identical" | Both JSON files carry `ids_md5 = f5333fac8ba1d833aa27c8b03b29c308`; `20_split_provenance_20260918.md` section 5 records the same md5 for beta2020 and dcic2021 and an exact intersection. |
| shared base checkpoint | `20_split_provenance_20260918.md` section 5: checkpoints `g2cp_v7_beta_ft.pt` / `g2cp_v7_dcic_ft.pt` are adaptations of the same base `g2cp_full_cpi_v7.pt`; the controlled-swap record (`holdout_readings_registry.json`) uses `g2cp_v7_beta_ft.pt` throughout. |
| 24 epochs, lr 3e-4, weight decay 1e-4, batch 256, cosine schedule, MSE + (1 − PCC), best checkpoint on validation PCC | `ft3_cellbal_20260911.py` defaults at lines 53–56 (`epochs=24`, `lr=3e-4`, `wd=1e-4`, `batch=256`), `CosineAnnealingLR` at line 213, `best` checkpoint selected on validation PCC at lines 214–215, and the MSE + (1 − PCC) loss in the same training loop. |

## Scope note

`20_split_provenance_20260918.md` is a dated forensic record with its own conclusion: the 20,370-drug
subset used for the controlled source-swap is the observation coverage of the early rebuilt cache, and
2,037 = 10% of that subset; the full per-release split uses 32,039 drugs with 3,204 held out. The two are
different analysis sets and must not be mixed. The main text's 2,037 held-out drugs refer to the
controlled source-swap set (11,275 evaluation pairs), while the training-state statement refers to the
per-release 10% drug holdout. Both use the same rule (seed 0, last 10% of sorted drug IDs).
