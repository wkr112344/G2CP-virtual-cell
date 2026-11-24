# The two training runs behind `beta-trained` and `dcic-trained`

These are the exact scripts that produced the two adapted checkpoints used in the paper. They are
archived so the hyperparameters quoted in Methods ("Prediction models") can be checked against the
code that ran, instead of against a settings summary.

| File | Checkpoint produced | Cache | Role |
|---|---|---|---|
| `verify_ft_20260911.py` | `assets/g2cp_v7_beta_ft.pt` | `data/g2cp_cache_beta_all` | beta-trained state |
| `p3_dcic_adapt_20260914.py` | `assets/g2cp_v7_dcic_ft.pt` | `data/g2cp_cache_2021` | dcic-trained state |
| `verify_ft2_20260911.py` | `assets/g2cp_v7_beta_ft2_both.pt` | `data/g2cp_cache_beta_v2` | control arm, **not** one of the two states compared in the paper |

## Where each Methods value comes from

| Methods clause | `verify_ft_20260911.py` | `p3_dcic_adapt_20260914.py` |
|---|---|---|
| seed 0 | L28 `SEED, EPOCHS, LR, WD, BATCH = 0, 24, 3e-4, 1e-4, 256`; L29 `torch.manual_seed(SEED)` | L40 `SEED, LR, WD, BATCH = 0, 3e-4, 1e-4, 256`; L63 `torch.manual_seed(SEED)` |
| 24 epochs | L28, same line | L55 `--epochs` default 24 |
| learning rate 3×10⁻⁴ | L28 | L40 |
| weight decay 10⁻⁴ | L28 | L40 |
| batch size 256 | L28 | L40 |
| cosine schedule | L118 `CosineAnnealingLR(opt, T_max=EPOCHS)` | L153 `CosineAnnealingLR(opt, T_max=EPOCHS)` |
| same loss, MSE + (1 − PCC) | L13 header; loss assembly in `main()` | L11 header; loss assembly in `main()` |
| same base checkpoint | `BASE_CKPT` = `assets/g2cp_full_cpi_v7.pt` | L38 `BASE_CKPT` = `assets/g2cp_full_cpi_v7.pt` |
| best-checkpoint selection on validation PCC | `val_pcc_best` in the results JSON, L171 | same pattern in `main()` |

The control arm (`verify_ft2_20260911.py`) is archived for completeness but uses different settings
(32 epochs, learning rate 2×10⁻⁴ by default, `--loss both`). It is not one of the two states the
manuscript compares, and no manuscript sentence quotes its settings.

## What is not reproducible from this archive

Both scripts read the training caches (`meta.npz`, `y.npy`, `drug_fps.npy` and the 12,328-gene target
matrices, several GB each) from the author's working directory, and write checkpoints into `assets/`.
Those caches are not redistributed here (see `DATA_SOURCES.md` for the rebuild path). What this folder
does support is the parameter claim itself: every value quoted in Methods is a literal in these files,
and the split they apply is the one audited in `provenance/` (drug-level, `RandomState(0)`, last 10% of
the 32,039-drug vocabulary → 3,204 held out).
