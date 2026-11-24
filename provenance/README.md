# G2CP beta-trained / dcic-trained training-state provenance

Raw records behind the Methods statement about how the two G2CP training states were built and what their
held-out set means. These were kept outside the earlier submission packages and are collected here so a
reviewer can check the claim directly. Nothing here changes a result; it documents the split, the
fine-tuning configuration, and the completed leakage audit.

## The question this answers

Does "held-out drugs" mean the evaluated drugs were excluded from training? For the two trained states used
in the paper the answer is yes, established by explicit recomputation on 19 Sep 2026
(`78_leakage_audit_20260919.md`, gates 8/8; `81_weight_provenance_20260919.md`, gates 6/6). The governing
result is `81_weight_provenance.json` → `holdout_vs_weights`: for each adapted weight, the number of
held-out drugs found in that weight's adaptation-training drug set is **0 / 3,204**, and all 3,204 fall in
that weight's held-out set. The beta-trained and dcic-trained adaptation holdouts are byte-identical
(md5 `29fcd23a0cf1d3dfa0712b8f035dfde6`), which is the basis for "the held-out drug list was identical".
`78_leakage_audit_20260919.md` §1 gives the same result as a gate: held-out drugs appearing in training =
0 (beta2020 3204/3204, dcic2021 3204/3204), with every held-out drug's full row set assigned to test.

## Two holdouts, two roles — do not conflate

| Set | Size | Definition | Role |
|---|---|---|---|
| **Training-state holdout** | 3,204 drugs | full 32,039-drug vocabulary, `RandomState(0)`, last 10% (`verify_ft_20260911.py` L48–58; `p3_dcic_adapt_20260914.py` L80–92) | Held out from the domain adaptation that produced beta-trained and dcic-trained. **This is the strict holdout behind the training claim.** |
| **Cross-source evaluation subset** (`H = 2,037`) | 2,037 drugs | 20,370-drug intersection space (`g2cp_cache_rebuilt` ∩ 2021), `RandomState(0)`, last 10% (`controlled_swap_source_sensitivity.py` L83–88; `step55_v2_beta2020.py`) | The 11,275 evaluation pairs for the fixed-prediction matrix comparison. A different analysis set, not the adaptation training set. |

`20_split_provenance_20260918.md` §4, `78_leakage_audit_20260919.md` §5 and `81_weight_provenance.json`
→ `old_split` document that the 3,204-set and the 2,037-set are draws from the same 32,039-drug vocabulary
restricted to different subsets. Their overlap (192) and "3,204 ∩ old-train = 1,909" are arithmetic
consequences of two samplings, not leakage. `81_weight_provenance.json` → `verdict_ABC` records the
explicit verdict "非 A、非 B，是 C": the 1,909 figure belongs to the separate cross-source experiment
(`g2cp_cache_rebuilt`), on which the main weights were never adapted, so it does not contaminate the
main-model arm.

**What the 192 therefore implies for the paper's wording.** The same arithmetic runs the other way for
the evaluation set: 2,037 − 192 = **1,845 of the 2,037 evaluation drugs are not in the adaptation
holdout**, and since the 20,370-drug shared-release space sits inside the 32,039-drug vocabulary, those
1,845 fall in the 28,835-drug adaptation-training portion. Only 192 (9.4%) of the evaluated drugs are
strictly held out from adaptation training. The manuscript therefore calls these 2,037 drugs
**evaluation drugs**, not held-out drugs, and Methods states the three sets separately (upstream base,
adaptation holdout, cross-matrix evaluation set). Nothing in the matrix-replacement comparison depends on
that label: predictions are fixed and only the evaluation matrix changes.

## Claim-by-claim mapping

| Methods claim | Evidence |
|---|---|
| "using random seed 0, 10% of perturbations were held out by drug identity" | `verify_ft_20260911.py` L48–58 and `p3_dcic_adapt_20260914.py` L80–92; `drug_holdout_beta2020.json` / `drug_holdout_dcic2021.json` (`split_unit = "drug"`, `seed = 0`, `RandomState(0).permutation` then last 10%). |
| "all records of those held-out drugs were excluded from training across doses and cell lines" | `78_leakage_audit_20260919.md` §1 and gate G78a: held-out drugs appearing in training = 0 (beta2020 3204/3204, dcic2021 3204/3204); `81_weight_provenance.json` `holdout_vs_weights`: 0/3,204 in train for all three adapted weights. |
| "the held-out drug list was identical" | both `drug_holdout_*.json` carry `ids_md5 = f5333fac8ba1d833aa27c8b03b29c308`; the two states' adaptation holdouts carry md5 `29fcd23a0cf1d3dfa0712b8f035dfde6`; `81_weight_provenance.json` §`holdout_vs_weights` gives 0/3,204 in train and 3,204/3,204 in test for all three weights. |
| shared base checkpoint and hyperparameters | `20_split_provenance_20260918.md` §5 and the 2026-09-27 work log: both states adapt the same base `g2cp_full_cpi_v7.pt` with an identical protocol, only the cache differs. `ft3_cellbal_20260911.py` L53–56 (`epochs=24`, `lr=3e-4`, `wd=1e-4`, `batch=256`), cosine schedule L213, MSE + (1 − PCC) loss and best-validation-PCC checkpoint. |

## Scope note

The three adapted weights (beta-trained, the beta_v2 control, dcic-trained) are fully documented and pass
the leakage gates. The upstream base `g2cp_full_cpi_v7.pt` is the built-in checkpoint from the public G2CP
package: its own training split is not shipped and is not in the archive, so whether the base has seen
these drugs cannot be closed from project data (`78_leakage_audit_20260919.md` §5.1,
`81_weight_provenance.json` G6/G81f). The paper's claim concerns the two trained evaluation states, both
of which are strict holdouts; no claim is made about the internal pretraining corpus of the upstream base.

## File index

- `20_split_provenance_20260918.md` — forensic record of 20,370 / 2,037 and the shared two-arm holdout.
- `78_leakage_audit_20260919.md` — 8/8 leakage gates; 0 held-out drugs in training; chemical-twin caveat.
- `81_weight_provenance_20260919.md` and `81_weight_provenance.json` — the A/B/C verdict that the main
  weights' adaptation holdout is identical to the strict holdout, with 0/3,204 in training.
- `83_in_domain_fitted_*` context: GSE70138 is in-domain fitted and is reference only; documented in the
  audit records above, not re-derived here.
- `drug_holdout_beta2020.json`, `drug_holdout_dcic2021.json` — the holdout recipe and ids_md5.
- `controlled_swap_source_sensitivity.py` — the split code for the 20,370 / 2,037 evaluation subset.
- `ft3_cellbal_20260911.py` — the fine-tuning hyperparameters.
- `holdout_readings_registry.json` — reading-level registry of the controlled source swap.
