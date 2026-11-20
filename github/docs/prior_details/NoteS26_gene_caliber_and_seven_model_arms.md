# Note S26 — Caliber disambiguation: 978 vs 963 gene axis, and the seven-model re-evaluation arms

> Purpose: this note supplies ready-to-paste clarification text for the **final English manuscript** (Methods /
> Data availability) and the **Zenodo version-of-record README**, so that reviewers do not misread 978 vs 963, or the
> "two arms of the seven-model re-evaluation", as a caliber contradiction.
> Underlying table: `TableS24_seven_predictor_fixed_output.csv` (per-model two-arm PCC, ΔPCC and bootstrap 95% CI).
> Source of truth for the numbers: `220_dpb_crossversion.json` (schema `dpb_crossversion/v1`, gene axis = 978,
> n_rows_paired = 9562, n_drugs = 1422, drug-level PCC, bootstrap B = 2000, seed 20260920).

---

## A. Gene axis: 978 and 963 are two alignment layers of the same gene axis, not a dimensionality contradiction

**Conclusion (paste into the English manuscript + Zenodo README):**

> **Gene-axis caliber (978 vs 963).** All primary scoring in this study is performed on the **978 LINCS L1000 landmark genes**, identified by Entrez / canonical HGNC symbols and sliced from the 12,328-gene space via the fixed 12,328→978 column index map (`internal_col`, length 978; canonicalization source `hgnc_symbols.tsv`, md5 `faf032b2…`). The figure **963** that appeared in an earlier supplement referred to a *naive* raw-release symbol alignment: 15 of the 978 landmark genes have since undergone HGNC symbol renames (old→new), so a release carrying current symbols fails a literal string match on exactly those 15, yielding 963 = 978 − 15. These 15 genes are **not** genuinely absent — they are recovered by HGNC alias / previous-symbol canonicalization, which restores the full 978. Thus **978 and 963 describe two layers of the same gene-axis alignment (canonical vs naive symbol match), not two different gene sets and not a dimensionality contradiction.** The main text reports 978 (the canonical count used for all primary scoring); the historical 963 reflected a raw-release naive-match intermediate and should not be read as a reduced gene panel.

Key points:

- 978 = the LINCS landmarks, **with HGNC canonicalization** (978/978 matched).
- 963 = when the raw release is matched by **literal symbols without canonicalization**, the 15 renamed landmark genes
  fail to match → 978 − 15 = 963; after canonicalization it returns to 978.
- The two are **two alignment layers of the same gene axis** — not two gene sets, and certainly not a
  dimensionality contradiction.
- Note: a different "963 column" in the algorithmic paper is the intersection of "978 landmarks ∩ columns actually
  measurable in the genetic corpus"; its cause is different and must not be conflated with the 963 here.

---

## B. The "two arms" of the seven-model re-evaluation (aligned with the Figure 3 caption)

**Key facts (to prevent misreading by reviewers):**

- The seven-model re-evaluation is a **fixed-output replacement contrast**: each published predictor's existing
  prediction matrix (CIGER / DeepCE / MultiDCP / PertDiT / PRnet / TranSiGen / XPert) is frozen, and only the Level-5
  **measured reference product** used for scoring is replaced.
- Each frozen prediction is scored against three references: **(i) the predictor's own SDST reference** (a third-party
  processed version, row-content correlation 0.3524 with level5beta2020 — the model's "training/native" reference),
  **(ii) LINCS level5beta2020 (MODZ)**, and **(iii) LINCS dcic2021 (CD)**.
- The **"paired ΔPCC median 0.3993, range 0.2599–0.6450, every 95% CI lower bound > 0"** reported in main-text
  section 4.1.5 and Figure 3 corresponds to **ΔPCC = PCC(own SDST reference) − PCC(dcic2021)** (i.e. the
  `DeltaPCC_own_minus_dcic2021` column of `TableS24`; across the seven models that column is
  0.2889 / 0.4397 / 0.2599 / 0.3993 / 0.3122 / 0.6273 / 0.6450, median 0.3993).
- **The Figure 3 caption's "arm A = MODZ-based level5beta2020" borrows the wording of Figure 1 and is not accurate for
  the seven-model panel**: the "arm A" of the seven-model re-evaluation is each predictor's **own SDST reference
  output** (a third-party processed version, not the LINCS level5beta2020 measured product); level5beta2020 is arm A
  only for the G2CP main analysis (Figure 1).
- For full transparency, `TableS24` also gives **PCC on LINCS level5beta2020** and the **β2020-vs-dcic2021 ΔPCC**
  (= PCC_beta2020 − PCC_dcic2021, very small, about 0.01–0.10), because the frozen external predictions score poorly
  on either LINCS product anyway; this further shows that the magnitude of 0.3993 comes from the "own reference vs
  dcic2021" contrast, not from "swapping the two LINCS versions".

**Recommended text for the English manuscript (section 4.1.5 + Figure 3 caption fix):**

> **Seven-model re-evaluation arms.** Each published predictor's prediction matrix was held fixed and re-scored against three references: the predictor's own SDST reference (a third-party processed version, nearest to level5beta2020 with row-content correlation 0.3524), the LINCS level5beta2020 (MODZ) product, and the LINCS dcic2021 (CD) product — all on the 978-gene landmark axis. The paired ΔPCC reported here is ΔPCC = PCC(own SDST reference) − PCC(dcic2021); across the seven predictors its median is 0.3993 (range 0.2599–0.6450) and every bootstrap 95% CI lower bound is above 0. (The two LINCS products alone, PCC(level5beta2020) − PCC(dcic2021), give a much smaller ΔPCC because the frozen external predictions score poorly on either LINCS product; see Table S24.) The earlier "arm A = level5beta2020" wording in the figure note applied to the G2CP main analysis (Fig. 1) and is not the arm definition for this seven-model panel.

**Recommended Figure 3 caption:**

> Figure 3. Cross-product re-evaluation of seven published predictors' fixed outputs. Each prediction matrix is held fixed; only the Level-5 reference product used for scoring is changed — the predictor's own SDST reference versus the LINCS dcic2021 (CD) product, on the 978-gene landmark axis. All seven predictors show a consistent direction of ΔPCC (median 0.3993; range 0.2599–0.6450; every bootstrap 95% CI lower bound > 0), but the magnitude differs across architectures.

---

## C. Link to the Zenodo README

Add the gene-caliber explanation of section A (978 with HGNC canonicalization; 963 as a naive literal-match
intermediate) to the `README.md` §/Data-availability of the Zenodo version-of-record
(`10.5281/zenodo.22720615`), and state explicitly that:

- `220_dpb_crossversion.json` and `dpb_crossversion_boot.npz` (sha256 `3140df42…`) in `results/` are the source of
  truth for the seven-model re-evaluation's per-model ΔPCC and bootstrap vectors;
- `TableS24_seven_predictor_fixed_output.csv` in this supplementary package is the per-model human-readable table.
