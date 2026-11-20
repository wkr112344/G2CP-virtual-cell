# Supplementary Material S14: LINCS cross-reference-product robustness and alignment checks

This file retains only the cross-reference checks used directly by the current GigaScience manuscript. Content from
the old version relating to LayerDiag, cancer/non-cancer stratification, CPI, genetic perturbation, or other
material that has left the main line has been removed from the submission package.

## S14.1 Main fixed-output comparison

With the same beta-trained frozen prediction, the same 64 common cell lines, the same 11,275 common
`(cell line, drug)` pairs, and the same scoring protocol, the per-sample PCC is 0.3680 for MODZ-based
level5beta2020 and 0.0724 for CD-based dcic2021, giving `Δ_ref = +0.2956`. This difference is a full
data-product replacement contrast and is not interpreted as the causal effect of any single preprocessing step.

## S14.2 Training-exposure check

Among the 2,037 controlled ranking candidates, 192 drugs appear in neither arm's currently traceable fine-tuning
training data, with `ΔPCC = +0.2289`. After further excluding drugs whose drug ID has been seen or that have an
exactly identical ECFP4 fingerprint on either training side, the 55 fingerprint-clean drugs have
`ΔPCC = +0.3118`. These results show that the main difference cannot be explained by direct drug exposure or by
identical fingerprints in the currently traceable fine-tuning training alone; the earlier base checkpoint's training
composition is incomplete, so no stronger extrapolation is made. The per-drug table is in S22.

## S14.3 Drug ranking and disease-signature shortlists

For the performance ranking of the 2,037 candidate drugs, the cross-product Spearman = 0.5675, the arm-A top-10%
retention rate in arm B is 59.80%, and Jaccard = 0.4266; however, 12.25% of the arm-A top-10% fall out of the arm-B
top-50%. On the 1,399 commonly evaluable drugs, the cross-product Spearman of the three disease-directional
signatures is 0.498–0.540 and the top-10% retention rate is 33.6%–42.9%, showing that global rank stability and
top-k/shortlist stability are not the same thing. The ranking table is in S20.

## S14.4 Fixed-output re-evaluation of seven published predictors

Figure 3 of the main text holds fixed the existing prediction matrices of CIGER, DeepCE, MultiDCP, PertDiT, PRnet,
TranSiGen, and XPert, replacing only the corresponding Level-5 reference. The seven predictors show a consistent
direction of paired `ΔPCC`, with median 0.3993 and range 0.2599–0.6450, and every bootstrap 95% CI lower bound
above 0. The per-model checkable artifacts are in the corresponding frozen Zenodo version of record; this local
submission package does not duplicate the large intermediate matrices.

## S14.5 Interpretation boundary

The cross-product difference in LINCS may simultaneously include signature estimation, normalization, replicate
aggregation, quality control, dynamic range, measurement repeatability, and processed-construct definition. The
current design has no within-product test-retest or technical-replicate noise baseline, so these sources are not
further decomposed.

## S14.6 Data-source layer: content differences of the common signatures and common-transformation checks

The two Level-5 products have a signature-key overlap of 99.61%, yet the 11,275 common evaluable responses in the
main text have a median per-row Pearson correlation of only 0.4172 across the 978 LINCS landmark genes. To rule out
artifacts from key alignment and simple numerical transforms, we realigned condition names and genes on the original
release matrices sharing signature IDs, obtaining a median per-row Pearson of 0.4262 and a median cosine of 0.4224,
the same order of magnitude as the main analysis. Per-row z-scoring, positive scaling, per-gene centering/z-scoring,
and per-gene calibration all failed to remove the main structural difference.

These checks only show that the difference is not caused solely by the simple alignment/transformation issues tested;
the generative lineages of MODZ and Characteristic Direction involve several differences, and this paper does not
attribute `Δ_ref` uniquely to normalization, replicate aggregation, quality control, or any single algorithmic step.

## S14.7 Gene-panel sensitivity

Switching the scoring gene panel to the 978 LINCS landmark genes increases the absolute `ΔPCC` while also raising
the top-10% overlap. This shows that absolute performance differences and rank stability can move in different
directions, so the main text reports both score-shift and rank/top-k metrics.

## S14.8 Common-set sensitivity

The main-text 2×2 uses the 11,275 pairs fully aligned with the main result; it is also repeated on the larger common
set `E_common = 15,990`, where the interaction direction is consistent but the absolute values differ. The two
results are reported separately to avoid mixing set changes into the same effect size as the reference replacement.
Full results are in S25.

## S14.9 Alignability check of other perturbation-transcriptome resources

We assessed whether resources such as GSE70138 and CPJUMP1 can provide the same task, the same endpoint, and a second
published processed reference product constructible on the same scoring unit as the main LINCS analysis. The available
material does not provide a combination that supports the same kind of controlled replication, so this paper does not
treat these resources as an equivalent replication; the cross-ecosystem test turns instead to DepMap CRISPR
gene-effect products (S23).

## S14.10 Interpretation of the main fixed-output result by the matched 2×2

Fully aligned with the 11,275-pair main fixed-output comparison, the 2×2 gives: beta-trained reference contrast
`+0.2956`, dcic-trained reference contrast `+0.0171`, difference-in-differences interaction
`+0.2785 [0.2741, 0.2828]`. The reference contrast differs markedly between the two training states, so the main text
frames the result as reference-product sensitivity being closely tied to training–evaluation target matching, rather
than as a fixed "product effect". The full four cells and the larger common-set sensitivity are in S25.

## S14.11 Interpretation and reporting caliber

This manuscript interprets its results as a comparative benchmark robustness / reproducibility study. The overall
drug ranking in LINCS retains moderate structure, but top-k and disease-signature shortlists are more sensitive; in
DepMap, CERES/Chronos are globally highly consistent, yet the performance reading and top-k recovery of the fixed
DeepDEP prediction still change by a small but measurable amount. DepMap thus provides a cross-ecosystem reproduction
of the phenomenon, not a direct replication of LINCS's specific effect size.
