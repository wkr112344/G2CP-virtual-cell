# Supplementary Note S25. Training-product × evaluation-product matched 2×2

## S25.1 Main-text caliber: the 11,275 pairs aligned with 0.3680 / 0.0724

To analyze the matching between training target and evaluation reference, we build a 2×2 on the same 11,275 common
`(cell line, drug)` pairs, the same 978 landmark-gene panel, and the same scoring protocol as the main-text analysis.
The two prediction states are beta-trained and dcic-trained, scored on the beta and dcic2021 references respectively.
The drug-cluster bootstrap resamples the 2,037 held-out drugs as the unit.

| Training product | beta-eval | dcic2021-eval | beta − dcic contrast |
|---|---:|---:|---:|
| beta-trained | 0.3680 [0.3627, 0.3733] | 0.0724 [0.0683, 0.0765] | +0.2956 [0.2917, 0.2996] |
| dcic-trained | 0.1705 [0.1652, 0.1757] | 0.1534 [0.1485, 0.1581] | +0.0171 [0.0140, 0.0204] |

Difference-in-differences interaction:

`I = (M_BB − M_BD) − (M_DB − M_DD) = +0.2785 [0.2741, 0.2828]`.

The beta-trained prediction's spread across the two references is clearly larger than the dcic-trained prediction's.
The model ordering also changes with the evaluation product: on beta-eval, beta-trained > dcic-trained, whereas on
dcic-eval, dcic-trained > beta-trained. This shows that the effect of the reference product on the benchmark reading
is closely tied to the training target.

This 2×2 estimates the score difference under the joint action of training source and evaluation reference. The
cross-reference shift may still include processing-lineage, measurement noise, dynamic range, and processed-construct
reliability; lacking an independent within-product test-retest baseline, this analysis does not decompose these
sources further.

## S25.2 Larger common-set sensitivity: E_common = 15,990

This caliber covers 112 cell lines and tests whether the main-text structure depends on the 11,275-pair main set.

| Training product | beta-eval | dcic2021-eval | beta − dcic contrast |
|---|---:|---:|---:|
| beta-trained | 0.3438 | 0.1124 | +0.2314 |
| dcic-trained | 0.1792 | 0.1390 | +0.0402 |

The corresponding interaction = +0.1912. The larger common set and the main-text caliber give the same interaction
direction but different absolute values, so the two four-cell results are reported separately.
