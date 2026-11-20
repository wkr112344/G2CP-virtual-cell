"""Item 13: drug x cell crossed bootstrap (answers "repeated cell lines violate independence")
   Item 10: near-boundary fragility check for the disease shortlist (conservative caliber, no full bootstrap)

Data:
  per-row frozen prediction scores: gigascience_figwork2/GigaScience_GTEx_checked/source_data/
                        frozen_prediction_matrices/frozen_perrow_three_metrics.csv (11,275 rows)
  per-drug disease scores: gigascience_v5/figure_source_data/TableS_disease_perdrug_scored.csv
Output: crossed_bootstrap_delta.csv / disease_list_boundary_fragility.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(20260929)
B = 5000
SRC = Path('gigascience_figwork2/GigaScience_GTEx_checked/source_data/frozen_prediction_matrices/frozen_perrow_three_metrics.csv')

d = pd.read_csv(SRC)[['drug', 'cell', 'pearson_A_beta2020', 'pearson_B_dcic2021']].dropna()
drugs = np.sort(d.drug.unique()); cells = np.sort(d.cell.unique())
di = {x: i for i, x in enumerate(drugs)}; ci = {x: i for i, x in enumerate(cells)}
nD, nC = len(drugs), len(cells)

# per (drug, cell) sums/counts of A/B -> enables weighting by cell multiplicity
S = np.zeros((nD, nC, 2)); N = np.zeros((nD, nC))
for r in d.itertuples(index=False):
    i, j = di[r.drug], ci[r.cell]
    S[i, j, 0] += r.pearson_A_beta2020; S[i, j, 1] += r.pearson_B_dcic2021; N[i, j] += 1

def delta_from(drug_idx, cell_mult, rng=None):
    """Given drug indices (with repeats) and a cell-multiplicity vector, return Δ = mean_drugs(A) - mean_drugs(B)."""
    w = N[drug_idx, :] * cell_mult[None, :]
    num = (S[drug_idx, :, :] * cell_mult[None, :, None]).sum(axis=1)   # (n, 2)
    den = w.sum(axis=1)
    ok = den > 0
    mean = num[ok] / den[ok, None]
    return float(mean[:, 0].mean() - mean[:, 1].mean())

all_drugs = np.arange(nD); all_cells = np.ones(nC)
obs = delta_from(all_drugs, all_cells)
print('observed Δ (all drugs equal weight) = %.6f   [main text 0.2956]' % obs)

def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

# 1) original caliber: drug-cluster bootstrap only
drug_only = np.empty(B)
for b in range(B):
    drug_only[b] = delta_from(RNG.integers(0, nD, nD), all_cells)
# 2) cell-cluster bootstrap only
cell_only = np.empty(B)
for b in range(B):
    m = np.bincount(RNG.integers(0, nC, nC), minlength=nC).astype(float)
    cell_only[b] = delta_from(all_drugs, m)
# 3) crossed: resample drugs and cells simultaneously
crossed = np.empty(B)
for b in range(B):
    m = np.bincount(RNG.integers(0, nC, nC), minlength=nC).astype(float)
    crossed[b] = delta_from(RNG.integers(0, nD, nD), m)

rows = []
for name, v in [('drug_clustered_primary', drug_only), ('cell_clustered', cell_only), ('crossed_drug_x_cell', crossed)]:
    lo, hi = ci(v)
    rows.append({'scheme': name, 'estimate': obs, 'mean_boot': float(v.mean()),
                 'ci_low': lo, 'ci_high': hi, 'sd': float(v.std(ddof=1)),
                 'fraction_positive': float((v > 0).mean())})
    print('%-22s mean %.6f  CI [%.6f, %.6f]  SD %.6f  P(Δ>0)=%.4f' % (name, v.mean(), lo, hi, v.std(ddof=1), (v > 0).mean()))
pd.DataFrame(rows).to_csv('gigascience_upload_work/crossed_bootstrap_delta.csv', index=False)

# ---------------- Item 10: near-boundary fragility of the disease shortlist ----------------
dis = pd.read_csv('gigascience_v5/figure_source_data/TableS_disease_perdrug_scored.csv')
out = []
NAME = {'ACEVEDO_LIVER_TUMOR_VS_NORMAL_ADJACENT_TISSUE': 'Liver (Acevedo)',
        'RODRIGUES_THYROID_CARCINOMA_ANAPLASTIC': 'Thyroid ATC (Rodrigues)',
        'CASORELLI_ACUTE_PROMYELOCYTIC_LEUKEMIA': 'APL (Casorelli)'}
for sig, g in dis.groupby('signature'):
    for prod, col, rk in [('beta2020', 'score_A_beta_reversal', 'rank_A'),
                          ('dcic2021', 'score_B_dcic_reversal', 'rank_B')]:
        s = g.dropna(subset=[col]).sort_values(col, ascending=False).reset_index(drop=True)
        top10 = s.iloc[:10]; span = float(top10[col].iloc[0] - top10[col].iloc[-1])
        b10, b11, b20 = float(s[col].iloc[9]), float(s[col].iloc[10]), float(s[col].iloc[19])
        gap = b10 - b11
        tie = int((s[col] >= b11 - gap).sum())          # drugs effectively tied with rank 10
        out.append({'signature': NAME.get(sig, sig), 'reference': prod,
                    'n_drugs_ranked': len(s), 'score_rank10': b10, 'score_rank11': b11,
                    'score_rank20': b20, 'gap_10_11': gap, 'gap_10_20': b10 - b20,
                    'top10_score_span': span, 'boundary_gap_over_span': gap / span if span else np.nan,
                    'tie_band_size': tie})
b = pd.DataFrame(out)
b.to_csv('gigascience_upload_work/disease_list_boundary_fragility.csv', index=False)
print()
print('near-boundary fragility of the disease shortlist:')
print(b.round(4).to_string(index=False))
