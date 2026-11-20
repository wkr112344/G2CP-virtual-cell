"""Two patches:

A. independent bootstrap pairs (within-reference stability, 2000 rounds, two independent resamples per round)
   -> directly yields the Spearman and top-k overlap distributions for within-beta / within-dcic, and regenerates S17 from them.

B. crossed bootstrap for the 2x2 model comparison (resampling drug x cell simultaneously)
   four cells: beta-trained->beta, beta-trained->dcic, dcic-trained->beta, dcic-trained->dcic
   D_β = r(βtrain,βref) − r(dcic-train,βref)，D_dcic = r(βtrain,dcicref) − r(dcic-train,dcicref)
   I = D_β − D_dcic
   reports: the fraction with D_beta>0 and D_dcic<0 (whether the reversal holds), and the crossed 95% CI for I.

Data: GigaScience_Submission_Upload/supplementary/frozen_multimetric_perrow_scores.csv(11,275 rows, 63 cells x 2,037 drugs)
Output: within_reference_pair_stability.csv, crossed_bootstrap_2x2.csv, S17/S20 updates
"""
import numpy as np
import pandas as pd
from pathlib import Path

SRC = Path('gigascience_upload_work/GigaScience_Submission_Upload/supplementary/frozen_multimetric_perrow_scores.csv')
SUP = Path('gigascience_upload_work/GigaScience_Submission_Upload/supplementary')
ROUNDS = 2000
KS = [10, 20, 50, 100, 204, 500, 1000]
COLS = {'beta': 'beta_trained__beta2020', 'dcic': 'beta_trained__dcic2021',
        'd_beta': 'dcic_trained__beta2020', 'd_dcic': 'dcic_trained__dcic2021'}
rng = np.random.default_rng(20260930)

d = pd.read_csv(SRC)
d = d[['drug', 'cell'] + list(COLS.values())].dropna()
drugs = np.sort(d.drug.unique()); cells = np.sort(d.cell.unique())
di = {x: i for i, x in enumerate(drugs)}; ci = {x: i for i, x in enumerate(cells)}
N, C = len(drugs), len(cells)

# ---------- pre-arrange: per-drug row values (padded to max) and per (drug,cell) sums/counts ----------
d['_di'] = d.drug.map(di); d['_ci'] = d.cell.map(ci)
counts = np.zeros(N, dtype=int)
for i, n in d.groupby('_di').size().items():
    counts[i] = n
nmax = counts.max()
pad = {c: np.full((N, nmax), np.nan) for c in COLS}
sums = {c: np.zeros((N, C)) for c in COLS.values()}
ncnt = np.zeros((N, C))
pad = {}
for k, c in enumerate(COLS.values()):
    arr = np.full((N, nmax), np.nan)
    vals = d[c].to_numpy(float)
    pos = np.zeros(N, dtype=int)
    for idx, (i, v) in enumerate(zip(d['_di'].to_numpy(), vals)):
        arr[i, pos[i]] = v; pos[i] += 1
    pad[c] = arr
np.add.at(ncnt, (d['_di'].to_numpy(), d['_ci'].to_numpy()), 1.0)   # count accumulated exactly once
for c in COLS.values():
    np.add.at(sums[c], (d['_di'].to_numpy(), d['_ci'].to_numpy()), d[c].to_numpy(float))
assert ncnt.sum() == len(d), 'ncnt row count mismatch: %s vs %s' % (ncnt.sum(), len(d))

def row_boot_means(col, rng_):
    """Row-level resampling (resample n_i rows per drug) -> the drug score vector."""
    A = pad[col]
    idx = rng_.integers(0, counts[:, None], size=(N, nmax))
    picked = np.take_along_axis(A, idx, axis=1)
    mask = np.arange(nmax)[None, :] < counts[:, None]
    with np.errstate(invalid='ignore'):
        s = np.nanmean(np.where(mask, picked, np.nan), axis=1)
    s[counts == 1] = A[counts == 1, 0]
    return s

def rank_spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])

# ================= A. independent bootstrap pairs =================
print('=== A. within-reference independent bootstrap pairs (%d rounds) ===' % ROUNDS)
res = {}
for tag in ['beta', 'dcic']:
    col = COLS[tag]
    rho = np.empty(ROUNDS); ov = {k: np.empty(ROUNDS) for k in KS}
    for r in range(ROUNDS):
        s1 = row_boot_means(col, rng); s2 = row_boot_means(col, rng)
        rho[r] = rank_spearman(s1, s2)
        o1 = np.argsort(-s1, kind='stable'); o2 = np.argsort(-s2, kind='stable')
        for k in KS:
            ov[k][r] = len(set(o1[:k]) & set(o2[:k]))
    res[tag] = {'rho': rho, 'ov': ov}
    print('  %-5s Spearman %.4f [%.4f, %.4f] | top-10 %.2f [%.0f, %.0f] | top-204 %.1f [%.0f, %.0f]' % (
        tag, rho.mean(), np.percentile(rho, 2.5), np.percentile(rho, 97.5),
        ov[10].mean(), np.percentile(ov[10], 2.5), np.percentile(ov[10], 97.5),
        ov[204].mean(), np.percentile(ov[204], 2.5), np.percentile(ov[204], 97.5)))

# observed cross-reference (same 2,037-drug set)
obs = {tag: np.array([pad[COLS[tag]][i, :counts[i]].mean() for i in range(N)]) for tag in ['beta', 'dcic']}
o_b = np.argsort(-obs['beta'], kind='stable'); o_d = np.argsort(-obs['dcic'], kind='stable')
cross = {k: len(set(o_b[:k]) & set(o_d[:k])) for k in KS}
print('  observed cross-reference Spearman %.4f | top-10 %d | top-204 %d' % (
    rank_spearman(obs['beta'], obs['dcic']), cross[10], cross[204]))

rows = []
for k in KS:
    b, c = res['beta']['ov'][k], res['dcic']['ov'][k]
    rows.append({'k': k,
                 'within_beta_pair_mean': b.mean(), 'within_beta_pair_lo': np.percentile(b, 2.5),
                 'within_beta_pair_hi': np.percentile(b, 97.5),
                 'within_dcic_pair_mean': c.mean(), 'within_dcic_pair_lo': np.percentile(c, 2.5),
                 'within_dcic_pair_hi': np.percentile(c, 97.5),
                 'cross_reference_observed': cross[k],
                 'cross_below_within_beta_lo': bool(cross[k] < np.percentile(b, 2.5)),
                 'cross_below_within_dcic_lo': bool(cross[k] < np.percentile(c, 2.5)),
                 'random_expected': k * k / N, 'rounds': ROUNDS})
pd.DataFrame(rows).to_csv('gigascience_upload_work/within_reference_pair_stability.csv', index=False)
pd.DataFrame(rows).to_csv(SUP / 'S17_within_reference_topk_stability.csv', index=False)
print('  -> S17 regenerated under the independent-pair caliber (%d rounds)' % ROUNDS)
print('  within-beta Spearman distribution: %.4f [%.4f, %.4f]; within-dcic: %.4f [%.4f, %.4f]' % (
    res['beta']['rho'].mean(), np.percentile(res['beta']['rho'], 2.5), np.percentile(res['beta']['rho'], 97.5),
    res['dcic']['rho'].mean(), np.percentile(res['dcic']['rho'], 2.5), np.percentile(res['dcic']['rho'], 97.5)))

# ================= B. 2×2 crossed bootstrap =================
print()
print('=== B. crossed (drug x cell) bootstrap for the 2x2 model comparison (%d rounds) ===' % ROUNDS)
B2 = 2000
draws = {c: np.empty(B2) for c in ['b_b', 'b_d', 'd_b', 'd_d']}
D_beta = np.empty(B2); D_dcic = np.empty(B2); I = np.empty(B2)
for b in range(B2):
    m = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
    pick = rng.integers(0, N, N)
    for c in COLS.values():
        num = sums[c][pick].dot(m); den = ncnt[pick].dot(m)
        ok = den > 0
        draws[{'beta_trained__beta2020': 'b_b', 'beta_trained__dcic2021': 'b_d',
               'dcic_trained__beta2020': 'd_b', 'dcic_trained__dcic2021': 'd_d'}[c]][b] = float((num[ok] / den[ok]).mean())
    D_beta[b] = draws['b_b'][b] - draws['d_b'][b]
    D_dcic[b] = draws['b_d'][b] - draws['d_d'][b]
    I[b] = D_beta[b] - D_dcic[b]

obsD_b = 0.3680 - 0.1705; obsD_d = 0.0724 - 0.1534
tab = []
for name, v, point in [('D_beta (beta ref)', D_beta, obsD_b),
                       ('D_dcic (dcic ref)', D_dcic, obsD_d),
                       ('interaction I', I, obsD_b - obsD_d)]:
    lo, hi = np.percentile(v, [2.5, 97.5])
    tab.append({'quantity': name, 'point_estimate_recomputed': point, 'mean_boot': v.mean(),
                'ci_low': lo, 'ci_high': hi, 'sd': v.std(ddof=1),
                'fraction_positive': float((v > 0).mean()), 'fraction_negative': float((v < 0).mean()), 'rounds': B2})
    print('  %-20s point %+.4f  crossed 95%% CI [%+.4f, %+.4f]  SD %.4f  P(>0)=%.4f P(<0)=%.4f' % (
        name, point, lo, hi, v.std(ddof=1), (v > 0).mean(), (v < 0).mean()))
both = int(((D_beta > 0) & (D_dcic < 0)).sum())
print('  * reversal holds (D_beta>0 and D_dcic<0): %d / %d = %.4f' % (both, B2, both / B2))
for c, k in [('b_b', 'beta-trained→beta'), ('b_d', 'beta-trained→dcic'),
             ('d_b', 'dcic-trained→beta'), ('d_d', 'dcic-trained→dcic')]:
    print('  %-22s mean %.4f  CI [%.4f, %.4f]' % (k, draws[c].mean(), *np.percentile(draws[c], [2.5, 97.5])))
pd.DataFrame(tab).to_csv('gigascience_upload_work/crossed_bootstrap_2x2.csv', index=False)
pd.DataFrame(tab).to_csv(SUP / 'S20_lincs_crossed_2x2_model_comparison.csv', index=False)
print('  -> S20 written')
