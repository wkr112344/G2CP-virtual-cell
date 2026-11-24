# -*- coding: utf-8 -*-
"""Canonical generator of Supplementary Table S21 (snapshot name: S23_lincs_crossed_bootstrap.csv).

Crossed (drug x cell) bootstrap for the raw and the gene-mean-removed scores, plus the
per-matrix gene-mean baseline. Seed 20260930, 2,000 rounds.

Per round: cells are resampled with replacement into a count vector m, drugs are resampled
with replacement into an index list, each drug's score is the m-weighted mean of its
per-(drug, cell) Pearson values, and the statistic is the mean over drugs. The same pair of
draws (m, pick) is applied to every arm/reference combination and to both analyses, so the
four cells of the 2 x 2 design stay paired within a replicate.

Expected output (all 21 rows reproduce to 0 difference against
zenodo/supplementary/S23_lincs_crossed_bootstrap.csv):

  interaction I (D_beta - D_dcic), raw      point 0.278456  95% CI 0.264558-0.288877
  interaction I (D_beta - D_dcic), removed  point 0.196906  95% CI 0.184872-0.207305

The manuscript reports 0.2785 (95% CI 0.2646-0.2889) before gene-mean removal and 0.1969
(0.1849-0.2073) after removal; those are the two rows above, rounded.

Do NOT use `_patch_A_B.py` section B for S21. That driver is a separate implementation of
the same estimator with the same nominal seed but a different RNG consumption order (its
state has already advanced through section A's 4,000 row-resamples), so it returns
0.2657-0.2887. It generated the earlier crossed-bootstrap scratch table
(crossed_bootstrap_2x2.csv / S20_lincs_crossed_2x2_model_comparison.csv), not S21.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]           # repository root
SD = ROOT / 'zenodo' / 'source_data'
REL = SD / 'lincs_reference_matrices'
FROZEN = SD / 'frozen_prediction_matrices'
SHIPPED = ROOT / 'zenodo' / 'supplementary' / 'frozen_multimetric_perrow_scores.csv'
OUT = ROOT / 'reproduction_outputs' / 'supplementary' / 'S21_lincs_crossed_bootstrap.csv'
ROUNDS, SEED = 2000, 20260930

met = pd.read_csv(FROZEN / 'frozen_perrow_three_metrics.csv')
Pb = np.asarray(np.load(FROZEN / 'frozen_pred_matrix_beta_trained_11275x978.npy'), np.float64)
Pd = np.asarray(np.load(FROZEN / 'frozen_pred_matrix_dcic_trained_11275x978.npy'), np.float64)
Yb = np.asarray(np.load(REL / 'Y_beta2020_11275x978.npy'), np.float64)
Yd = np.asarray(np.load(REL / 'Y_dcic2021_11275x978.npy'), np.float64)

drug, cell = met['drug'].to_numpy(), met['cell'].to_numpy()
udrug, di = np.unique(drug, return_inverse=True)
ucell, ci = np.unique(cell, return_inverse=True)
N, C = len(udrug), len(ucell)
print('drugs N=%d  cells C=%d  rows=%d' % (N, C, len(drug)))


def lodo_mean(X):
    """Gene means computed across all response records of the other drugs (leave-one-drug-out)."""
    tot = X.sum(0)
    cnt = np.bincount(di, minlength=N)
    dsum = np.zeros((N, X.shape[1]))
    np.add.at(dsum, di, X)
    return (tot[None, :] - dsum[di]) / (len(drug) - cnt[di])[:, None]


def row_pearson(X, Y):
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    den = np.sqrt((Xc ** 2).sum(1) * (Yc ** 2).sum(1))
    return np.where(den > 0, (Xc * Yc).sum(1) / np.where(den > 0, den, 1), np.nan)


PRED = {'b': Pb, 'd': Pd}
REF = {'b': Yb, 'd': Yd}
PRED_R = {k: v - lodo_mean(v) for k, v in PRED.items()}
REF_R = {k: v - lodo_mean(v) for k, v in REF.items()}

sc = pd.read_csv(SHIPPED)
assert len(sc) == len(drug) and (sc.drug.to_numpy() == drug).all(), 'CSV row order differs'
RAW_COL = {('b', 'b'): 'beta_trained__beta2020', ('b', 'd'): 'beta_trained__dcic2021',
           ('d', 'b'): 'dcic_trained__beta2020', ('d', 'd'): 'dcic_trained__dcic2021'}

percell = {}
for (arm, ref), col in RAW_COL.items():
    percell[(arm + ref, 'raw')] = sc[col].to_numpy(float)
for arm in ('b', 'd'):
    for ref in ('b', 'd'):
        percell[(arm + ref, 'removed')] = row_pearson(PRED_R[arm], REF_R[ref])
percell[('base_b', 'raw')] = row_pearson(lodo_mean(Yb), Yb)
percell[('base_d', 'raw')] = row_pearson(lodo_mean(Yd), Yd)

ncnt = np.zeros((N, C))
np.add.at(ncnt, (di, ci), 1.0)
assert ncnt.sum() == len(drug)
sums = {k: np.zeros((N, C)) for k in percell}
for k, arr in percell.items():
    np.add.at(sums[k], (di, ci), np.nan_to_num(arr))

rng = np.random.default_rng(SEED)
draws = {k: np.empty(ROUNDS) for k in percell}
for b in range(ROUNDS):
    m = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
    pick = rng.integers(0, N, N)
    for k in percell:
        den = ncnt[pick].dot(m)
        ok = den > 0
        draws[k][b] = float((sums[k][pick].dot(m)[ok] / den[ok]).mean())
print('bootstrap done: %d rounds' % ROUNDS)

rows = []


def add(label, arm, analysis, v, point, ref=''):
    lo, hi = np.percentile(v, [2.5, 97.5])
    rows.append(dict(quantity=label, arm=arm, reference=ref, analysis=analysis, point_estimate=point,
                     boot_mean=float(v.mean()), ci_low=float(lo), ci_high=float(hi),
                     sd=float(v.std(ddof=1)), fraction_positive=float((v > 0).mean()),
                     fraction_negative=float((v < 0).mean()), rounds=ROUNDS))
    print('  %-24s %-13s %-10s point %+.4f  CI [%+.4f, %+.4f]  P(>0)=%.3f'
          % (label, arm, analysis, point, lo, hi, (v > 0).mean()))


def observed(t):
    out = {}
    for k in ('bb', 'bd', 'db', 'dd'):
        v = percell[(k, t)]
        out[k] = float(pd.DataFrame({'drug': drug, 'r': v}).groupby('drug')['r'].mean().mean())
    return out


for t, lab in (('raw', 'raw'), ('removed', 'removed')):
    bb, bd = draws[('bb', t)], draws[('bd', t)]
    db, dd = draws[('db', t)], draws[('dd', t)]
    pt = observed(t)
    print('\n[%s]' % lab)
    add('score', 'beta-trained', lab, bb, pt['bb'], ref='beta2020')
    add('score', 'beta-trained', lab, bd, pt['bd'], ref='dcic2021')
    add('score', 'dcic-trained', lab, db, pt['db'], ref='beta2020')
    add('score', 'dcic-trained', lab, dd, pt['dd'], ref='dcic2021')
    add('delta (beta2020-dcic2021)', 'beta-trained', lab, bb - bd, pt['bb'] - pt['bd'], ref='both')
    add('delta (beta2020-dcic2021)', 'dcic-trained', lab, db - dd, pt['db'] - pt['dd'], ref='both')
    D_beta, D_dcic = bb - db, bd - dd
    add('D_beta (beta ref)', '-', lab, D_beta, pt['bb'] - pt['db'], ref='beta2020')
    add('D_dcic (dcic ref)', '-', lab, D_dcic, pt['bd'] - pt['dd'], ref='dcic2021')
    add('interaction I (D_beta - D_dcic)', '-', lab, D_beta - D_dcic,
        (pt['bb'] - pt['db']) - (pt['bd'] - pt['dd']), ref='both')
    flip = int(((D_beta > 0) & (D_dcic < 0)).sum())
    print('      reversal kept (D_beta>0 & D_dcic<0): %d/%d = %.4f' % (flip, ROUNDS, flip / ROUNDS))

print('\n[gene-mean baseline]')
for key, ref in (('base_b', 'beta2020'), ('base_d', 'dcic2021')):
    pt = float(pd.DataFrame({'drug': drug, 'r': percell[(key, 'raw')]})
               .groupby('drug')['r'].mean().mean())
    add('gene-mean baseline', '-', 'raw', draws[(key, 'raw')], pt, ref=ref)

tab = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
tab.to_csv(OUT, index=False)
print('\nwrote %s (%d rows)' % (OUT, len(tab)))

print('\n=== check against the archived table ===')
ARCH = ROOT / 'zenodo' / 'supplementary' / 'S23_lincs_crossed_bootstrap.csv'
if ARCH.exists():
    a = pd.read_csv(ARCH)
    key = ['quantity', 'arm', 'reference', 'analysis']
    m = tab.merge(a, on=key, suffixes=('_new', '_arc'))
    cols = ['point_estimate', 'boot_mean', 'ci_low', 'ci_high', 'sd']
    diffs = {c: float(np.nanmax(np.abs(m[c + '_new'] - m[c + '_arc']))) for c in cols}
    print('  rows compared: %d   max abs diff: %s' % (len(m), {k: '%.2e' % v for k, v in diffs.items()}))
    ok = len(m) == len(a) and all(v < 1e-12 for v in diffs.values())
    print('  PASS (agreement at float64 round-off)' if ok else '  MISMATCH')
