"""K_topk_exact_nulls.py
For each ranking panel in TableS20, compute the significance of the beta(A) vs dcic(B) top-k overlap against a hypergeometric null.
observed overlap = |topA_k ∩ topB_k|；null = Hypergeometric(N, K=k, n=k)。
"""
import csv, os
import numpy as np
from scipy.stats import hypergeom
from math import erf, sqrt

def _norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

PKG = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/supplementary'
RES = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
os.makedirs(RES, exist_ok=True)

rows = list(csv.DictReader(open(f'{PKG}/TableS20_absolute_topk_screening_rank.csv', encoding='utf-8-sig')))
cols = list(rows[0].keys())
rank_cols = [c for c in cols if c.startswith('rank_') and (c.endswith('_A') or c.endswith('_B'))]
bases = {}
for c in rank_cols:
    bases.setdefault(c[:-2], {})[c[-1]] = c
pairs = {b: v for b, v in bases.items() if 'A' in v and 'B' in v}
N = len(rows)

def topk(col, k):
    return set(i for i, v in sorted(enumerate(float(r[col]) for r in rows), key=lambda x: x[1])[:k])

ks = [10, 20, 50, 100]
out = []
for base, ab in pairs.items():
    cA, cB = ab['A'], ab['B']
    for k in ks:
        topA, topB = topk(cA, k), topk(cB, k)
        obs = len(topA & topB)
        rv = hypergeom(N, k, k)
        exp = rv.mean(); var = rv.var()
        z = (obs - exp) / np.sqrt(var) if var > 0 else 0.0
        p = 2 * (1 - _norm_cdf(abs(z)))
        out.append({
            'panel': base, 'k': k, 'n_drugs': N,
            'observed_overlap': obs, 'expected_overlap': round(exp, 3),
            'z_score': round(z, 3), 'p_value': round(p, 6),
        })

with open(f'{RES}/K_topk_hypergeom_nulls.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print('wrote K_topk_hypergeom_nulls.csv  rows=', len(out))
for o in out[:6]:
    print('  ', o['panel'][:40], 'k=', o['k'], 'obs=', o['observed_overlap'], 'exp=%.2f' % o['expected_overlap'], 'z=%.2f' % o['z_score'], 'p=%.4g' % o['p_value'])
