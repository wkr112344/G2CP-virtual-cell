"""J_expand_existing_signature_panel.py
Based on the existing ranking columns of supplementary/TableS20_absolute_topk_screening_rank.csv,
systematically compute the rank stability of each signature/disease panel under the two references beta(A) vs dcic(B).
"""
import csv, os
import numpy as np
from scipy.stats import spearmanr, kendalltau

PKG = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/supplementary'
RES = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
os.makedirs(RES, exist_ok=True)

rows = list(csv.DictReader(open(f'{PKG}/TableS20_absolute_topk_screening_rank.csv', encoding='utf-8-sig')))
cols = list(rows[0].keys())
rank_cols = [c for c in cols if c.startswith('rank_') and (c.endswith('_A') or c.endswith('_B'))]
bases = {}
for c in rank_cols:
    base = c[:-2]
    bases.setdefault(base, {})[c[-1]] = c
# keep only bases that have both A and B
pairs = {b: v for b, v in bases.items() if 'A' in v and 'B' in v}

N = len(rows)
def topk_set(col, k):
    vals = sorted(((float(r[col]), i) for i, r in enumerate(rows)), key=lambda x: x[0])
    return set(i for _, i in vals[:k])

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0

out = []
for base, ab in pairs.items():
    cA, cB = ab['A'], ab['B']
    rA = np.array([float(r[cA]) for r in rows])
    rB = np.array([float(r[cB]) for r in rows])
    sp = spearmanr(rA, rB).statistic
    kt = kendalltau(rA, rB).statistic
    abs_shift = np.abs(rA - rB)
    # top-k jaccard
    ks = [10, 20, 50, 100, max(1, N // 10)]
    jac = {k: jaccard(topk_set(cA, k), topk_set(cB, k)) for k in ks}
    # top decile -> bottom half
    orderA = np.argsort(rA)
    top_decile = set(orderA[: max(1, N // 10)].tolist())
    bottom_half = set(np.argsort(rB)[N // 2:].tolist())
    td2bh = len(top_decile & bottom_half) / len(top_decile) if top_decile else 0.0
    out.append({
        'panel': base,
        'n_drugs': N,
        'spearman_rank': round(sp, 4),
        'kendall_tau': round(kt, 4),
        'top10_jaccard': round(jac[10], 4),
        'top20_jaccard': round(jac[20], 4),
        'top50_jaccard': round(jac[50], 4),
        'top100_jaccard': round(jac[100], 4),
        'top10pct_jaccard': round(jac[ks[-1]], 4),
        'top_decile_to_bottom_half': round(td2bh, 4),
        'median_abs_rank_shift': round(float(np.median(abs_shift)), 2),
        'mean_abs_rank_shift': round(float(np.mean(abs_shift)), 2),
    })

with open(f'{RES}/J_existing_signature_panel_metrics.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print('wrote J_existing_signature_panel_metrics.csv  panels=', len(out))
for o in out:
    print('  %-55s sp=%.3f kT=%.3f top10j=%.3f meanShift=%.1f' % (o['panel'][:55], o['spearman_rank'], o['kendall_tau'], o['top10_jaccard'], o['mean_abs_rank_shift']))
