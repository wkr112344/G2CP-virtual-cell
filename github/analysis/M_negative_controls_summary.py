"""M_negative_controls_summary.py
A label-permutation negative control on the ranking columns of TableS20:
for each panel, observed = mean|rank_A - rank_B|; null = recompute after shuffling the drug order of column B,
with 1,000 permutations giving an empirical p (whether observed is unusual relative to the null).
"""
import csv, os
import numpy as np

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
rng = np.random.default_rng(20260928)
B = 1000

def mean_abs_delta(rA, rB):
    return np.mean(np.abs(rA - rB))

out = []
for base, ab in pairs.items():
    rA = np.array([float(r[ab['A']]) for r in rows])
    rB = np.array([float(r[ab['B']]) for r in rows])
    obs = mean_abs_delta(rA, rB)
    nulls = np.empty(B)
    for b in range(B):
        perm = rng.permutation(rB)
        nulls[b] = mean_abs_delta(rA, perm)
    p = np.mean(nulls >= obs)
    out.append({
        'panel': base, 'n_drugs': N,
        'observed_mean_abs_rank_shift': round(float(obs), 3),
        'null_mean': round(float(np.mean(nulls)), 3),
        'null_std': round(float(np.std(nulls)), 3),
        'z_score': round(float((obs - np.mean(nulls)) / np.std(nulls)), 3) if np.std(nulls) > 0 else 0.0,
        'empirical_p_ge_observed': round(float(p), 4),
    })

with open(f'{RES}/M_negative_controls.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print('wrote M_negative_controls.csv  panels=', len(out))
for o in out:
    print('  %-55s obs=%.2f null=%.2f z=%.2f p=%.3f' % (o['panel'][:55], o['observed_mean_abs_rank_shift'], o['null_mean'], o['z_score'], o['empirical_p_ge_observed']))
