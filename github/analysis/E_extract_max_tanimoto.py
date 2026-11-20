"""E_extract_max_tanimoto.py  (E2)
Using data/g2cp_cache_beta_v2/drug_fps.npy (32039 x 2048 continuous fingerprints), recompute each test drug's
max Tanimoto-to-train, then combine with the per-drug beta(2020)-dcic(2021) delta from TableS22
to run a Tanimoto threshold sensitivity analysis (the "continuous stratification" part of E).

Alignment: drug_fps row order = meta.npz drug_vocab; TableS22.drug matches drug_vocab case-insensitively / literally.
Tanimoto (continuous fingerprints) = (A.B) / (||A||^2 + ||B||^2 - A.B).
"""
import csv, os, json
import numpy as np

WS = 'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37'
PKG = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/supplementary'
RES = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
os.makedirs(RES, exist_ok=True)

fps = np.load(f'{WS}/data/g2cp_cache_beta_v2/drug_fps.npy')          # (32039, 2048) float32
meta = np.load(f'{WS}/data/g2cp_cache_beta_v2/meta.npz', allow_pickle=True)
vocab = list(meta['drug_vocab']); vocab_idx = {d: i for i, d in enumerate(vocab)}
norms2 = np.sum(fps * fps, axis=1)                                    # (32039,)

# per-drug delta from TableS22
t22 = list(csv.DictReader(open(f'{PKG}/TableS22_strict_two_sided_absent_per_drug.csv', encoding='utf-8-sig')))
rows_out = []
for r in t22:
    drug = r['drug']
    if drug not in vocab_idx:
        continue
    i = vocab_idx[drug]
    fp = fps[i]
    dots = fps @ fp
    denom = norms2 + norms2[i] - dots
    T = dots / np.where(denom > 0, denom, 1e-12)
    T[i] = -1.0  # exclude self
    maxT = float(T.max())
    try:
        dA = float(r['pcc_A_beta2020_978']); dB = float(r['pcc_B_dcic2021_978'])
    except (KeyError, ValueError):
        dA = float(r['pcc_A_beta2020_12328']); dB = float(r['pcc_B_dcic2021_12328'])
    delta = dA - dB
    rows_out.append({'drug': drug, 'max_tanimoto_to_train': round(maxT, 4),
                     'delta_pcc_beta_minus_dcic': round(delta, 4)})

with open(f'{RES}/E_perdrug_max_tanimoto.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['drug', 'max_tanimoto_to_train', 'delta_pcc_beta_minus_dcic'])
    w.writeheader(); w.writerows(rows_out)
print('wrote E_perdrug_max_tanimoto.csv  n=', len(rows_out))

# threshold sensitivity
thresholds = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]
arr = np.array([[r['max_tanimoto_to_train'], r['delta_pcc_beta_minus_dcic']] for r in rows_out])
mt, dl = arr[:, 0], arr[:, 1]
import scipy.stats as st
out = []
for thr in thresholds:
    mask = mt < thr
    sub = dl[mask]
    if len(sub) >= 2:
        ci = st.t.interval(0.95, len(sub) - 1, loc=sub.mean(), scale=st.sem(sub))
    else:
        ci = (float('nan'), float('nan'))
    out.append({'threshold': f'<{thr:.2f}', 'n_drugs': int(mask.sum()),
                'mean_delta': round(float(sub.mean()), 4),
                'median_delta': round(float(np.median(sub)), 4),
                'ci95_low': round(float(ci[0]), 4), 'ci95_high': round(float(ci[1]), 4),
                'prop_positive': round(float(np.mean(sub > 0)), 4)})
with open(f'{RES}/E_tanimoto_threshold_robustness.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print('wrote E_tanimoto_threshold_robustness.csv')
for o in out:
    print('  thr=%s n=%4d meanΔ=%.4f medΔ=%.4f prop+=%s' % (o['threshold'], o['n_drugs'], o['mean_delta'], o['median_delta'], o['prop_positive']))
