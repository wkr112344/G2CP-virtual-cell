"""
B_extract_sevenmodel_beta_dcic_boot.py

Extract the beta2020 vs dcic2021 comparison statistics for the seven models from 220_dpb_crossversion.json.

Data reality (verified on disk):
- 220_dpb_crossversion.json embeds per_model[*].boot.{beta2020,dcic2021}.ci95 / frac_lt_0 / n_boot
  -- these are the bootstrap results for the "SDST (each predictor's own reference) - X" contrast.
- the top-level boot_vectors is only a pointer (file/sha256/bytes); the real vectors live in npz/dpb_crossversion_boot.npz,
  which does not exist in the current working tree => an exact paired bootstrap CI for beta - dcic cannot be computed.
- this script therefore:
    * gives per-model PCC_beta2020 / PCC_dcic2021 / delta_beta_dcic (= the beta - dcic point estimate);
    * propagates variance from the two-arm CIs to give an APPROXIMATE CI for delta_beta_dcic (assuming independence, conservative);
    * also lists the REAL bootstrap CIs and frac_lt_0 for the SDST - dcic two arms (a significance proxy);
    * if dpb_crossversion_boot.npz is recovered later, this can be upgraded to an exact paired CI (see the placeholder B_sevenmodel_permutation.py).

Output:
  results/B_sevenmodel_beta_dcic_ci.csv
  figures/FigS_sevenmodel_beta_dcic_forest.png
"""
import json, csv, os, math
import numpy as np

ROOT = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5'
JSON = 'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37/_scratch/v31_mvpa/220_dpb_crossversion.json'
OUT_CSV = os.path.join(ROOT, 'results', 'B_sevenmodel_beta_dcic_ci.csv')
OUT_PNG = os.path.join(ROOT, 'figures', 'FigS_sevenmodel_beta_dcic_forest.png')

Z = 1.959963984540054  # 95%

def ci_to_se(ci):
    lo, hi = ci
    return (hi - lo) / (2 * Z)

d = json.load(open(JSON, encoding='utf-8'))
pm = d['per_model']

rows = []
for name, m in pm.items():
    pcc = m['drug_pcc_mean']               # sdst / beta2020 / dcic2021
    pcc_beta = pcc['beta2020']
    pcc_dcic = pcc['dcic2021']
    pcc_sdst = pcc['sdst']
    delta_bd = pcc_beta - pcc_dcic        # beta - dcic point estimate

    boot_b = m['boot']['beta2020']        # SDST - beta2020
    boot_d = m['boot']['dcic2021']        # SDST - dcic2021
    ci_b = boot_b['ci95']; ci_d = boot_d['ci95']
    se_b = ci_to_se(ci_b); se_d = ci_to_se(ci_d)
    # approximate CI: delta_beta_dcic = (SDST-dcic) - (SDST-beta), assuming independence (conservative)
    se_delta = math.sqrt(se_b**2 + se_d**2)
    lo_a = delta_bd - Z * se_delta
    hi_a = delta_bd + Z * se_delta

    rows.append({
        'predictor': name,
        'PCC_beta': round(pcc_beta, 4),
        'PCC_dcic': round(pcc_dcic, 4),
        'delta_beta_dcic': round(delta_bd, 4),
        'ci95_low': round(lo_a, 4),
        'ci95_high': round(hi_a, 4),
        'perm_p_proxy': boot_d['frac_lt_0'],
        'n_boot': boot_d['n_boot'],
        # ---- supplementary (real bootstrap quantities, from the json) ----
        'PCC_sdst': round(pcc_sdst, 4),
        'ci95_low_sdst_dcic': round(ci_d[0], 4),
        'ci95_high_sdst_dcic': round(ci_d[1], 4),
        'frac_lt_0_sdst_dcic': boot_d['frac_lt_0'],
        'frac_lt_0_sdst_beta': boot_b['frac_lt_0'],
        'n_drugs': m['n_drugs'],
        'note': 'ci95=approx(var propagation, indep-assumed); exact paired CI needs dpb_crossversion_boot.npz',
    })

# sort: delta descending
rows.sort(key=lambda r: r['delta_beta_dcic'], reverse=True)

core_cols = ['predictor','PCC_beta','PCC_dcic','delta_beta_dcic','ci95_low','ci95_high','perm_p_proxy','n_boot']
supp_cols = ['PCC_sdst','ci95_low_sdst_dcic','ci95_high_sdst_dcic','frac_lt_0_sdst_dcic','frac_lt_0_sdst_beta','n_drugs','note']
with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=core_cols + supp_cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print('wrote', OUT_CSV)
print('\n=== beta-dcic delta summary (answering: is 0.011-0.097 noise?) ===')
deltas = [r['delta_beta_dcic'] for r in rows]
print('n_models =', len(rows))
print('delta_beta_dcic range = [%.4f, %.4f]' % (min(deltas), max(deltas)))
print('all ci95_low > 0 ?', all(r['ci95_low'] > 0 for r in rows))
print('all perm_p_proxy (frac_lt_0 sdst-dcic) == 0 ?', all(r['perm_p_proxy'] == 0 for r in rows))
for r in rows:
    print('  %-10s beta=%.4f dcic=%.4f delta=%.4f [%.4f,%.4f]'
          % (r['predictor'], r['PCC_beta'], r['PCC_dcic'], r['delta_beta_dcic'], r['ci95_low'], r['ci95_high']))

# ---- forest plot ----
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [r['predictor'] for r in rows][::-1]
    pts = [r['delta_beta_dcic'] for r in rows][::-1]
    err = [[r['delta_beta_dcic']-r['ci95_low'] for r in rows][::-1],
           [r['ci95_high']-r['delta_beta_dcic'] for r in rows][::-1]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(labels))
    ax.errorbar(pts, y, xerr=err, fmt='o', color='#1f77b4', ecolor='#888', capsize=3)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('ΔPCC = PCC(beta2020) − PCC(dcic2021)   (approx 95% CI)')
    ax.set_title('Seven-model beta2020 vs dcic2021 (reference-product effect)')
    ax.grid(axis='x', ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print('\nwrote', OUT_PNG)
except Exception as e:
    print('\n(forest plot skipped:', e, ')')
