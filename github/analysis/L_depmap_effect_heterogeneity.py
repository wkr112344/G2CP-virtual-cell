"""L_depmap_effect_heterogeneity.py
Based on figure_source_data/DepMap_percell_delta_eval.csv (272 cell lines x 27 columns),
characterize the heterogeneity of the reference-product effect in the DeepDEP fixed-output evaluation:
- the distributions of delta_pearson / delta_spearman
- the correlation of delta with product concordance (the mean PCC across the two references)
- the delta distribution of top-k retention
"""
import csv, os
import numpy as np

SRC = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/figure_source_data'
RES = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
os.makedirs(RES, exist_ok=True)

rows = list(csv.DictReader(open(f'{SRC}/DepMap_percell_delta_eval.csv', encoding='utf-8-sig')))
n = len(rows)
def col(name): return np.array([float(r[name]) for r in rows])

dp = col('delta_pearson'); ds = col('delta_spearman')
pc_ceres = col('pearson_vs_ceres'); pc_chronos = col('pearson_vs_chronos')
prod_conc = (pc_ceres + pc_chronos) / 2.0

def desc(x):
    return dict(n=len(x), mean=round(float(np.mean(x)),4), median=round(float(np.median(x)),4),
               std=round(float(np.std(x)),4), min=round(float(np.min(x)),4), max=round(float(np.max(x)),4))

from scipy.stats import pearsonr
corr_dp_prod = pearsonr(dp, prod_conc).statistic
corr_ds_prod = pearsonr(ds, prod_conc).statistic

summary = {
    'n_cell_lines': n,
    'delta_pearson': desc(dp),
    'delta_spearman': desc(ds),
    'corr_delta_pearson_vs_product_concordance': round(float(corr_dp_prod),4),
    'corr_delta_spearman_vs_product_concordance': round(float(corr_ds_prod),4),
    'frac_cells_delta_pearson_gt_0': round(float(np.mean(dp > 0)),4),
    'frac_cells_delta_spearman_gt_0': round(float(np.mean(ds > 0)),4),
}
# per top-k retention delta
for k in [10, 50, 100]:
    rc = col(f'top{k}_retention_vs_ceres'); rk = col(f'top{k}_retention_vs_chronos')
    d = rc - rk
    summary[f'delta_top{k}_retention'] = desc(d)

with open(f'{RES}/L_depmap_effect_modifiers.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'value'])
    for k, v in summary.items():
        w.writerow([k, v if not isinstance(v, dict) else str(v)])
print('wrote L_depmap_effect_modifiers.csv')
for k, v in summary.items():
    print('  ', k, '=', v)
