# -*- coding: utf-8 -*-
"""Export B-N experiment results into plot-ready CSVs with the exact schemas the
user requested. Writes everything into results/for_figures/.

Honesty notes are encoded in column names / a companion README, not faked data:
  - B: CI is approximate variance propagation; permutation_p is a *proxy* placeholder
       (all 0.0); no bootstrap .npz exists for B.
  - C: delta = pcc_beta - pcc_dcic (same convention as B); ranks recomputed.
  - D: NOT a true train x eval 2x2. It is a fixed-output two-condition comparison
       (beta2020 vs dcic2021) across several metrics. Delivered as an honest
       two-condition table, not a 2x2.
  - F: 4000 rows are per-(cell,pert) reproducible *group means* of pairwise Pearson
       (not 4000 raw per-instance correlations). Delivered as cell,pert_id,pearson.
  - G/H: 5 seeds of CC/CH/HC/HH/interaction + 5000 hierarchical-bootstrap interaction
         replicates. Single predictor = VAE-DeepDEP.
  - L: real 272-cell rows reconstructed from the raw per-cell source.
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5'
RES = os.path.join(ROOT, 'results')
SRC = os.path.join(ROOT, 'figure_source_data')
GH_TMP = os.path.join(RES, '__gh_tmp')
OUT = os.path.join(RES, 'for_figures')
os.makedirs(OUT, exist_ok=True)

def w(name, df):
    p = os.path.join(OUT, name)
    df.to_csv(p, index=False)
    print(f'  wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols')

# ---------------------------------------------------------------- B: seven models
print('[B] seven models')
b = pd.read_csv(os.path.join(RES, 'B_sevenmodel_beta_dcic_ci.csv'))
b_out = b[['predictor', 'PCC_beta', 'PCC_dcic', 'delta_beta_dcic',
           'ci95_low', 'ci95_high', 'perm_p_proxy']].copy()
b_out.columns = ['model', 'pcc_beta', 'pcc_dcic', 'delta', 'ci_low', 'ci_high', 'permutation_p']
w('B_sevenmodel.csv', b_out)

# ---------------------------------------------------------------- C: per-drug stability
print('[C] per-drug stability')
# NOTE: the workspace copy results/C_perdrug_refswap.csv is corrupted (header repeated
# 2037x, no data rows). Rebuild from the real prior-pipeline source JSON instead.
cj = json.load(open(os.path.join(RES, '__C_src_tmp.json'), encoding='utf-8-sig'))
pdlist = cj['per_drug']  # 2037 rows of dicts
c = pd.DataFrame(pdlist)
c = c.rename(columns={'drug': 'drug_id',
                      'pcc_A_beta2020': 'pcc_beta',
                      'pcc_B_dcic2021': 'pcc_dcic'})
c['pcc_beta'] = c['pcc_beta'].astype(float)
c['pcc_dcic'] = c['pcc_dcic'].astype(float)
c['delta'] = c['pcc_beta'] - c['pcc_dcic']
# higher pcc = better -> rank 1 is the highest
c['rank_beta'] = c['pcc_beta'].rank(ascending=False, method='first').astype(int)
c['rank_dcic'] = c['pcc_dcic'].rank(ascending=False, method='first').astype(int)
n = len(c)
top10pct = max(1, int(round(n * 0.10)))
c['in_top10_beta'] = c['rank_beta'] <= 10
c['in_top10_dcic'] = c['rank_dcic'] <= 10
c['in_top10pct_beta'] = c['rank_beta'] <= top10pct
c['in_top10pct_dcic'] = c['rank_dcic'] <= top10pct
c_out = c[['drug_id', 'pcc_beta', 'pcc_dcic', 'delta', 'rank_beta', 'rank_dcic',
           'in_top10_beta', 'in_top10_dcic', 'in_top10pct_beta', 'in_top10pct_dcic']].copy()
w('C_perdrug_stability.csv', c_out)
# also restore the corrupted canonical C CSV so the workspace is consistent
_canon = c[['rank_beta', 'drug_id', 'pcc_beta', 'pcc_dcic',
            'in_top10_beta', 'in_top10_dcic', 'in_B_top50']].copy()
_canon = _canon.rename(columns={'rank_beta': 'rank_A', 'in_top10_beta': 'in_A_top10',
                                'in_top10_dcic': 'in_B_top10'})
_canon.to_csv(os.path.join(RES, 'C_perdrug_refswap.csv'), index=False)
print('  restored canonical results/C_perdrug_refswap.csv:', _canon.shape[0], 'rows')

# ---------------------------------------------------------------- D: G2CP (honest, NOT true 2x2)
print('[D] G2CP refswap (honest two-condition, not a true 2x2)')
d = pd.read_csv(os.path.join(RES, 'D_g2cp_refswap_2x2.csv'))
metric_rows = ['pcc', 'dir', 'd1', 'd5', 'd10', 'd20', 'd50']
dd = d[d['metric'].isin(metric_rows)].copy()
d_out = pd.DataFrame({
    'metric': dd['metric'].values,
    'reference_product': np.where(dd['metric'].notna(), 'beta2020', ''),
    'PCC_beta2020': dd['A_beta2020_est'].values,
    'CI_low_beta2020': dd['A_ci_low'].values,
    'CI_high_beta2020': dd['A_ci_high'].values,
    'PCC_dcic2021': dd['B_dcic2021_est'].values,
    'CI_low_dcic2021': dd['B_ci_low'].values,
    'CI_high_dcic2021': dd['B_ci_high'].values,
    'paired_diff': dd['paired_diff_est'].values,
    'paired_diff_CI_low': dd['paired_diff_ci_low'].values,
    'paired_diff_CI_high': dd['paired_diff_ci_high'].values,
})
w('D_g2cp_HONEST_not_true2x2.csv', d_out)

# ---------------------------------------------------------------- E: novelty tiers
print('[E] novelty tiers')
e = pd.read_csv(os.path.join(RES, 'E_tanimoto_threshold_robustness.csv'))
e_out = e.rename(columns={'threshold': 'tanimoto_threshold',
                          'mean_delta': 'delta_mean',
                          'median_delta': 'delta_median',
                          'ci95_low': 'ci_low',
                          'ci95_high': 'ci_high',
                          'prop_positive': 'prop_positive'})
e_out = e_out[['tanimoto_threshold', 'n_drugs', 'delta_mean', 'delta_median',
               'ci_low', 'ci_high', 'prop_positive']]
w('E_novelty_tiers.csv', e_out)

# ---------------------------------------------------------------- F: within-product reproducibility
print('[F] within-product reproducibility (group-mean pairwise Pearson)')
f = pd.read_csv(os.path.join(RES, 'F_within_product_reproducibility.csv'))
f_out = f.rename(columns={'cell': 'cell', 'pert': 'pert_id',
                          'mean_pairwise_pearson': 'pearson'})
f_out = f_out[['cell', 'pert_id', 'pearson', 'n_instances']]
w('F_within_product_repro.csv', f_out)

# ---------------------------------------------------------------- G: DepMap 2x2 (5 seeds)
print('[G] DepMap 2x2 (5 seeds)')
g = pd.read_csv(os.path.join(GH_TMP, 'multiseed_2x2_summary.csv'))
g_out = g[['seed', 'CC', 'CH', 'HC', 'HH', 'interaction']].copy()
g_out['predictor'] = 'VAE-DeepDEP'
g_out = g_out[['seed', 'predictor', 'CC', 'CH', 'HC', 'HH', 'interaction']]
w('G_depmap_2x2_5seed.csv', g_out)

# ---------------------------------------------------------------- H: interaction bootstrap 5000
print('[H] DepMap interaction hierarchical bootstrap (5000 replicates)')
z = np.load(os.path.join(RES, '__gh_boot_tmp.npz'))
rep = z['hierarchical_interaction']
h_out = pd.DataFrame({'replicate': np.arange(len(rep)),
                      'predictor': 'VAE-DeepDEP',
                      'interaction': rep.astype(float)})
w('H_depmap_interaction_bootstrap5000.csv', h_out)

# ---------------------------------------------------------------- I: product-level decomposition
print('[I] product-level (cell,drug) decomposition')
i = pd.read_csv(os.path.join(RES, 'I_perproduct_pearson.csv'))
i_out = i.rename(columns={'pearson_beta_vs_2021': 'concordance'})
i_out = i_out[['cell', 'drug', 'concordance']]
w('I_perproduct_concordance.csv', i_out)

# ---------------------------------------------------------------- J: disease panels
print('[J] disease panels')
j = pd.read_csv(os.path.join(RES, 'J_existing_signature_panel_metrics.csv'))
j_out = j[['panel', 'spearman_rank', 'top10_jaccard', 'n_drugs']].copy()
j_out.columns = ['panel', 'spearman', 'top10_jaccard', 'n_drugs']
w('J_disease_panels.csv', j_out)

# ---------------------------------------------------------------- K: hypergeometric nulls
print('[K] hypergeometric nulls')
k = pd.read_csv(os.path.join(RES, 'K_topk_hypergeom_nulls.csv'))
k_out = k[['panel', 'k', 'observed_overlap', 'expected_overlap', 'p_value']].copy()
w('K_hypergeom_nulls.csv', k_out)

# ---------------------------------------------------------------- L: heterogeneity (272 cells)
print('[L] heterogeneity (272 cells, reconstructed from raw per-cell source)')
l = pd.read_csv(os.path.join(SRC, 'DepMap_percell_delta_eval.csv'), encoding='utf-8-sig')
l['product_concordance'] = (l['pearson_vs_ceres'] + l['pearson_vs_chronos']) / 2.0
l_out = l[['cell_line', 'delta_pearson', 'product_concordance']].copy()
l_out = l_out.rename(columns={'cell_line': 'cell'})
w('L_heterogeneity_272cell.csv', l_out)

# ---------------------------------------------------------------- M: negative controls
print('[M] negative controls')
m = pd.read_csv(os.path.join(RES, 'M_negative_controls.csv'))
m['null_CI_low'] = m['null_mean'] - 1.96 * m['null_std']
m['null_CI_high'] = m['null_mean'] + 1.96 * m['null_std']
m_out = m[['panel', 'observed_mean_abs_rank_shift', 'null_mean',
           'null_CI_low', 'null_CI_high', 'empirical_p_ge_observed']].copy()
w('M_negative_controls.csv', m_out)

# ---------------------------------------------------------------- N: trt_cp vs trt_sh
print('[N] trt_cp vs trt_sh')
n = pd.read_csv(os.path.join(RES, 'N_crossmodality_beta2020_cellprofile.csv'))
n_out = n[['cell', 'pearson_meanprofile_cp_vs_sh']].copy()
n_out = n_out.rename(columns={'pearson_meanprofile_cp_vs_sh': 'pearson'})
w('N_crossmodality.csv', n_out)

print('\nALL DONE ->', OUT)
