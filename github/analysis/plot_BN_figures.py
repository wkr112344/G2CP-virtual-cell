# -*- coding: utf-8 -*-
"""Plot every B-N experiment from the plot-ready CSVs in results/for_figures/.
Outputs PNGs into results/for_figures/figures/.

Run with:  C:/Users/wkr20/miniconda3/envs/dpb311/python.exe plot_BN_figures.py
(matplotlib lives in the dpb311 env, not the default python3)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5'
FF = os.path.join(ROOT, 'results', 'for_figures')
FIG = os.path.join(FF, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})
C_BETA = '#2c7fb8'
C_DCIC = '#d95f02'
C_DELTA = '#1b9e77'


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved', name)


# ---------------------------------------------------------------- B: seven models
def fig_B():
    df = pd.read_csv(os.path.join(FF, 'B_sevenmodel.csv')).sort_values('delta')
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    h = 0.38
    ax.barh(y + h / 2, df['pcc_beta'], height=h, color=C_BETA, label='PCC vs beta2020')
    ax.barh(y - h / 2, df['pcc_dcic'], height=h, color=C_DCIC, label='PCC vs dcic2021')
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.errorbar(r['delta'], yi, xerr=[[r['delta'] - r['ci_low']], [r['ci_high'] - r['delta']]],
                    fmt='none', ecolor='k', capsize=3, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df['model'])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('PCC')
    ax.set_title('B. Seven models: reference-swap PCC (black bar = delta 95% CI)')
    ax.legend(loc='lower right')
    save(fig, 'B_sevenmodel.png')


# ---------------------------------------------------------------- C: per-drug stability
def fig_C():
    df = pd.read_csv(os.path.join(FF, 'C_perdrug_stability.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    sc = ax.scatter(df['pcc_beta'], df['pcc_dcic'], s=12, c=df['delta'],
                   cmap='RdBu_r', vmin=-0.6, vmax=0.6, alpha=0.7, edgecolors='none')
    lim = [-0.2, 1.0]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('PCC vs beta2020')
    ax.set_ylabel('PCC vs dcic2021')
    ax.set_title('C1. Paired per-drug PCC (color = delta)')
    fig.colorbar(sc, ax=ax, label='delta (beta - dcic)')
    ax = axes[1]
    ax.scatter(df['rank_beta'], df['rank_dcic'], s=12, alpha=0.5, color=C_BETA)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('rank by beta2020 PCC')
    ax.set_ylabel('rank by dcic2021 PCC')
    ax.set_title('C2. Rank-rank (top-left = best)')
    save(fig, 'C_perdrug_stability.png')


# ---------------------------------------------------------------- D: G2CP honest two-condition
def fig_D():
    df = pd.read_csv(os.path.join(FF, 'D_g2cp_HONEST_not_true2x2.csv'))
    metrics = df['metric'].tolist()
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - w / 2, df['PCC_beta2020'], w,
           yerr=[df['PCC_beta2020'] - df['CI_low_beta2020'], df['CI_high_beta2020'] - df['PCC_beta2020']],
           color=C_BETA, label='beta2020', capsize=3)
    ax.bar(x + w / 2, df['PCC_dcic2021'], w,
           yerr=[df['PCC_dcic2021'] - df['CI_low_dcic2021'], df['CI_high_dcic2021'] - df['PCC_dcic2021']],
           color=C_DCIC, label='dcic2021', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('PCC / recovery')
    ax.set_title('D. G2CP fixed-output comparison  (NOT a train x eval 2x2)')
    ax.legend()
    save(fig, 'D_g2cp_honest.png')


# ---------------------------------------------------------------- E: novelty tiers
def fig_E():
    df = pd.read_csv(os.path.join(FF, 'E_novelty_tiers.csv')).sort_values('tanimoto_threshold')
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df['tanimoto_threshold'], df['delta_mean'], 'o-', color=C_DELTA, label='mean delta')
    ax.fill_between(df['tanimoto_threshold'], df['ci_low'], df['ci_high'],
                    color=C_DELTA, alpha=0.2, label='95% CI')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('max Tanimoto to training set')
    ax.set_ylabel('delta (beta - dcic) PCC')
    ax.set_title('E. Novelty stratification')
    ax2 = ax.twinx()
    ax2.plot(df['tanimoto_threshold'], df['prop_positive'], 's--', color='gray', label='prop positive')
    ax2.set_ylabel('prop positive', color='gray')
    ax2.tick_params(axis='y', colors='gray')
    ax.legend(loc='upper left')
    save(fig, 'E_novelty_tiers.png')


# ---------------------------------------------------------------- F: within-product reproducibility
def fig_F():
    df = pd.read_csv(os.path.join(FF, 'F_within_product_repro.csv'))
    v = df['pearson'].dropna().values
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    s = np.sort(v)
    cdf = np.arange(1, len(s) + 1) / len(s)
    ax.step(s, cdf, where='post', color=C_BETA)
    ax.set_xlabel('within-product mean pairwise Pearson')
    ax.set_ylabel('ECDF')
    ax.set_title('F1. ECDF (n=%d)' % len(v))
    ax = axes[1]
    ax.hist(v, bins=40, color=C_DCIC, alpha=0.8)
    ax.set_xlabel('within-product mean pairwise Pearson')
    ax.set_ylabel('count')
    ax.set_title('F2. Histogram')
    save(fig, 'F_within_product.png')


# ---------------------------------------------------------------- G/H: DepMap interaction
def fig_GH():
    g = pd.read_csv(os.path.join(FF, 'G_depmap_2x2_5seed.csv'))
    h = pd.read_csv(os.path.join(FF, 'H_depmap_interaction_bootstrap5000.csv'))
    med = float(np.median(h['interaction']))
    lo, hi = np.percentile(h['interaction'], [2.5, 97.5])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ys = np.arange(len(g))
    ax.scatter(g['interaction'], ys, color=C_BETA, s=40, zorder=3, label='per-seed point')
    ax.errorbar(med, len(g) + 0.5, xerr=[[med - lo], [hi - med]], fmt='s', color='k',
                capsize=5, label='hierarchical bootstrap 95% CI')
    ax.axvline(0, color='r', lw=0.8, ls='--')
    ax.set_yticks(list(ys) + [len(g) + 0.5])
    ax.set_yticklabels([f"seed {int(s)}" for s in g['seed']] + ['bootstrap'])
    ax.set_xlabel('interaction = (CC-CH)-(HC-HH)')
    ax.set_title('G/H. DepMap 2x2 interaction  (predictor = VAE-DeepDEP)')
    ax.legend()
    ax = axes[1]
    ax.hist(h['interaction'], bins=40, color=C_DELTA, alpha=0.85)
    ax.axvline(med, color='k', lw=1, label=f'median {med:.4f}')
    ax.axvspan(lo, hi, color='gray', alpha=0.25, label='95% CI')
    ax.set_xlabel('interaction replicate')
    ax.set_ylabel('count')
    ax.set_title('H. 5000 hierarchical bootstrap replicates')
    ax.legend()
    save(fig, 'GH_depmap_interaction.png')


# ---------------------------------------------------------------- I: product-level decomposition
def fig_I():
    df = pd.read_csv(os.path.join(FF, 'I_perproduct_concordance.csv'))
    v = df['concordance'].dropna().values
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(v, bins=80, color=C_BETA, alpha=0.85)
    ax.axvline(np.median(v), color='k', lw=1, label=f'median {np.median(v):.3f}')
    ax.axvline(0, color='r', ls='--', lw=0.8)
    ax.set_xlabel('beta vs dcic Pearson per (cell,drug)')
    ax.set_ylabel('count')
    ax.set_title(f'I. Product-level decomposition (n={len(v):,})')
    ax.legend()
    save(fig, 'I_perproduct.png')


# ---------------------------------------------------------------- J: disease panels
def fig_J():
    df = pd.read_csv(os.path.join(FF, 'J_disease_panels.csv')).sort_values('spearman')
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(y, df['spearman'], color=C_BETA, label='spearman')
    ax.set_yticks(y)
    ax.set_yticklabels(df['panel'], fontsize=8)
    ax.set_xlabel('spearman rank corr (beta vs dcic signature)')
    ax.set_title('J. Disease panels')
    ax2 = ax.twinx()
    ax2.scatter(df['top10_jaccard'], y, color=C_DCIC, marker='D', s=40, label='top10 jaccard')
    ax2.set_ylabel('top10 jaccard', color=C_DCIC)
    ax2.tick_params(axis='y', colors=C_DCIC)
    ax.legend(loc='lower right')
    save(fig, 'J_disease_panels.png')


# ---------------------------------------------------------------- K: hypergeometric nulls
def fig_K():
    df = pd.read_csv(os.path.join(FF, 'K_hypergeom_nulls.csv'))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    lp = -np.log10(df['p_value'].clip(lower=1e-300))
    sc = ax.scatter(df['expected_overlap'], df['observed_overlap'], c=lp, cmap='viridis', s=45)
    mx = max(df['expected_overlap'].max(), df['observed_overlap'].max()) * 1.1
    ax.plot([0, mx], [0, mx], 'k--', lw=1, label='expected = observed')
    ax.set_xlabel('expected overlap')
    ax.set_ylabel('observed overlap')
    ax.set_title('K. Hypergeometric nulls (color = -log10 p)')
    fig.colorbar(sc, label='-log10 p')
    ax.legend()
    save(fig, 'K_hypergeom.png')


# ---------------------------------------------------------------- L: heterogeneity
def fig_L():
    df = pd.read_csv(os.path.join(FF, 'L_heterogeneity_272cell.csv'))
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111)
    sc = ax.scatter(df['product_concordance'], df['delta_pearson'], s=20, c=df['delta_pearson'],
                   cmap='RdBu_r', vmin=-0.05, vmax=0.12, alpha=0.85)
    divider = make_axes_locatable(ax)
    axh = divider.append_axes('top', 1.2, pad=0.1, sharex=ax)
    axh.hist(df['product_concordance'], bins=20, color=C_BETA, alpha=0.7)
    axh.axis('off')
    axr = divider.append_axes('right', 1.2, pad=0.1, sharey=ax)
    axr.hist(df['delta_pearson'], bins=20, color=C_DCIC, alpha=0.7, orientation='horizontal')
    axr.axis('off')
    ax.set_xlabel('product concordance (mean of 2 reference PCCs)')
    ax.set_ylabel('delta_pearson')
    fig.suptitle('L. Heterogeneity across 272 cell lines', y=1.02)
    save(fig, 'L_heterogeneity.png')


# ---------------------------------------------------------------- M: negative controls
def fig_M():
    df = pd.read_csv(os.path.join(FF, 'M_negative_controls.csv'))
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.errorbar(df['null_mean'], y,
                xerr=[df['null_mean'] - df['null_CI_low'], df['null_CI_high'] - df['null_mean']],
                fmt='o', color='gray', capsize=3, label='null mean +/- 1.96 SD')
    ax.scatter(df['observed_mean_abs_rank_shift'], y, color=C_DCIC, zorder=5, s=35, label='observed')
    ax.set_yticks(y)
    ax.set_yticklabels(df['panel'], fontsize=8)
    ax.set_xlabel('mean abs rank shift')
    ax.set_title('M. Negative controls (observed vs null)')
    ax.legend()
    save(fig, 'M_negative_controls.png')


# ---------------------------------------------------------------- N: crossmodality
def fig_N():
    df = pd.read_csv(os.path.join(FF, 'N_crossmodality.csv')).sort_values('pearson')
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(y, df['pearson'], color=C_BETA)
    ax.set_yticks(y)
    ax.set_yticklabels(df['cell'], fontsize=9)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Pearson (trt_cp mean profile vs trt_sh)')
    ax.set_title('N. Cross-modality (trt_cp vs trt_sh)')
    save(fig, 'N_crossmodality.png')


if __name__ == '__main__':
    print('Plotting B-N figures ->', FIG)
    fig_B()
    fig_C()
    fig_D()
    fig_E()
    fig_F()
    fig_GH()
    fig_I()
    fig_J()
    fig_K()
    fig_L()
    fig_M()
    fig_N()
    print('ALL FIGURES DONE')
