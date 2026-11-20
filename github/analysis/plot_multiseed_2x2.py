#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_multiseed_2x2.py — figures for the LINCS PRnet controlled 2x2 multi-seed study.
Reads the outputs of PRnet_commonX_train_eval_multiseed.py:
  - multiseed_2x2_summary.csv  (per-seed four cells + interaction)
  - multiseed_bootstrap.npz     (hierarchical_interaction_spearman/pearson, per_seed_interaction)
Writes G/H-style forest plots + bootstrap histograms under for_figures/figures/.

Must be run with a python that has matplotlib (dpb311).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = 'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5'
OUTDIR = os.path.join(ROOT, 'data', 'prnet_commonX')
FIG = os.path.join(ROOT, 'results', 'for_figures', 'figures')
os.makedirs(FIG, exist_ok=True)
TAG = 'multiseed'

plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 11,
                     'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})
C_DELTA = '#1b9e77'


def main():
    sumdf = pd.read_csv(os.path.join(OUTDIR, f'{TAG}_2x2_summary.csv'))
    z = np.load(os.path.join(OUTDIR, f'{TAG}_bootstrap.npz'))
    rep_sp = z['hierarchical_interaction_spearman']
    rep_pe = z['hierarchical_interaction_pearson']
    med = float(np.median(rep_sp)); lo, hi = np.percentile(rep_sp, [2.5, 97.5])

    # ---------- forest plot + bootstrap histogram ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ys = np.arange(len(sumdf))
    ax.scatter(sumdf['interaction_sp'], ys, color='#2c7fb8', s=45, zorder=3, label='per-seed interaction')
    ax.errorbar(med, len(sumdf) + 0.5, xerr=[[med - lo], [hi - med]], fmt='s', color='k',
                capsize=5, label='hierarchical bootstrap 95% CI')
    ax.axvline(0, color='r', lw=0.8, ls='--')
    ax.set_yticks(list(ys) + [len(sumdf) + 0.5])
    ax.set_yticklabels([f"seed {int(s)}" for s in sumdf['seed']] + ['bootstrap'])
    ax.set_xlabel('interaction = (BB-BD)-(DB-DD)  [spearman]')
    ax.set_title('LINCS PRnet 2x2 interaction (multi-seed)')
    ax.legend()

    ax = axes[1]
    ax.hist(rep_sp, bins=40, color=C_DELTA, alpha=0.85)
    ax.axvline(med, color='k', lw=1, label=f'median {med:.4f}')
    ax.axvspan(lo, hi, color='gray', alpha=0.25, label='95% CI')
    ax.set_xlabel('interaction replicate')
    ax.set_ylabel('count')
    ax.set_title('H. 5000 hierarchical bootstrap replicates')
    ax.legend()

    fig.tight_layout()
    p = os.path.join(FIG, 'A_multiseed_2x2_interaction.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', p)

    # ---------- four-cell heatmap (spearman, per seed) ----------
    cells = ['BB_sp', 'BD_sp', 'DB_sp', 'DD_sp']
    labels = ['BB', 'BD', 'DB', 'DD']
    mat = sumdf[cells].values.astype(float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(sumdf))); ax.set_yticklabels([f"seed {int(s)}" for s in sumdf['seed']])
    ax.set_title('Per-seed 2x2 four-cell PCC (spearman)')
    for i in range(mat.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha='center', va='center', fontsize=8,
                    color='white' if abs(mat[i, j]) > 0.5 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p2 = os.path.join(FIG, 'A_multiseed_2x2_fourcell.png')
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', p2)


if __name__ == '__main__':
    main()
