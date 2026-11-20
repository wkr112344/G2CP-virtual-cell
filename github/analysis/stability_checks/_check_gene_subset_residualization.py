"""Verification: whether whole-gene residualization vs gene-subset residualization both give a training x evaluation interaction of -0.00047.

Calibers follow `.../as_supplied/depmap_background_control.py` exactly:
  - the reference matrices first build an observed matrix obs (136x17,393) from common_observed_mask and non-missing values;
  - the four cells share the same mask Mt (the same gene set is used for the four correlations within a cell);
  - residualization: subtract the per-gene mean of its own training arm from the prediction, and that of the training set from the reference;
  - per-cell interaction I(i)=[CC-CH]-[HC-HH] -> median within a seed -> mean over the five seeds.

Output: residual values under the whole-gene definition and each "subset" definition, side by side.
"""
import numpy as np
from pathlib import Path

D = Path('depmap_analysis_ready_20260929')
Yce = np.load(D/'reference_test/Y_CERES_test136.npy').astype(float)
Ych = np.load(D/'reference_test/Y_Chronos_test136.npy').astype(float)
mask = np.load(D/'reference_test/common_observed_mask_test136.npy').astype(bool)
muc = np.load(D/'train_gene_mean/train_genemean_Y_CERES.npy').astype(float)
muh = np.load(D/'train_gene_mean/train_genemean_Y_Chronos.npy').astype(float)

obs = np.isfinite(Yce) & np.isfinite(Ych) & mask
frac = obs.mean(axis=0)


def masked_corr(A, B, M):
    A2 = np.where(M, A, np.nan); B2 = np.where(M, B, np.nan)
    ma = np.nanmean(A2, 1, keepdims=True); mb = np.nanmean(B2, 1, keepdims=True)
    da = np.where(M, A - ma, 0.0); db = np.where(M, B - mb, 0.0)
    num = (da*db).sum(1); den = np.sqrt((da*da).sum(1)*(db*db).sum(1)); n = M.sum(1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0) & (n >= 3))


def interaction_residualized(M):
    res = np.empty((5, 136))
    for s in range(5):
        Pc = np.load(D/f'predictions_test/seed{s}_pred_ceres_arm_test.npy').astype(float)
        Ph = np.load(D/f'predictions_test/seed{s}_pred_chronos_arm_test.npy').astype(float)
        res[s] = (masked_corr(Pc-muc, Yce-muc, M) - masked_corr(Pc-muc, Ych-muh, M)
                  - masked_corr(Ph-muh, Yce-muc, M) + masked_corr(Ph-muh, Ych-muh, M))
    per_seed = [float(np.nanmedian(res[s])) for s in range(5)]
    return float(np.mean(per_seed)), per_seed


if __name__ == '__main__':
    print('observation structure: %d genes observed in all 136 cells; %d genes with missingness (%.1f%%)'
          % ((frac == 1).sum(), (frac < 1).sum(), 100*(frac < 1).mean()))
    print()
    print('%-44s %-14s %s' % ('caliber', 'residual interaction', 'per-seed medians'))
    m, ps = interaction_residualized(obs)
    print('%-44s %+.8f   %s' % ('all genes (original caliber, shared four-cell mask)', m, np.round(ps, 5).tolist()))
    for th, name in [(1.0, 'genes observed in all 136 cells'), (0.99, 'observation rate >=99%'), (0.95, 'observation rate >=95%'),
                     (0.90, 'observation rate >=90%'), (0.50, 'observation rate >=50%')]:
        sel = obs & (frac >= th)[None, :]
        m2, ps2 = interaction_residualized(sel)
        print('%-44s %+.8f   %s   (%d genes)' % (name, m2, np.round(ps2, 5).tolist(), int((frac >= th).sum())))
    print()
    print('Note: only %d of the 17,393 genes (%.1f%%) have cell-level missingness,'
          'so any observation-rate threshold below 100%% is equivalent to "all genes" and the values are necessarily identical.'
          % ((frac < 1).sum(), 100*(frac < 1).mean()))
