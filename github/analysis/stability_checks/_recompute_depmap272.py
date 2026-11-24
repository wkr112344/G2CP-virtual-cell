"""DepMap 272-cell common grid: recomputation of the two-reference agreement, frozen-prediction means, and the paired Δ with its bootstrap CI."""
import pandas as pd, numpy as np, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
V = str(ROOT/'zenodo'/'source_data'/'figure_source_data')+'/'
cer = pd.read_csv(V + 'DepMap_CERES_272x1244.csv', index_col=0)
chr_ = pd.read_csv(V + 'DepMap_Chronos_272x1244.csv', index_col=0)
pred = pd.read_csv(ROOT/'zenodo'/'source_data'/'depmap_raw_artifacts_20260929'/'first_predictor_DeepDEP'/'DeepDEP_predictor_strict.csv', index_col=0)

def cw(A, B):
    out = {}
    for c in A.index:
        if c not in B.index: continue
        a, b = A.loc[c].to_numpy(float), B.loc[c].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0: continue
        out[c] = float(np.corrcoef(a[m], b[m])[0, 1])
    return pd.Series(out)

cons, pc, pcx = cw(cer, chr_), cw(pred, cer), cw(pred, chr_)
common = cons.index.intersection(pc.index).intersection(pcx.index)
v = (pc[common] - pcx[common]).to_numpy()
bs = v[np.random.default_rng(74).integers(0, len(v), (5000, len(v)))].mean(1)
expected = json.load(open(ROOT/'EXPECTED_OUTPUTS.json', encoding='utf-8'))['depmap']
print('consistency mean/median      %.6f / %.6f   (expected %.4f / %.4f)' % (cons[common].mean(), cons[common].median(), expected['consistency_mean_272'], expected['consistency_median_272']))
print('frozen prediction vs CERES/Chronos %.6f / %.6f (expected %.4f / %.4f)' % (pc[common].mean(), pcx[common].mean(), expected['frozen_pred_vs_CERES_mean'], expected['frozen_pred_vs_Chronos_mean']))
print('paired delta mean          %.6f           (expected %.4f)' % (v.mean(), expected['paired_delta_272']))
print('delta CI (seed=74, recomputed) [%.6f, %.6f]' % (np.percentile(bs,2.5), np.percentile(bs,97.5)))
print('delta>0 cells %d/%d  range %.6f ~ %.6f' % ((v>0).sum(), len(v), v.min(), v.max()))
