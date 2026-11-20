"""Export the **two reference matrices** for LINCS 11,275x978 (closing the reproduction gap).

The calibers are row-for-row identical to `pair_level_978()` in `P_frozen_multimetric_cohorts.py`:
  - row order = the (cell, drug) order of the archived per-row table `frozen_multimetric_perrow_scores.csv`
  - column order = `internal_col` of `93_gene_map_12328_to_978.json` (the 978 landmarks)
  - reference values come from g2cp_cache: beta_v2/y.npy (level5beta2020), 2021/y.npy (dcic2021)

Check (decisive): compute the per-row Pearson between the archived frozen prediction matrix and the exported reference matrices;
  the drug-equal-weight mean must reproduce the main text 0.3680 (beta2020) / 0.0724 (dcic2021); otherwise abort.
"""
import hashlib, io, json, os, time
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37')
CACHE = {'beta2020': BASE / 'data/g2cp_cache_beta_v2', 'dcic2021': BASE / 'data/g2cp_cache_2021'}
GENE_MAP = BASE / '_scratch/v31_mvpa/93_gene_map_12328_to_978.json'
ROWS = Path('gigascience_upload_work/GigaScience_Submission_Upload/supplementary/frozen_multimetric_perrow_scores.csv')
# main-text caliber: the same beta-trained frozen prediction x the two references (M_BB=0.3680, M_BD=0.0724)
_PB = Path('gigascience_upload_work/GigaScience_Reproducibility_Package/source_data/'
           'frozen_prediction_matrices/frozen_pred_matrix_beta_trained_11275x978.npy')
PRED = {'beta2020': _PB, 'dcic2021': _PB}
OUT = Path('GigaScience_reproducibility_release/zenodo/source_data/lincs_reference_matrices')
OUT.mkdir(parents=True, exist_ok=True)
EXPECT = {'beta2020': 0.3680, 'dcic2021': 0.0724}
TOL = 2e-3


def log(m): print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


rows = pd.read_csv(ROWS)
pairs = list(zip(rows.cell.astype(str), rows.drug.astype(str)))
nP = len(pairs)
log('per-row table: %d rows (this row order is authoritative)' % nP)

gm = json.load(io.open(GENE_MAP, encoding='utf-8'))
icol = np.asarray(gm['internal_col'], dtype=np.int64)
assert icol.size == 978, icol.size
log('978 gene axis OK (internal_col)')

for tag, cache in CACHE.items():
    pm = np.load(cache / 'pairs_meta.npz', allow_pickle=True)
    cell = np.array([str(x) for x in pm['cell_name']], dtype=object)
    drug = np.array([str(x) for x in pm['drug_id']], dtype=object)
    pidx = {(cell[i], drug[i]): i for i in range(len(cell))}
    Y = np.load(cache / 'y.npy', mmap_mode='r')
    log('%s: y.npy %s, pairs %d, selecting rows' % (tag, Y.shape, len(cell)))
    sel, tgt = [], []
    for i, p in enumerate(pairs):
        j = pidx.get(p)
        if j is not None:
            sel.append(j); tgt.append(i)
    sel = np.asarray(sel, dtype=np.int64); tgt = np.asarray(tgt, dtype=np.int64)
    T = np.zeros((nP, 978), dtype=np.float32)
    CH = 5000
    for s in range(0, len(sel), CH):
        v = np.asarray(Y[sel[s:s+CH]], dtype=np.float32)
        T[tgt[s:s+CH]] = v[:, icol]
        del v
    log('  %d/%d rows matched -> T %s' % (len(sel), nP, T.shape))
    out = OUT / ('Y_%s_11275x978.npy' % tag)
    np.save(out, T)
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    log('  wrote %s (%.1f MB, md5 %s)' % (out.name, out.stat().st_size / 1e6, md5[:12]))

    P = np.load(PRED[tag], mmap_mode='r')
    assert P.shape == T.shape, (P.shape, T.shape)
    Pn = P - P.mean(axis=1, keepdims=True); Tn = T - T.mean(axis=1, keepdims=True)
    den = np.sqrt((Pn**2).sum(1) * (Tn**2).sum(1))
    r = (Pn * Tn).sum(1) / np.where(den > 0, den, np.nan)
    per_drug = pd.DataFrame({'drug': rows.drug.values, 'r': r}).groupby('drug').r.mean()
    got = float(per_drug.mean())
    ok = abs(got - EXPECT[tag]) <= TOL
    log('  * check: prediction x exported reference -> drug-equal-weight Pearson = %.6f (main text %.4f) %s' % (got, EXPECT[tag], 'OK' if ok else 'X MISMATCH'))
    json.dump({'tag': tag, 'file': out.name, 'shape': list(T.shape), 'md5': md5,
               'rows': 'row order of frozen_multimetric_perrow_scores.csv (cell, drug)',
               'cols': 'internal_col of 93_gene_map_12328_to_978.json (the 978 landmark genes)',
               'source': str(cache / 'y.npy'), 'verify_drug_equal_weight_pearson': got,
               'expected': EXPECT[tag], 'pass': ok},
              open(OUT / ('Y_%s_11275x978.json' % tag), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if not ok:
        raise SystemExit('check failed; aborting')
log('all done')
