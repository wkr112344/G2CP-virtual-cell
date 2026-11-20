"""F_within_product_reproducibility.py  (F)
A within-product reproducibility proxy on Desktop Level5 (MODZ GSE92742, 473647 sig_ids x 12328 genes):
parse each sig_id into a (cell, pert) product (id like 'CPC005_A375_6H:BRD-..:10'),
for products with >=2 instances, compute the mean pairwise Pearson between their instance profiles = within-product consistency.
Randomly sample 4,000 multi-instance products to estimate the distribution.
Note: Level5 is already a replicate-consensus layer, so between-instance differences mix in dose/time heterogeneity; and no technical-replicate
split is available -> this is an upper-bound proxy for "within-product reproducibility / structural consistency" (an approximate fill for the noise-baseline gap noted in S14.5).
"""
import h5py, numpy as np, csv, os, json, random
from collections import defaultdict

MODZ = r'C:/Users/wkr20/Desktop/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx'
RES = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
os.makedirs(RES, exist_ok=True)
SAMPLE = 4000
KCAP = 30

def parse_cell_pert(sid):
    s = sid.decode('utf-8', 'replace') if isinstance(sid, bytes) else sid
    prefix = s.split(':')[0]; parts = prefix.split('_')
    cell = parts[1] if len(parts) > 1 else prefix
    pert = s.split(':')[1] if ':' in s else ''
    return cell, pert

f = h5py.File(MODZ, 'r')
colg = f['0/META/COL']
ids = colg['id'][:]
mat = f['0/DATA/0/matrix']
N = mat.shape[1]
print('sig_ids', N)

groups = defaultdict(list)
for j, sid in enumerate(ids):
    c, p = parse_cell_pert(sid)
    groups[(c, p)].append(j)
multi = [k for k, v in groups.items() if len(v) >= 2]
print('products with >=2 instances:', len(multi))

random.seed(0)
samp = random.sample(multi, min(SAMPLE, len(multi)))
print('sampled products', len(samp))

rows = []
for (c, p) in samp:
    idxs = groups[(c, p)]
    if len(idxs) > KCAP: idxs = idxs[:KCAP]
    blk = np.asarray(mat[idxs, :], dtype=np.float64)   # (k instances, 12328 genes); each row = a sig_id
    k = blk.shape[0]
    Xc = blk - blk.mean(1, keepdims=True)              # center each instance across genes
    norms = np.sqrt((Xc * Xc).sum(1)); norms[norms == 0] = 1e-12
    C = (Xc @ Xc.T) / (norms[:, None] * norms[None, :])   # (k instances) x (k instances)
    iu = np.triu_indices(k, 1)
    vals = C[iu]
    mpr = float(np.nanmean(vals)) if vals.size else float('nan')
    rows.append((c, p, k, mpr))

with open(f'{RES}/F_within_product_reproducibility.csv', 'w', newline='', encoding='utf-8-sig') as fo:
    w = csv.writer(fo); w.writerow(['cell', 'pert', 'n_instances', 'mean_pairwise_pearson'])
    for c, p, k, mpr in rows: w.writerow([c, p, k, f'{mpr:.4f}'])

vals = np.array([r[3] for r in rows])
summary = {
    'n_sampled_products': len(rows),
    'within_product_pearson_median': float(np.median(vals)),
    'within_product_pearson_mean': float(np.mean(vals)),
    'within_product_pearson_std': float(np.std(vals)),
    'frac_gt_0': float(np.mean(vals > 0)),
    'frac_gt_0_5': float(np.mean(vals > 0.5)),
    'note': 'MODZ Level5 within-(cell,pert) cross-instance Pearson; Level5-instance approx, mixes dose/time heterogeneity; no technical-replicate split available',
}
json.dump(summary, open(f'{RES}/F_summary.json', 'w'), indent=2, ensure_ascii=False)
print('F summary:', json.dumps(summary, indent=2))
f.close()
