"""I_product_decomposition.py  (I)  --  chunked / memory-bounded
The two Level-5 ground-truth products (beta_v2 y.npy 158094x12328, 2021 y.npy 156931x12328)
are aligned by (cell_name, drug_id); common products = 156,760 (the full intersection, larger than the 11,275 evaluable pairs in the main analysis,
covering the complete common product population). Per product, compute the across-gene Pearson between the beta2020 and dcic2021 profiles = that product's two-reference agreement,
then decompose by cell / drug. Chunked processing avoids loading ~30 GB at once.
"""
import numpy as np, csv, os, json
from collections import defaultdict

WS = r'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37'
RES = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
os.makedirs(RES, exist_ok=True)
CHUNK = 4000

def load_idx(tag):
    pm = np.load(f'{WS}/data/g2cp_cache_{tag}/pairs_meta.npz', allow_pickle=True)
    cells = list(pm['cell_name']); drugs = list(pm['drug_id'])
    keymap = {}
    for i, (c, d) in enumerate(zip(cells, drugs)):
        keymap.setdefault((c, d), i)
    return np.load(f'{WS}/data/g2cp_cache_{tag}/y.npy', mmap_mode='r'), keymap

yb, mb = load_idx('beta_v2')
y2, m2 = load_idx('2021')

common = sorted(set(mb) & set(m2))
n = len(common)
print('common (cell,drug) products:', n)
Ai = np.array([mb[k] for k in common])
Bi = np.array([m2[k] for k in common])
cells_c = np.array([k[0] for k in common])
drugs_c = np.array([k[1] for k in common])

r_all = np.empty(n, dtype=np.float32)
for s in range(0, n, CHUNK):
    e = min(s + CHUNK, n)
    A = np.array(yb[Ai[s:e]], dtype=np.float32)
    B = np.array(y2[Bi[s:e]], dtype=np.float32)
    Ac = A - A.mean(1, keepdims=True)
    Bc = B - B.mean(1, keepdims=True)
    na = np.sqrt((Ac*Ac).sum(1)); nb = np.sqrt((Bc*Bc).sum(1))
    denom = na*nb; denom[denom == 0] = 1e-12
    r_all[s:e] = (Ac*Bc).sum(1) / denom
    if (s // CHUNK) % 10 == 0:
        print(f'  chunk {s}-{e} done')

with open(f'{RES}/I_perproduct_pearson.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['cell', 'drug', 'pearson_beta_vs_2021'])
    for c, d, rv in zip(cells_c, drugs_c, r_all):
        w.writerow([c, d, f'{rv:.4f}'])

cell_stat = defaultdict(list); drug_stat = defaultdict(list)
for c, d, rv in zip(cells_c, drugs_c, r_all):
    cell_stat[c].append(float(rv)); drug_stat[d].append(float(rv))

def agg(d): return {'n': len(d), 'mean': float(np.mean(d)), 'median': float(np.median(d))}
with open(f'{RES}/I_by_cell.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['cell', 'n_products', 'mean_pearson', 'median_pearson'])
    for c in sorted(cell_stat, key=lambda x: -len(cell_stat[x])):
        a = agg(cell_stat[c]); w.writerow([c, a['n'], f"{a['mean']:.4f}", f"{a['median']:.4f}"])
with open(f'{RES}/I_by_drug.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['drug', 'n_products', 'mean_pearson', 'median_pearson'])
    for d in sorted(drug_stat, key=lambda x: -len(drug_stat[x])):
        a = agg(drug_stat[d]); w.writerow([d, a['n'], f"{a['mean']:.4f}", f"{a['median']:.4f}"])

summary = {
    'n_common_products': int(n),
    'pearson_median': float(np.median(r_all)),
    'pearson_mean': float(np.mean(r_all)),
    'pearson_std': float(np.std(r_all)),
    'frac_pearson_gt_0': float(np.mean(r_all > 0)),
    'frac_pearson_gt_0_3': float(np.mean(r_all > 0.3)),
    'pearson_min': float(r_all.min()), 'pearson_max': float(r_all.max()),
    'n_cells': len(cell_stat), 'n_drugs': len(drug_stat),
    'cell_mean_pearson_median': float(np.median([agg(v)['mean'] for v in cell_stat.values()])),
    'drug_mean_pearson_median': float(np.median([agg(v)['mean'] for v in drug_stat.values()])),
    'gene_panel': 12328,
}
with open(f'{RES}/I_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('I summary:', json.dumps(summary, indent=2))
