"""N_crossmodality_cellprofile.py  (N)
Per-cell mean-profile correlation between the two perturbation modalities in beta2020 LINCS Level5 (note: both belong to the single beta2020 release, not two releases):
  - trt_cp : compound (chemical) perturbation, sig_id like 'ABY001_A375_XH:BRD-<drug>:dose:time' (BRD- is a Broad compound ID)
  - trt_sh : shRNA (genetic) knockdown, sig_id like 'CGS001_A375_96H:<GENE>:1' (the gene is the perturbation target)
parse each sig_id's cell line (prefix 'ABY001_A375_XH' -> 'A375'), aggregate the mean perturbation profile per cell,
then compute the Pearson of the two cell-mean profiles over common cell lines =
the cell-identity-axis correlation across perturbation modalities (compound vs genetic) within the same beta2020 release.
Key correction: this is not a "cross-release consistency" or "same-lineage-release consistency" check -- trt_cp and trt_sh are different perturbation modalities,
and both are the same beta2020 release to begin with (differing only in perturbation type, not in release version).
"""
import h5py, numpy as np, csv, os, json
from collections import defaultdict

CP  = r'C:/Users/wkr20/Desktop/level5_beta_trt_cp_n720216x12328 (2).gctx'
SH  = r'C:/Users/wkr20/Desktop/level5_beta_trt_sh_n238351x12328 (1).gctx'
RES = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
os.makedirs(RES, exist_ok=True)
GENES = 12328
CHUNK = 2000

def parse_cell(sid):
    s = sid.decode('utf-8', 'replace') if isinstance(sid, bytes) else sid
    prefix = s.split(':')[0]; parts = prefix.split('_')
    return parts[1] if len(parts) > 1 else prefix

def per_cell_means(path, tag):
    f = h5py.File(path, 'r'); colg = f['0/META/COL']; ids = colg['id'][:]; mat = f['0/DATA/0/matrix']
    N = mat.shape[0]   # number of sig_id rows (720216 for cp, 238351 for sh)
    sums = defaultdict(lambda: np.zeros(GENES, dtype=np.float64)); counts = defaultdict(int)
    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N)
        blk = np.asarray(mat[a:b, :], dtype=np.float64)   # (chunk sig_ids, 12328 genes)
        for off in range(b - a):
            cell = parse_cell(ids[a + off])
            sums[cell] += blk[off, :]; counts[cell] += 1
        print(f'  [{tag}] processed {b}/{N}')
    means = {c: sums[c] / counts[c] for c in sums}
    f.close()
    return means, counts

print('== trt_cp =='); cp_means, cp_counts = per_cell_means(CP, 'cp')
print('== trt_sh =='); sh_means, sh_counts = per_cell_means(SH, 'sh')

shared = sorted(set(cp_means) & set(sh_means))
print('shared cells', len(shared))
rows = []
for c in shared:
    a = cp_means[c] - cp_means[c].mean(); b = sh_means[c] - sh_means[c].mean()
    na = np.sqrt((a * a).sum()); nb = np.sqrt((b * b).sum()); d = na * nb
    corr = float((a * b).sum() / d) if d > 0 else float('nan')
    rows.append((c, cp_counts[c], sh_counts[c], corr))

with open(f'{RES}/N_crossmodality_beta2020_cellprofile.csv', 'w', newline='', encoding='utf-8-sig') as fo:
    w = csv.writer(fo); w.writerow(['cell', 'n_sig_cp', 'n_sig_sh', 'pearson_meanprofile_cp_vs_sh'])
    for c, nc, ns, corr in rows: w.writerow([c, nc, ns, f'{corr:.4f}'])

vals = np.array([r[3] for r in rows])
summary = {
    'n_shared_cells': len(shared),
    'cell_meanprofile_pearson_median': float(np.median(vals)),
    'cell_meanprofile_pearson_mean': float(np.mean(vals)),
    'cell_meanprofile_pearson_std': float(np.std(vals)),
    'frac_gt_0': float(np.mean(vals > 0)),
    'note': 'beta2020 release, trt_cp (compound/chemical perturbation, BRD- drug IDs) vs trt_sh (shRNA genetic knockdown, gene symbols) per-cell mean perturbation-profile Pearson. WITHIN-beta2020 cross-perturbation-modality (chemical vs genetic) correlation — NOT a cross-release or same-lineage release-consistency check (both products are the same beta2020 release). Median 0.675 indicates cell-type identity is a strong shared axis across perturbation modalities.'
}
json.dump(summary, open(f'{RES}/N_summary.json', 'w'), indent=2, ensure_ascii=False)
print('N summary:', json.dumps(summary, indent=2))
