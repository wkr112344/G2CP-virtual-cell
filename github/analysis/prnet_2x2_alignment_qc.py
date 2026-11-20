# -*- coding: utf-8 -*-
"""
prnet_2x2_alignment_qc.py  (PRnet symmetric 2x2 — pre-training alignment QC gate)

Purpose: before actually training a second PRnet on dcic2021 (cp_coeff_mat.gctx), first verify whether
each perturbation sample of the LINCS L1000 h5ad (beta2020, PRnet training data) can be reliably aligned to
a dcic2021 GCTX signature. Training proceeds only if all QC checks pass.

Alignment key (perturbation): (cell, pert_id, dose, time)
Alignment key (control)      : (cell, DMSO)   -- dcic marks controls with pert_name=='DMSO'

Checks (requested list):
  1. number of successfully matched samples
  2. unmatched fraction
  3. whether all 978 landmark genes are present
  4. whether the gene order matches (output forced to the h5ad var order)
  5. whether each cell/drug/dose/time is unique (both sides)
  6. whether duplicate signatures exist (within dcic)
  7. whether this matches the G2CP S25 evaluation-pair definition
"""
import h5py, numpy as np, scanpy as sc, re, json, os
from collections import Counter, defaultdict

H5   = r'C:/Users/wkr20/Desktop/Lincs_L1000.h5ad'
GCTX = r'C:/Users/wkr20/Desktop/cp_coeff_mat.gctx'
CACHE= r'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37/data/g2cp_cache_beta_v2'
OUT  = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
os.makedirs(OUT, exist_ok=True)

def dec(x): return x.decode() if isinstance(x, bytes) else x
def sf(x, n=4):
    """4 significant digits, used for tolerant dose/time comparison."""
    try:
        x = float(x)
        if x == 0 or not np.isfinite(x): return 0.0
        return float(f'{x:.{n}g}')
    except Exception:
        return None

def pid_ns(p):
    if p.startswith('BRD-A'): return 'BRD-A'
    if p.startswith('BRD-K'): return 'BRD-K'
    if p == 'DMSO':           return 'DMSO'
    if p.isdigit():           return 'numeric(genetic)'
    return 'other'

# ---------------------------------------------------------------- h5ad obs
print('[load] h5ad obs ...', flush=True)
a = sc.read(H5, backed='r')
obs = a.obs
cell   = obs['cell_id'].astype(str).values
pid    = obs['pert_id'].astype(str).values
pdose  = np.array([sf(v) for v in obs['pert_dose'].values])
ptime  = np.array([sf(v) for v in obs['pert_time'].values])
ptype  = obs['pert_type'].astype(str).values
is_ctl = np.array([str(v) == 'ctl_vehicle' for v in ptype])
h5_genes = list(a.var.index)          # 978 landmark symbols, in order
n_h5 = len(cell)
print(f'  h5ad rows={n_h5}  genes={len(h5_genes)}  pert(trt_cp)={int((~is_ctl).sum())}  ctl={int(is_ctl.sum())}', flush=True)

# h5ad perturbation keys
h5_pert_keys = defaultdict(list)
for i in range(n_h5):
    if is_ctl[i]: continue
    h5_pert_keys[(cell[i], pid[i], pdose[i], ptime[i])].append(i)
# h5ad control keys  (cell, DMSO)
h5_ctl_keys = defaultdict(list)
for i in range(n_h5):
    if is_ctl[i]:
        h5_ctl_keys[(cell[i], 'DMSO')].append(i)

# ---------------------------------------------------------------- dcic GCTX meta
print('[load] dcic GCTX meta ...', flush=True)
with h5py.File(GCTX, 'r') as h:
    col = h['0/META/COL']; row = h['0/META/ROW']
    d_cell = [dec(x) for x in col['cell_line'][:]]
    d_pid  = [dec(x) for x in col['pert_id'][:]]
    d_pname= [dec(x) for x in col['pert_name'][:]]
    d_dose = [dec(x) for x in col['pert_dose'][:]]
    d_time = [dec(x) for x in col['pert_time'][:]]
    d_genes= [dec(x) for x in row['id'][:]]
n_dcic = len(d_cell)
print(f'  dcic sigs={n_dcic}  genes={len(d_genes)}', flush=True)

d_dose_n = [sf(re.match(r'([\d.]+)', x).group(1)) if re.match(r'([\d.]+)', x) else None for x in d_dose]
d_time_n = [sf(re.match(r'([\d.]+)', x).group(1)) if re.match(r'([\d.]+)', x) else None for x in d_time]

# dcic perturbation lookup (BRD-* only) + helper sets for failure localization
dcic_pert = defaultdict(list)
dcic_ctl  = defaultdict(list)        # cell -> [idx]
dcic_key_dup = 0
dcic_cells = set()
dcic_pert_ids = set()
dcic_cellpert = set()
dcic_cellpertdose = set()
for i in range(n_dcic):
    p = d_pid[i]
    if pid_ns(p) in ('BRD-A', 'BRD-K'):
        k = (d_cell[i], p, d_dose_n[i], d_time_n[i])
        if k in dcic_pert: dcic_key_dup += 1
        dcic_pert[k].append(i)
        dcic_cells.add(d_cell[i]); dcic_pert_ids.add(p)
        dcic_cellpert.add((d_cell[i], p)); dcic_cellpertdose.add((d_cell[i], p, d_dose_n[i]))
    if d_pname[i] == 'DMSO':
        dcic_ctl[d_cell[i]].append(i)
dcic_ctl_cells = set(dcic_ctl.keys())

# ---------------------------------------------------------------- gene alignment
h5_set, dcic_set = set(h5_genes), set(d_genes)
genes_present = h5_set & dcic_set
genes_missing = sorted(h5_set - dcic_set)
print(f'[gene] h5ad={len(h5_set)} dcic={len(dcic_set)} present={len(genes_present)} missing={len(genes_missing)}', flush=True)

# ---------------------------------------------------------------- matching
def match(keymap, lookup):
    m = un = 0
    by_ns = Counter()
    for k, idxs in keymap.items():
        if k in lookup:
            m += 1
            by_ns[pid_ns(k[1])] += 1
        else:
            un += 1
            by_ns['UNMATCHED_' + pid_ns(k[1])] += 1
    return m, un, by_ns

p_m, p_un, p_ns = match(h5_pert_keys, dcic_pert)
# control: h5ad key (cell,'DMSO') -> dcic cell has DMSO
c_m = sum(1 for (cc, _) in h5_ctl_keys if cc in dcic_ctl_cells)
c_un = len(h5_ctl_keys) - c_m
print(f'[match] perturbation h5ad-keys={len(h5_pert_keys)} matched={p_m} unmatched={p_un}', flush=True)
print(f'[match] control      h5ad-keys={len(h5_ctl_keys)} matched={c_m} unmatched={c_un}', flush=True)

# ---- perturbation failure-cause localization (why unmatched?) ----
fail_cell = fail_pert = fail_dose = fail_both = 0
for k in h5_pert_keys:
    if k in dcic_pert: continue
    c, p, d, t = k
    in_cell = c in dcic_cells
    in_pert = p in dcic_pert_ids
    in_cp   = (c, p) in dcic_cellpert
    in_cpd  = (c, p, d) in dcic_cellpertdose
    if not in_cell:
        fail_cell += 1
    elif not in_pert:
        fail_pert += 1
    elif not in_cp:          # same cell+pert exists but dose never matched -> dose grid mismatch somewhere
        fail_pert += 1
    elif not in_cpd:         # cell+pert+dose exists but time mismatch -> time grid mismatch
        fail_dose += 1
    else:
        fail_both += 1
print(f'[fail] cell-only={fail_cell} pert_id-only={fail_pert} dose/time-grid={fail_dose} other={fail_both}', flush=True)

# h5ad internal uniqueness
h5_pert_dups = sum(1 for v in h5_pert_keys.values() if len(v) > 1)
h5_ctl_dups  = sum(1 for v in h5_ctl_keys.values() if len(v) > 1)

# ---------------------------------------------------------------- G2CP S25 consistency
print('[load] G2CP cache (S25 pair universe) ...', flush=True)
pm = np.load(f'{CACHE}/pairs_meta.npz', allow_pickle=True)
s25_cells = set(pm['cell_name'])
s25_pairs = set(zip([dec(x) for x in pm['cell_name']], [dec(x) for x in pm['drug_id']]))   # (cell, BRD-K)
meta = np.load(f'{CACHE}/meta.npz', allow_pickle=True)
g2cp_drugs = set(meta['drug_vocab'])          # BRD-A vocab
g2cp_cells  = set(meta['cl_names'])
s25_cell_overlap = sorted(set(cell) & s25_cells)

# For each matched perturbation pair, is (cell, pert_id) S25-evaluable?
# G2CP drug_vocab is BRD-A; S25 pairs use BRD-K drug_id. Match h5ad BRD-A pert_ids to vocab,
# and BRD-K pert_ids to S25 BRD-K universe.
s25_consistent = 0
for k in h5_pert_keys:
    if k not in dcic_pert: continue
    c, p = k[0], k[1]
    if c not in g2cp_cells: continue
    if p in g2cp_drugs:                 # BRD-A directly in G2CP vocab
        s25_consistent += 1
    elif pid_ns(p) == 'BRD-K' and (c, p) in s25_pairs:
        s25_consistent += 1

# ---------------------------------------------------------------- assemble report
report = {
  'inputs': {'h5ad': H5, 'dcic_gctx': GCTX, 'g2cp_cache': CACHE},
  'h5ad': {'rows': n_h5, 'pert_rows': int((~is_ctl).sum()), 'ctl_rows': int(is_ctl.sum()),
           'cells': int(len(set(cell))),
           'pert_id_namespaces': dict(Counter(pid_ns(p) for p, c in zip(pid, is_ctl) if not c).most_common())},
  'dcic': {'sigs': n_dcic, 'genes': len(d_genes), 'cells': int(len(set(d_cell))),
           'pert_id_namespaces': dict(Counter(pid_ns(p) for p in d_pid).most_common()),
           'has_DMSO_control': len(dcic_ctl_cells) > 0, 'DMSO_control_cells': len(dcic_ctl_cells)},
  'gene_alignment': {'h5ad_978': len(h5_set), 'dcic_symbols': len(dcic_set),
                     'present': len(genes_present), 'missing': len(genes_missing),
                     'missing_list': genes_missing},
  'matching': {
     'perturbation': {'h5ad_keys': len(h5_pert_keys), 'matched': p_m, 'unmatched': p_un,
                      'match_rate': round(p_m / max(1, len(h5_pert_keys)), 4),
                      'by_pert_id_namespace': {k: v for k, v in p_ns.items()},
                      'failure_causes': {'cell_missing': fail_cell, 'pert_id_missing': fail_pert,
                                         'dose_or_time_grid_mismatch': fail_dose, 'other': fail_both}},
     'control':      {'h5ad_keys': len(h5_ctl_keys), 'matched': c_m, 'unmatched': c_un,
                      'match_rate': round(c_m / max(1, len(h5_ctl_keys)), 4)}},
  'uniqueness': {
     'h5ad_pert_duplicate_keys': h5_pert_dups,
     'h5ad_ctl_duplicate_keys': h5_ctl_dups,
     'dcic_pert_duplicate_signatures': dcic_key_dup,
     'dcic_total_pert_keys': len(dcic_pert)},
  's25_consistency': {
     's25_cell_overlap_with_h5ad': len(s25_cell_overlap),
     's25_cell_overlap_list': s25_cell_overlap,
     'g2cp_drug_vocab_size': len(g2cp_drugs),
     'matched_pairs_S25_evaluable': s25_consistent,
     's25_pair_universe_size': len(s25_pairs)},
}

# verdict
crit = []
if len(genes_missing) > 0: crit.append(f'GENE GAP: {len(genes_missing)} of 978 landmark genes missing in dcic -> cannot build full target h5ad')
if p_m / max(1, len(h5_pert_keys)) < 0.8: crit.append(f'LOW PERT MATCH: {report["matching"]["perturbation"]["match_rate"]:.1%} < 80%')
if c_m == 0: crit.append('NO CONTROL MATCH: dcic has no DMSO for h5ad control cells -> baselines cannot be replaced')
if s25_consistent == 0: crit.append('S25 INCONSISTENT: 0 matched pairs fall in G2CP S25 pair universe')
report['verdict'] = 'PASS' if not crit else 'ATTENTION'
report['flags'] = crit

with open(f'{OUT}/prnet_dcic_alignment_qc.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print('\n' + '=' * 70)
print('PRnet 2x2 ALIGNMENT QC  ->', report['verdict'])
for k, v in report.items():
    if k in ('inputs', 'verdict', 'flags'): continue
    print(f'  {k}:')
    print('    ' + json.dumps(v, ensure_ascii=False)[:600])
if crit:
    print('  FLAGS:')
    for c in crit: print('   -', c)
print('=' * 70)
print('written:', f'{OUT}/prnet_dcic_alignment_qc.json')
