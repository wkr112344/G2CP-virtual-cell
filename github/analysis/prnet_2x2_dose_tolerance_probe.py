# -*- coding: utf-8 -*-
"""Dose-tolerance sensitivity for PRnet 2x2: at what relative dose tolerance does
h5ad (beta2020) perturbation keys match dcic2021 signatures? Informs whether a
tolerant hard-swap is defensible (the note 'inconsistent dose/time definitions cannot be hard-swapped')."""
import h5py, numpy as np, scanpy as sc, re, json, os
from collections import defaultdict

H5   = r'C:/Users/wkr20/Desktop/Lincs_L1000.h5ad'
GCTX = r'C:/Users/wkr20/Desktop/cp_coeff_mat.gctx'
OUT  = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
def dec(x): return x.decode() if isinstance(x, bytes) else x
def sf(x, n=4):
    x=float(x)
    return 0.0 if x==0 else float(f'{x:.{n}g}')

a=sc.read(H5, backed='r'); obs=a.obs
cell=obs['cell_id'].astype(str).values; pid=obs['pert_id'].astype(str).values
pdose=np.array([sf(v) for v in obs['pert_dose'].values]); ptime=np.array([sf(v) for v in obs['pert_time'].values])
ptype=obs['pert_type'].astype(str).values; is_ctl=np.array([str(v)=='ctl_vehicle' for v in ptype])

with h5py.File(GCTX,'r') as h:
    col=h['0/META/COL']
    d_cl=[dec(x) for x in col['cell_line'][:]]; d_pid=[dec(x) for x in col['pert_id'][:]]
    d_dose=[dec(x) for x in col['pert_dose'][:]]; d_time=[dec(x) for x in col['pert_time'][:]]

def ns(p):
    if p.startswith('BRD-A'): return 'BRD-A'
    if p.startswith('BRD-K'): return 'BRD-K'
    return 'other'

# dcic grouped by (cell, pert_id, time) -> sorted dose list  (BRD-* only)
grp=defaultdict(list)
for i in range(len(d_cl)):
    p=d_pid[i]
    if ns(p) in ('BRD-A','BRD-K'):
        md=re.match(r'([\d.]+)', d_dose[i]); mt=re.match(r'([\d.]+)', d_time[i])
        if md and mt:
            d=sf(md.group(1)); t=sf(mt.group(1))
            grp[(d_cl[i], p, t)].append(d)
for k in grp: grp[k].sort()

# h5ad perturbation keys
hkeys=defaultdict(list)
for i in range(len(cell)):
    if is_ctl[i]: continue
    if ns(pid[i]) in ('BRD-A','BRD-K'):
        hkeys[(cell[i], pid[i], pdose[i], ptime[i])].append(i)

def match_tol(tol_rel):
    m=0
    for (c,p,d,t) in hkeys:
        lst=grp.get((c,p,t))
        if not lst: continue
        for x in lst:
            if x>0 and abs(x-d)/x <= tol_rel:
                m+=1; break
    return m

res={}
for tol in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    mm=match_tol(tol)
    res[f'tol_{tol:.3f}']={'matched':mm,'rate':round(mm/len(hkeys),4)}
print('h5ad pert unique keys:', len(hkeys))
for k,v in res.items(): print(f'  {k}: matched={v["matched"]}  rate={v["rate"]:.1%}')

json.dump({'h5ad_pert_keys':len(hkeys),'tolerance_sweep':res},
          open(f'{OUT}/prnet_dcic_dose_tolerance_sweep.json','w'), indent=2)
print('written:', f'{OUT}/prnet_dcic_dose_tolerance_sweep.json')
