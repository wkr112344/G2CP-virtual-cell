#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_ridge_ceiling.py -- measure the "feature ceiling" with a strong linear baseline (ridge regression).

Purpose: distinguish "PRnet too weak / wrong loss" (a model problem) from "these features carry no learnable signal for Y" (a feature/target problem).
If even ridge regression (a regularized estimator strong on linear signal) has val_mse ~ 1.0 on val (= predicting the marginal mean),
then there is simply no generalizable X->Y mapping in the features, and strengthening PRnet will not help -- the root cause is upstream (features/target).

Features are built exactly as in the training script: (X_baseline, drug_emb, dose_z, time_z), with targets Y_beta / Y_dcic.
To save memory it runs on a subset only. val_mse uses the same caliber as the training script (z-scored target variance ~ 1, floor = 1.0).
"""
import os, sys, time
import h5py
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import rankdata

DATA = r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/prnet_commonX/PRnet_commonX_dualY.h5"
N_TRAIN = 30000
N_VAL = 10000
ALPHA = 1.0

def build_feat(h5, idx, dose_z, time_z):
    Xb = h5["X_baseline"][idx].astype(np.float32)
    de = h5["drug_emb"][idx].astype(np.float32)
    dz = dose_z[idx][:, None].astype(np.float32)
    tz = time_z[idx][:, None].astype(np.float32)
    return np.concatenate([Xb, de, dz, tz], 1).astype(np.float32)

def per_sample_corr(pred, true, method="spearman"):
    out = []
    for i in range(len(pred)):
        p = pred[i]; t = true[i]
        if method == "pearson":
            pm = p - p.mean(); tm = t - t.mean()
        else:
            pm = rankdata(p) - (len(p)+1)/2.0
            tm = rankdata(t) - (len(t)+1)/2.0
        den = np.sqrt((pm**2).sum() * (tm**2).sum())
        out.append((pm*tm).sum()/den if den > 1e-9 else np.nan)
    return np.array(out)

t0 = time.time()
with h5py.File(DATA, "r") as f:
    split = f["split"][:].astype(int)
    dose = f["dose"][:].astype(np.float64)
    tcol = f["time"][:].astype(np.float64)
dm, ds = dose.mean(), dose.std() or 1.0
tm_, ts = tcol.mean(), tcol.std() or 1.0
dose_z = ((dose - dm)/ds).astype(np.float32)
time_z = ((tcol - tm_)/ts).astype(np.float32)

tr = np.where(split == 0)[0]; va = np.where(split == 1)[0]
rng = np.random.default_rng(2024)
tr = rng.choice(tr, min(N_TRAIN, len(tr)), replace=False)
va = rng.choice(va, min(N_VAL, len(va)), replace=False)
tr = np.sort(tr); va = np.sort(va)   # h5py requires fancy indices in ascending order

with h5py.File(DATA, "r") as f:
    Xtr = build_feat(f, tr, dose_z, time_z); Xva = build_feat(f, va, dose_z, time_z)
    Yb_tr = f["Y_beta"][tr].astype(np.float32); Yb_va = f["Y_beta"][va].astype(np.float32)
    Yd_tr = f["Y_dcic"][tr].astype(np.float32); Yd_va = f["Y_dcic"][va].astype(np.float32)
print(f"[load] train={Xtr.shape} val={Xva.shape} ({time.time()-t0:.1f}s)")

for name, Ytr, Yva in [("Y_beta", Yb_tr, Yb_va), ("Y_dcic", Yd_tr, Yd_va)]:
    t1 = time.time()
    clf = Ridge(alpha=ALPHA, solver="cholesky")
    clf.fit(Xtr, Ytr)
    pred = clf.predict(Xva)
    mse = float(np.mean((pred - Yva)**2))
    sp = per_sample_corr(pred, Yva, "spearman")
    pe = per_sample_corr(pred, Yva, "pearson")
    # marginal-mean prediction baseline
    base_mse = float(np.mean((Ytr.mean(0, keepdims=True) - Yva)**2))
    print(f"[{name}] ridge val_mse={mse:.4f}  (marginal-mean baseline mse={base_mse:.4f})  "
          f"mean per-sample sp={np.nanmean(sp):+.4f} pe={np.nanmean(pe):+.4f}  ({time.time()-t1:.1f}s)")

print(f"[done] total {time.time()-t0:.1f}s")
