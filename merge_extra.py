# -*- coding: utf-8 -*-
"""merge_extra.py —— 合并 pool（5 系：52k 基因扰动 + 41k 药物扰动）+ 全细胞系新数据（216k 药物扰动，289 系）
按 (pert_id, cell_line, pert_type) 分组均值去重，var 对齐 pool 顺序。
"""
import sys, os
import numpy as np
import pandas as pd
import anndata as ad

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(BASE, "data", "g2cp", "data", "LINCS", "pool", "pool_gene_chem_ctrl_adata.h5ad")
EXTRA = os.path.join(BASE, "data", "lincs_extra", "all_cells.h5ad")
OUT = os.path.join(BASE, "data", "lincs_extra", "merged.h5ad")


def log(m):
    print(f"[merge] {m}", flush=True)


a = ad.read_h5ad(POOL, backed="r")
Xp = np.asarray(a.X[:], dtype=np.float32) if not hasattr(a.X[:], "toarray") else a.X[:].toarray().astype(np.float32)
op = a.obs[["pert_id", "pert_type", "cmap_name", "cell_line", "qc_pass"]].copy()
vp = list(a.var_names)
a.file.close()
log(f"pool: {Xp.shape}, 细胞系 {op['cell_line'].nunique()}")

b = ad.read_h5ad(EXTRA, backed="r")
Xe = np.asarray(b.X[:], dtype=np.float32) if not hasattr(b.X[:], "toarray") else b.X[:].toarray().astype(np.float32)
oe = b.obs[["pert_id", "pert_type", "cmap_name", "cell_line", "qc_pass"]].copy()
ve = list(b.var_names)
b.file.close()
log(f"extra: {Xe.shape}, 细胞系 {oe['cell_line'].nunique()}")

# var 对齐到 pool 顺序
if ve != vp:
    idx = [ve.index(g) for g in vp]
    Xe = Xe[:, idx]
log("var 已对齐")

X = np.concatenate([Xp, Xe], axis=0)
o = pd.concat([op, oe], ignore_index=True)
log(f"合并原始: {X.shape}（样本 {len(o)}）")

# 按 (pert_id, cell_line, pert_type) 分组均值去重
grp = o.groupby(["pert_id", "cell_line", "pert_type"], sort=False)
keys = grp.groups.keys()
rows = []
for i, k in enumerate(keys):
    gi = grp.groups[k].values
    rows.append(gi)
gidx = np.array(rows, dtype=object)
n_grp = len(keys)
Xm = np.zeros((n_grp, X.shape[1]), dtype=np.float32)
for i, gi in enumerate(gidx):
    Xm[i] = X[gi].mean(axis=0)
log(f"分组去重后: {n_grp} 组")

obs = pd.DataFrame({
    "pert_id": [k[0] for k in keys],
    "cell_line": [k[1] for k in keys],
    "pert_type": [k[2] for k in keys],
    "cmap_name": [k[0] for k in keys],
    "qc_pass": 1,
}, index=[f"s{i}" for i in range(n_grp)])
var = pd.DataFrame(index=vp)
m = ad.AnnData(X=Xm, obs=obs, var=var)
m.write_h5ad(OUT)
log(f"✅ merged: {m.shape[0]} 样本 × {m.shape[1]} 基因 | 细胞系 {obs['cell_line'].nunique()} | "
    f"基因扰动 {(obs['pert_type']=='trt_xpr').sum()} + 药物扰动 {(obs['pert_type']=='trt_cp').sum()} | 药物 {obs.loc[obs['pert_type']=='trt_cp','pert_id'].nunique()}")
