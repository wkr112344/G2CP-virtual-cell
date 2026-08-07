# -*- coding: utf-8 -*-
"""prep_fusion.py —— 干净标签融合数据重建（替代被污染的 merged 缓存）。

核心修复：merge_extra 曾把 pool(poly-G2CP 处理版) 与 extra(level5_beta 原始版)
按 (pert_id, cell, pert_type) 均值合并 —— 但两批对同一实验的读数 PCC≈0.03，
均值=互相矛盾的答案平均 → 训练标签污染 → PCC 天花板从 0.46 掉到 0.35。

本脚本：
- 5 系(HT29/A375/A549/MCF7/PC3)：只用 pool 原始样本(含重复，保留实验内噪声结构)
- 284 个新细胞系：只用 extra 原始样本(level5_beta 单一口径)
- 不合并、不平均，样本级直存 → 每个标签都是"单一来源的真实读数"
- var 统一 pool 顺序(pool 与 extra var 已完全一致)

输出：data/g2cp_cache_fusion/{meta.npz, y.npy, drug_fps.npy, hvg.json}
"""
import sys, os, json
import numpy as np
import pandas as pd
import anndata as ad

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(BASE, "data", "g2cp", "data", "LINCS", "pool", "pool_gene_chem_ctrl_adata.h5ad")
EXTRA = os.path.join(BASE, "data", "lincs_extra", "all_cells.h5ad")
OUT = os.path.join(BASE, "data", "g2cp_cache_fusion")
FIVE = ["HT29", "A375", "A549", "MCF7", "PC3"]


def log(m):
    print(f"[fusion] {m}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---------- pool: 5系 干净样本 ----------
    a = ad.read_h5ad(POOL, backed="r")
    op = a.obs.copy()
    vp = list(a.var_names)
    Xp = np.asarray(a.X[:], dtype=np.float32)
    a.file.close()
    log(f"pool: {Xp.shape}, 细胞系 {op['cell_line'].nunique()}")

    # ---------- extra: 全细胞系 ----------
    b = ad.read_h5ad(EXTRA, backed="r")
    oe = b.obs.copy()
    ve = list(b.var_names)
    Xe = np.asarray(b.X[:], dtype=np.float32)
    b.file.close()
    if ve != vp:
        idx = [ve.index(g) for g in vp]
        Xe = Xe[:, idx]
    log(f"extra: {Xe.shape}, 细胞系 {oe['cell_line'].nunique()}")

    # ---------- 样本选源 ----------
    # pool 全部 trt_cp / trt_xpr（5系，含重复）
    m_pool = op["pert_type"].astype(str).isin(["trt_cp", "trt_xpr"])
    # extra 只取非5系 trt_cp（5系与 pool 重复且口径不同 → 丢弃）
    m_extra = oe["pert_type"].astype(str).eq("trt_cp") & ~oe["cell_line"].astype(str).isin(FIVE)
    log(f"pool 样本 {int(m_pool.sum())} | extra 非5系样本 {int(m_extra.sum())}")

    X = np.concatenate([Xp[m_pool.values], Xe[m_extra.values]], axis=0)
    op2 = op[m_pool.values].copy()
    oe2 = oe[m_extra.values].copy()
    op2["_src"] = "pool"
    oe2["_src"] = "extra"
    o = pd.concat([op2, oe2], ignore_index=True)
    n = len(o)
    log(f"融合总样本 {n}")

    # ---------- 词表 ----------
    # 基因：pool trt_xpr 的 cmap_name（基因符号，4994）
    gmask = o["pert_type"].astype(str).eq("trt_xpr")
    gacc = sorted(o.loc[gmask, "cmap_name"].astype(str).unique().tolist())
    gene_vocab = {g: i for i, g in enumerate(gacc)}
    # 药物：pool trt_cp(BRD-) + extra 非5系 trt_cp(BRD-)
    cpmask = o["pert_type"].astype(str).eq("trt_cp")
    cp_ids = sorted(o.loc[cpmask, "pert_id"].astype(str).unique().tolist())
    cp_ids = [p for p in cp_ids if p.startswith("BRD-")]
    drug_vocab = {p: i for i, p in enumerate(cp_ids)}
    log(f"基因 {len(gene_vocab)} | 药物 {len(drug_vocab)}")

    # ---------- SMILES 指纹 ----------
    smi_path = os.path.join(BASE, "data", "g2cp", "data", "processed_CMAP_compound_info.csv")
    pid2sm = {}
    if os.path.isfile(smi_path):
        info = pd.read_csv(smi_path)
        if "pert_id" in info.columns and "canonical_smiles" in info.columns:
            pid2sm = {str(r["pert_id"]): str(r["canonical_smiles"])
                      for _, r in info.iterrows() if pd.notna(r["canonical_smiles"])}
    legacy = os.path.join(BASE, "data", "lincs_drug_smiles.json")
    if os.path.isfile(legacy):
        try:
            pid2sm.update(json.load(open(legacy)))
        except Exception:
            pass
    from unipret.compound_encoder import smiles_to_ecfp4, ECFP4_BITS
    fps = np.zeros((len(cp_ids), ECFP4_BITS), dtype=np.float32)
    n_ok = 0
    for i, pid in enumerate(cp_ids):
        sm = pid2sm.get(pid)
        if sm:
            f = smiles_to_ecfp4(sm)
            if f is not None and f.any():
                fps[i] = f
                n_ok += 1
    log(f"药物指纹解析 {n_ok}/{len(cp_ids)}")

    # ---------- 细胞系 ----------
    cl_vals = o["cell_line"].astype(str).unique().tolist()
    cl_map = {v: i for i, v in enumerate(cl_vals)}
    log(f"细胞系 {len(cl_vals)} 个（5系 pool + {len(cl_vals)-5} 新系 extra）")

    # ---------- 构建 meta ----------
    kind = np.zeros(n, dtype=np.int8)
    key = np.zeros(n, dtype=np.int32)
    cell = np.zeros(n, dtype=np.int32)
    pid = o["pert_id"].astype(str)
    cmap = o["cmap_name"].astype(str)
    pt = o["pert_type"].astype(str)
    cl = o["cell_line"].astype(str)
    for j in range(n):
        if pt.iloc[j] == "trt_xpr":
            kind[j] = 0
            key[j] = gene_vocab.get(cmap.iloc[j], -1)
        else:
            kind[j] = 1
            key[j] = drug_vocab.get(pid.iloc[j], -1)
        cell[j] = cl_map.get(cl.iloc[j], 0)
    valid = key >= 0
    # 无指纹药物样本过滤
    zero_fp = ~fps.any(axis=1)
    if zero_fp.any():
        bad = 0
        for i in range(n):
            if kind[i] == 1 and key[i] >= 0 and zero_fp[key[i]]:
                valid[i] = False
                bad += 1
        log(f"过滤无指纹药物样本 {bad} 个")
    n_valid = int(valid.sum())
    log(f"有效样本 {n_valid}（基因 {(kind[valid]==0).sum()} + 药物 {(kind[valid]==1).sum()}）")
    # 保留 valid 中所有细胞系（含 pool 5系 + extra 新系）
    kept_cl = np.unique(cell[valid])
    log(f"有效样本覆盖细胞系 {len(kept_cl)}/{len(cl_vals)}")

    np.savez(os.path.join(OUT, "meta.npz"),
             kind=kind[valid], key=key[valid], cell=cell[valid],
             cl_names=np.array(cl_vals, dtype=object),
             gene_vocab=np.array(gacc, dtype=object),
             drug_vocab=np.array(cp_ids, dtype=object))
    np.save(os.path.join(OUT, "drug_fps.npy"), fps)
    vj = np.where(valid)[0]
    y = np.zeros((n_valid, X.shape[1]), dtype=np.float32)
    for b0 in range(0, len(vj), 5000):
        bi = vj[b0:b0 + 5000]
        y[b0:b0 + len(bi)] = X[bi]
    np.save(os.path.join(OUT, "y.npy"), y)
    json.dump(vp, open(os.path.join(OUT, "hvg.json"), "w"))
    log(f"y {y.shape} | 缓存完成 → {OUT}")


if __name__ == "__main__":
    main()
