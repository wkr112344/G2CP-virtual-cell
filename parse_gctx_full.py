# -*- coding: utf-8 -*-
"""parse_gctx_full.py —— 全基因版 gctx 解析（12,327 基因输出,含 FABP4/CD36/LPL/ADIPOQ）。
流式写入,内存友好。输出 data/g2cp_cache_fullgene/{meta.npz,y.npy,drug_fps.npy,hvg.json}
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import anndata as ad

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(BASE, "data", "g2cp", "data", "LINCS", "pool", "pool_gene_chem_ctrl_adata.h5ad")
GCTX = r"D:/下载/level5_beta_trt_cp_n720216x12328.gctx"
OUT = os.path.join(BASE, "data", "g2cp_cache_fullgene")


def log(m):
    print(f"[full] {m}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    import h5py

    # 1) gctx 基因(Entrez ID → symbol)
    f = h5py.File(GCTX, "r")
    rows = [x.decode() for x in f["/0/META/ROW/id"][:]]
    n_gene = len(rows)
    log(f"gctx 基因数 {n_gene}")

    # gctx ROW/id 是 Entrez 数字 ID → 用 MyGene 映射文件拿符号
    pool = ad.read_h5ad(POOL, backed="r")
    vp = list(pool.var_names)
    pool_ent = [int(pool.var["gene_id"].iloc[j]) for j in range(len(vp))]
    pool.file.close()
    ent2sym = json.load(open(os.path.join(BASE, "data", "g2cp_cache_fullgene_entrez2sym.json")))
    sym_list = [ent2sym.get(x, f"GENE_{x}") for x in rows]
    gctx_sym_set = set(sym_list)

    # 2) 样本元数据
    sigs = [x.decode() for x in f["/0/META/COL/id"][:]]
    n_samp = len(sigs)
    cells, pert_ids = [], []
    for s in sigs:
        head, tail = s.split(":", 1)
        parts = head.split("_")
        cell = parts[1] if len(parts) > 1 else "?"
        pid = tail.split(":")[0]
        cells.append(cell)
        pert_ids.append(pid)
    cells = np.array(cells)
    pert_ids = np.array(pert_ids)
    log(f"gctx 样本 {n_samp}")

    # 3) SMILES 指纹
    from unipret.compound_encoder import smiles_to_ecfp4, ECFP4_BITS
    pid2sm = {}
    smi_path = os.path.join(BASE, "data", "g2cp", "data", "processed_CMAP_compound_info.csv")
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

    # 4) 药物样本: 全基因(trt_cp + BRD- + 有指纹)
    # gctx 药物 ID 带批次后缀(BRD-K09764130-001-02-6), 归一化到 BRD-K09764130 匹配 SMILES 表
    import re
    def norm_pid(pid):
        if pid.startswith("BRD-"):
            m = re.match(r"(BRD-[A-Z]\d+)(?:-\d+){1,3}", pid)
            if m:
                return m.group(1)
        return pid
    pert_ids_norm = [norm_pid(str(p)) for p in pert_ids]
    is_cp = np.array([p.startswith("BRD-") and p in pid2sm for p in pert_ids_norm])
    cp_idx = np.where(is_cp)[0]
    log(f"trt_cp 有效药物样本 {len(cp_idx)}")

    # 药物词表用归一化 ID
    drug_ids = sorted(set(str(pert_ids_norm[i]) for i in cp_idx))
    drug_vocab = {p: i for i, p in enumerate(drug_ids)}
    fps = np.zeros((len(drug_ids), ECFP4_BITS), dtype=np.float32)
    for i, pid in enumerate(drug_ids):
        sm = pid2sm.get(pid)
        if sm:
            fv = smiles_to_ecfp4(sm)
            if fv is not None and fv.any():
                fps[i] = fv
    zero_fp = ~fps.any(axis=1)
    # 过滤无指纹药物样本
    keep_cp = np.array([not zero_fp[drug_vocab[pert_ids_norm[i]]] for i in cp_idx], dtype=bool)
    cp_idx2 = cp_idx[keep_cp]
    log(f"过滤无指纹后药物样本 {len(cp_idx2)}")

    cl_vals = sorted(set(str(cells[i]) for i in cp_idx2))
    cl_map = {v: i for i, v in enumerate(cl_vals)}
    log(f"细胞系 {len(cl_vals)}")

    n_drug = len(cp_idx2)
    kind = np.ones(n_drug, dtype=np.int8)
    key = np.array([drug_vocab[pert_ids_norm[i]] for i in cp_idx2], dtype=np.int32)
    cell = np.array([cl_map[str(cells[i])] for i in cp_idx2], dtype=np.int32)

    # 5) 基因扰动(pool trt_xpr, 5系) → 映射到全基因列
    pool = ad.read_h5ad(POOL, backed="r")
    op = pool.obs.copy()
    Xp = np.asarray(pool.X[:], dtype=np.float32)
    vp = list(pool.var_names)
    pool.file.close()
    # pool 978 列 → gctx 全基因列(按 Entrez 匹配)
    row_ent = np.array([int(x) for x in rows])
    pool_col_idx, pool_src_idx = [], []
    for j, ent in enumerate(pool_ent):
        hit = np.where(row_ent == ent)[0]
        if len(hit):
            pool_col_idx.append(int(hit[0]))
            pool_src_idx.append(j)
    pool_col_idx = np.array(pool_col_idx, dtype=np.int64)
    pool_src_idx = np.array(pool_src_idx, dtype=np.int64)
    log(f"pool {len(vp)} 基因 → gctx 列映射 {len(pool_col_idx)}")

    xpr_mask = op["pert_type"].astype(str) == "trt_xpr"
    xpr_idx = np.where(xpr_mask.values)[0]
    n_xpr = len(xpr_idx)
    gacc = sorted(op.loc[xpr_mask, "cmap_name"].astype(str).unique())
    gene_vocab = {g: i for i, g in enumerate(gacc)}
    log(f"基因扰动 {n_xpr} 样本 / {len(gacc)} 基因")

    kind2 = np.concatenate([kind, np.zeros(n_xpr, dtype=np.int8)])
    key2 = np.concatenate([key, np.array([gene_vocab.get(str(op["cmap_name"].iloc[i]), -1) for i in xpr_idx], dtype=np.int32)])
    cell2 = np.concatenate([cell, np.array([cl_map.get(str(op["cell_line"].iloc[i]), 0) for i in xpr_idx], dtype=np.int32)])
    valid2 = key2 >= 0
    n_tot = int(valid2.sum())
    log(f"总样本 {n_tot}（药物 {int((kind2[valid2]==1).sum())} + 基因 {int((kind2[valid2]==0).sum())}）")

    # 6) 流式写 y
    y_path = os.path.join(OUT, "y.npy")
    y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32, shape=(n_tot, n_gene))
    ds = f["/0/DATA/0/matrix"]
    # 药物样本(全基因)
    cp_rows = cp_idx2  # 样本在 gctx 中的行号
    B = 4000
    pos = 0
    for s in range(0, len(cp_rows), B):
        bi = cp_rows[s:s + B]
        y[pos:pos + len(bi)] = ds[bi, :]
        pos += len(bi)
        if s % 40000 == 0:
            log(f"  药物 y {pos}/{len(cp_rows)}")
    # 基因扰动样本(pool 978 → 全基因列, 其余 0)
    for k, i in enumerate(xpr_idx):
        if valid2[n_drug + k]:
            y[n_drug + k, pool_col_idx] = Xp[i, pool_src_idx]
        if k % 10000 == 0:
            log(f"  基因 y {k}/{n_xpr}")
    del y
    log(f"y 写入完成 {n_tot}×{n_gene}")

    # 7) 保存 meta
    np.savez(os.path.join(OUT, "meta.npz"),
             kind=kind2[valid2], key=key2[valid2], cell=cell2[valid2],
             cl_names=np.array(cl_vals, dtype=object),
             gene_vocab=np.array(gacc, dtype=object),
             drug_vocab=np.array(drug_ids, dtype=object))
    np.save(os.path.join(OUT, "drug_fps.npy"), fps)
    json.dump(sym_list, open(os.path.join(OUT, "hvg.json"), "w"))
    json.dump([int(x) for x in pool_col_idx], open(os.path.join(OUT, "gene_cols.json"), "w"))
    log(f"✅ 全基因缓存完成 → {OUT}")
    f.close()


if __name__ == "__main__":
    main()
