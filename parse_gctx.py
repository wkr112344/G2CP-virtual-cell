# -*- coding: utf-8 -*-
"""parse_gctx.py v2 —— 解析 level5 全细胞系 gctx（本文件结构实测）：
- /0/DATA/0/matrix: 720216 x 12328 float32
- /0/META/COL/id: sig_id，格式 <batch>_<cell>_<pert_type>:<pert_id>:<dose>:<time>
- /0/META/ROW/id: Entrez 基因 ID
输出：与 G2CP pool h5ad 同构的训练数据（978 landmark 基因，var 带 gene_id）。

用法：python parse_gctx.py --gctx "D:/下载/level5_....gctx" [--out data/lincs_extra/all_cells.h5ad]
"""
import sys, os, argparse
import numpy as np
import pandas as pd
import anndata as ad

BASE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(BASE, "data", "g2cp", "data", "LINCS", "pool", "pool_gene_chem_ctrl_adata.h5ad")


def log(m):
    print(f"[parse_gctx] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gctx", required=True)
    ap.add_argument("--out", default=os.path.join(BASE, "data", "lincs_extra", "all_cells.h5ad"))
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()

    import h5py
    if not os.path.isfile(args.gctx):
        log(f"ERROR: 找不到 {args.gctx}"); sys.exit(1)

    # 1) 目标基因：pool 的 978 landmark（gene_id=Entrez → symbol）
    pool = ad.read_h5ad(POOL, backed="r")
    ent2sym = {int(g): s for s, g in zip(pool.var_names, pool.var["gene_id"])}
    pool.file.close()
    log(f"pool 978 landmark 基因（Entrez→symbol 映射）就绪")

    f = h5py.File(args.gctx, "r")
    rows = [x.decode() for x in f["/0/META/ROW/id"][:]]
    row_ent = np.array([int(x) for x in rows])
    rowset = set(row_ent.tolist())
    target_ent = np.array(sorted(ent2sym.keys()))
    pairs = sorted((int(np.where(row_ent == e)[0][0]), e) for e in target_ent if e in rowset)
    col_idx = np.array([p[0] for p in pairs])
    used_ent = [p[1] for p in pairs]
    log(f"目标 Entrez {len(target_ent)}，gctx 中命中 {len(used_ent)} 个（未命中可能不在 12328 内）")
    symbols = [ent2sym[e] for e in used_ent]

    # 2) sig_id 解析：<batch>_<cell>_<pt>:<pert_id>:<dose>:<time>
    sigs = [x.decode() for x in f["/0/META/COL/id"][:]]
    cells, pert_ids = [], []
    for s in sigs:
        head, tail = s.split(":", 1)
        parts = head.split("_")
        cell = parts[1] if len(parts) > 1 else "?"
        pid = tail.split(":")[0]
        cells.append(cell); pert_ids.append(pid)
    cells = np.array(cells); pert_ids = np.array(pert_ids)
    log(f"样本 {len(sigs)} | 细胞系 {len(set(cells))} | 药物 {len(set(pert_ids))}")
    log("细胞系示例: " + ", ".join(sorted(set(cells))[:12]))

    # 3) 分块读取 matrix 目标列 + 分组累加
    mat = f["/0/DATA/0/matrix"]
    sums, cnts = {}, {}
    n_chunk = 0
    nrows = mat.shape[0]
    for s0 in range(0, nrows, args.chunk):
        s1 = min(s0 + args.chunk, nrows)
        X = mat[s0:s1, col_idx].astype(np.float32)
        for j in range(s1 - s0):
            gi = s0 + j
            key = (pert_ids[gi], cells[gi])
            v = X[j]
            if key in sums:
                sums[key] += v; cnts[key] += 1
            else:
                sums[key] = v.copy(); cnts[key] = 1
        n_chunk += 1
        if n_chunk % 4 == 0:
            log(f"  已读 {s1}/{nrows} 样本，{len(sums)} 组（内存 {sums[key].nbytes*len(sums)/1048576:.0f} MB）")
    log(f"读取完成：{n_chunk} 块，{len(sums)} 组")

    # 4) 组装 h5ad
    keys = sorted(sums.keys())
    X = np.stack([sums[k] / cnts[k] for k in keys]).astype(np.float32)
    obs = pd.DataFrame({
        "pert_id": [k[0] for k in keys],
        "cell_line": [k[1] for k in keys],
        "cmap_name": [k[0] for k in keys],   # 药物名先用 pert_id（后续可映射）
        "pert_type": "trt_cp",
        "qc_pass": 1,
    }, index=[f"x{i}" for i in range(len(keys))])
    var = pd.DataFrame({"gene_id": used_ent}, index=symbols)
    a = ad.AnnData(X=X, obs=obs, var=var)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    a.write_h5ad(args.out)
    log(f"✅ 已保存 {args.out}（{a.shape[0]} 样本 × {a.shape[1]} 基因，{len(set(keys))} 细胞系" )
    log(f"   实际细胞系: {len(set(k[1] for k in keys))}")


if __name__ == "__main__":
    main()
