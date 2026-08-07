"""
LINCS Level5 consensus 签名处理（P4：33,609 药 × 12,327 基因表达签名）
=====================================================================
输入：
  data/lincs/cp_mean_coeff_mat.tsv.gz   每行=一个药物名 + 12327 基因系数（相对对照）
  LINCS_small_molecules.tsv             pert_name → SMILES
输出：
  data/lincs/lincs_train.npz  {names[], smiles[], X[N,12327] float32, genes[]}
  以及数据统计打印。

用法：python unipret/lincs_process.py
"""
import os
import sys
import gzip
import csv
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINCS = os.path.join(BASE, "data", "lincs")
MATRIX = os.path.join(LINCS, "cp_mean_coeff_mat.tsv.gz")
META = os.path.join(BASE, "LINCS_small_molecules.tsv")
OUT = os.path.join(LINCS, "lincs_train.npz")


def main():
    t0 = time.time()
    print(">>> 读 LINCS 药物元数据 (SMILES)...", flush=True)
    meta = {}
    with open(META, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            meta[row["pert_name"]] = row

    print(">>> 流式解析 consensus 矩阵（33.6k 药 × 12327 基因）...", flush=True)
    names, smiles, deltas, genes = [], [], [], None
    with gzip.open(MATRIX, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        genes = header[1:]                       # 12327 基因符号
        n_skip = 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            name = parts[0]
            sm = meta.get(name, {}).get("canonical_smiles")
            if not sm:
                n_skip += 1
                continue
            vals = np.asarray(parts[1:], dtype=np.float32)   # [12327]
            names.append(name)
            smiles.append(sm)
            deltas.append(vals)
    X = np.stack(deltas)
    print(f"    样本 {len(names)}（跳过无 SMILES {n_skip}）| X {X.shape} | "
          f"耗时 {time.time()-t0:.0f}s", flush=True)

    np.savez(OUT, names=np.array(names), smiles=np.array(smiles),
             X=X, genes=np.array(genes))
    print(f"✅ 已存 {OUT}（{os.path.getsize(OUT)//2**20}MB）", flush=True)
    # 快速质检：随机抽 3 个药的签名幅值
    for i in [0, len(names) // 2, len(names) - 1]:
        print(f"    {names[i]:20s} |Δ|均值={np.abs(X[i]).mean():.4f} "
              f"| 基因数={len(genes)}", flush=True)


if __name__ == "__main__":
    main()
