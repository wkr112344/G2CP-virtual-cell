# -*- coding: utf-8 -*-
"""
S29 / 正文 p = 0.0090 的完整计算链条与统计公式。

正文第 11 页「MODZ − CD 优势差 −0.00298，95% CI −0.00510–−0.00088，p = 0.0090」
此前只有 1,391 全药物路径的脚本（score_three_refs.py，得 p = 0.002），
本脚本补上剔除整数 ID 碰撞后的 1,343 药物路径，使该 p 值可被逐位复现。

统计公式
--------
逐行得分      r = Pearson(模型预测 978 维, 参考 978 维)          （逐细胞系）
聚合          step1 按 (model, drug, fold) 取均值
              step2 按 (model, drug) 取均值                    （药物等权）
优势          A_ref[drug] = r_ref[drug, XPert] − r_ref[drug, TranSiGen]
优势差        Δ_obs = mean_drug( A_ref1 − A_ref2 )
配对 bootstrap 第 b 次：idx_b ~ Uniform{0..n−1}（有放回，两侧同一组索引）
              d_b = mean(A_ref1[idx_b]) − mean(A_ref2[idx_b])
95% 范围      [ percentile(d, 2.5), percentile(d, 97.5) ]
双侧 p        p = min( 1, 2 · min( #{d ≤ 0}, #{d > 0} ) / B )

B = 2000，seed = 20261011；每对比较各自重置随机流（与正文归档口径一致）。
B = 2000 时 p 的最小可分辨值约为 2/2000 = 0.001；较小一侧计数为 0 时按该
下限报告 0.001（bootstrap 经验 P 值不可能为 0）。

输入
----
逐行得分表需先由 score_three_refs.py 产出（列：model, fold, cell, drug, r_sdst, r_modz, r_cd）。
默认从 P.out / "score_three_refs_by_drug.csv" 读取，或用 --by 指定。

输出
    S29_paired_advantage_bootstrap.csv   三行（SDST−MODZ / SDST−CD / MODZ−CD）
    p00090_calculation_chain.csv         含 bootstrap 落在零两侧的计数
"""
import argparse
import csv
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from config_paths import P, require

warnings.filterwarnings("ignore")

B, SEED = 2000, 20261011


def cat_col(f, n):
    g = f["obs"][n]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
    k = g["codes"][:]
    o = np.empty(len(k), dtype=object)
    ok = k >= 0
    o[ok] = cats[k[ok]]
    o[~ok] = None
    return o


def p2(d, null=0.0):
    """双侧经验 P 值。

    取 bootstrap 差值分布中位于零两侧的较小比例乘以 2：
        p = min(1, 2 * min(#{d <= null}, #{d > null}) / B)

    B = 2000 时经验分辨率下限为 2 / B = 0.001；若较小一侧计数为 0，
    按分辨率下限报告 0.001（而非 0），避免出现「bootstrap P 值为 0」的
    不可分辨表述。
    """
    d = np.asarray(d)
    n_le = int((d <= null).sum())
    n_gt = int((d > null).sum())
    B = len(d)
    k = min(n_le, n_gt)
    if k == 0:
        return 2.0 / B
    return float(min(1.0, 2.0 * k / B))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", default=str(P.out / "score_three_refs_by_drug.csv"))
    a = ap.parse_args()
    by = Path(a.by)
    require(P.h5ad, by)

    with h5py.File(P.h5ad, "r") as f:
        pi = f["obs"]["pert_idx"][:]
        pid = cat_col(f, "pert_id")
    s = pd.DataFrame({"idx": pi, "name": pid}).drop_duplicates()
    bad = set(s.groupby("idx").name.nunique().pipe(lambda x: x[x > 1]).index.tolist())
    badnames = set(s[s.idx.isin(bad)].name.unique())
    print("整数 ID 碰撞涉及药名：%d 个" % len(badnames))

    df = pd.read_csv(by)
    n_all = df.drug.nunique()
    df = df[~df.drug.isin(badnames)].copy()
    per = (df.groupby(["model", "drug", "fold"])[["r_sdst", "r_modz", "r_cd"]].mean()
             .reset_index().groupby(["model", "drug"])[["r_sdst", "r_modz", "r_cd"]].mean()
             .reset_index())
    piv = per.pivot(index="drug", columns="model", values=["r_sdst", "r_modz", "r_cd"]).dropna()
    n = len(piv.index)
    print("全药物 %d → 剔除碰撞后 %d" % (n_all, n))
    A = {r: piv[(r, "xpert")].to_numpy() - piv[(r, "transigen")].to_numpy()
         for r in ("r_sdst", "r_modz", "r_cd")}

    print("公式 p = min(1, 2 × min(#{d≤0}, #{d>0}) / B), B=%d, seed=%d" % (B, SEED))
    main_rows, chain_rows = [], []
    for r1, r2, nm in (("r_sdst", "r_modz", "SDST-MODZ"),
                       ("r_sdst", "r_cd", "SDST-CD"),
                       ("r_modz", "r_cd", "MODZ-CD")):
        obs = float(A[r1].mean() - A[r2].mean())
        rng = np.random.default_rng(SEED)
        d = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, n)
            d[b] = A[r1][idx].mean() - A[r2][idx].mean()
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        n_le, n_gt = int((d <= 0).sum()), int((d > 0).sum())
        p = p2(d)
        print("  %-9s obs %+.5f  CI [%+.5f,%+.5f]  #(d<=0)=%d #(d>0)=%d  p=%.4f"
              % (nm, obs, lo, hi, n_le, n_gt, p))
        main_rows.append({"advantage_difference": nm, "observed": round(obs, 5),
                          "ci95_low": round(lo, 5), "ci95_high": round(hi, 5),
                          "p_two_sided": round(p, 4), "B": B, "seed": SEED, "n_drugs": n})
        chain_rows.append({"advantage_difference": nm, "observed": round(obs, 5),
                           "ci95_low": round(lo, 5), "ci95_high": round(hi, 5),
                           "n_boot_le0": n_le, "n_boot_gt0": n_gt, "p_two_sided": round(p, 4),
                           "formula": "p = min(1, 2*min(#{d<=0},#{d>0})/B)",
                           "B": B, "seed": SEED, "n_drugs": n})

    for name, rows in (("S29_paired_advantage_bootstrap.csv", main_rows),
                       ("p00090_calculation_chain.csv", chain_rows)):
        out = P.out / name
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print("写入:", out)


if __name__ == "__main__":
    main()
