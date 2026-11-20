# -*- coding: utf-8 -*-
"""
S25 疾病签名的 UP/DN 基因数 —— 正确口径：12,328 个共同基因（与评分脚本一致）。

口径来源（决定性）
------------------
评分脚本 `434_disease_reversal.py` 的做法是：
    hidx = {g: i for i, g in enumerate(cache['hvg'])}
    up = [hidx[g] for g in gmt[s+'_UP'] if g in hidx]
    dn = [hidx[g] for g in gmt[s+'_DN'] if g in hidx]

其中 `hvg` 来自 `g2cp_cache_*/meta.npz` 的 `hvg` 字段 —— 即 **12,328 个共同基因的
原始符号列表（两套参考完全一致），不做 HGNC 规范化**。

因此本脚本直接用该 hvg 列表统计，而不是：
  - 978 landmark 子集（口径过窄，早期版本误用）；
  - geneinfo_beta.txt + HGNC 规范化（会引入 alias/previous 符号，导致轴大小变化）。

结果（与正文及核查一致）
    Liver   UP 679 / DN 236
    Thyroid UP 573 / DN 392
    APL     UP 156 / DN 656

输入（external/，见 README）
    g2cp_cache_beta_v2/meta.npz   （hvg 字段 = 12,328 共同基因）
    msigdb/c2.cgp.v2023.2.Hs.symbols.gmt

输出
    S25_disease_signature_gene_counts.csv
"""
import csv
import io
import warnings

import numpy as np

from config_paths import P, require

warnings.filterwarnings("ignore")

SIGS = [
    ("Liver", "ACEVEDO_LIVER_TUMOR_VS_NORMAL_ADJACENT_TISSUE"),
    ("Thyroid", "RODRIGUES_THYROID_CARCINOMA_ANAPLASTIC"),
    ("APL", "CASORELLI_ACUTE_PROMYELOCYTIC_LEUKEMIA"),
]


def main():
    require(P.cache_modz / "meta.npz", P.msigdb_gmt)

    # 12,328 共同基因（两套参考的 hvg 完全一致，取其一即可）
    meta = np.load(P.cache_modz / "meta.npz", allow_pickle=True)
    hvg = [str(x) for x in meta["hvg"]]
    axis = set(hvg)
    print("12,328 共同基因轴：hvg 长度 %d，唯一符号 %d" % (len(hvg), len(axis)))
    if len(axis) != 12328:
        raise SystemExit("hvg 轴不是 12,328，口径不符，停止。")

    gmt = {}
    for ln in io.open(P.msigdb_gmt, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > 2:
            gmt[p[0]] = p[2:]

    rows = []
    for short, s in SIGS:
        gu = [g for g in gmt[s + "_UP"] if g in axis]
        gd = [g for g in gmt[s + "_DN"] if g in axis]
        rows.append({
            "disease_signature": short,
            "msigdb_set": s,
            "up_genes_total": len(gmt[s + "_UP"]),
            "dn_genes_total": len(gmt[s + "_DN"]),
            "up_genes": len(gu),
            "dn_genes": len(gd),
        })
        print("%-9s 原始 UP/DN %d/%d → 12,328 轴内 UP/DN %d/%d"
              % (short, len(gmt[s + "_UP"]), len(gmt[s + "_DN"]), len(gu), len(gd)))

    out = P.out / "S25_disease_signature_gene_counts.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("\n写入:", out)


if __name__ == "__main__":
    main()
