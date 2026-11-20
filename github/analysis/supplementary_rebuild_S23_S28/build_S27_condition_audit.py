# -*- coding: utf-8 -*-
"""
S27 条件核查 —— 同一药物、同一细胞系下，比较 SDST 与 beta2020 的
处理时间、药物浓度与原始实验孔位编号（well token）。

被正文/补充材料引用的四个数字
-----------------------------
    SDST 记录为 24 h / 10 µM 且带孔位编号：             78,453 / 78,453   (100%)
    beta2020 中存在同药物/同细胞系/同时间/同浓度的签名：  78,453 / 78,453   (100%)
    SDST 原始孔位编号可在 beta2020 的 distil_ids 中找到： 219,010 / 219,066 (99.97%)
    已命中孔位的处理时间与浓度一致：                     218,953 / 219,010 (99.97%)

方法
----
SDST 侧直接取自 `l1000_sdst_78453.h5ad` 的 obs 字段：
  - `cell_drug_dose_time` = "CELL:DRUG:24 h:10.0"（本数据集唯一取值）
  - `sample_id`            = "PLATE:WELL;PLATE:WELL;..."（原始实验孔位编号 token）

beta2020 侧取自 LINCS level-5 签名元数据 `siginfo_beta.txt`
（GSE92742；取 `pert_type == trt_cp` 的行）：
  - `distil_ids` 提供原始实验孔位编号 token（与 SDST 的 PLATE:WELL 同构）
  - `(cell_iname, pert_id, pert_time, pert_dose, pert_dose_unit)` 用于条件比对

配对率的分母为 SDST 侧全部孔位 token 数（219,066），分子为能在 beta2020
`distil_ids` 中命中的 token 数（219,010）。第 2、4 项在 (cell, drug) 层比较。

输入
----
    P.h5ad                        LINCS SDST 处理后表达矩阵（随 benchmark 仓库提供）
    <external>/siginfo_beta.txt   beta2020 level-5 签名元数据（约 465 MB，需自行下载；
                                  见 README「数据从哪里来」），或用 --siginfo 指定

输出
    S27_sdst_beta2020_condition_audit.csv —— item / matched / total / ratio
"""
import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from config_paths import P, require

warnings.filterwarnings("ignore")

SIGINFO_CANDS = ("siginfo_beta.txt", "siginfo_beta (1).txt")
TIME_TOKENS = ("24", "24 h", "24h", "24.0")


def str_col(f, name):
    return np.array([s.decode() if isinstance(s, bytes) else s for s in f["obs"][name][:]],
                    dtype=object)


def find_siginfo():
    for nm in SIGINFO_CANDS:
        p = P.external / nm
        if p.exists():
            return p
    return P.external / SIGINFO_CANDS[0]


def dose_num(s):
    try:
        return float(s)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siginfo", default=str(find_siginfo()),
                    help="beta2020 level-5 签名元数据 siginfo_beta.txt 路径")
    a = ap.parse_args()
    siginfo = Path(a.siginfo)

    require(P.h5ad, siginfo)

    # ------------------------------------------------- SDST 侧
    with h5py.File(P.h5ad, "r") as f:
        cdt = str_col(f, "cell_drug_dose_time")
        sid = str_col(f, "sample_id")

    rows = []
    for c, s in zip(cdt, sid):
        cell, drug, t, d = c.split(":")[0], c.split(":")[1], c.split(":")[2], c.split(":")[3]
        rows.append({"cell": cell, "drug": drug, "time": t, "dose": d,
                     "tokens": [r for r in s.split(";") if ":" in r]})
    n_sdst = len(rows)
    n_tokens = sum(len(r["tokens"]) for r in rows)
    print("SDST 行 %d；(cell,drug,time,dose) 唯一组合 %d；孔位 token 总数 %d"
          % (n_sdst, len({(r["cell"], r["drug"], r["time"], r["dose"]) for r in rows}), n_tokens))
    print("SDST 时间取值 %s | 浓度取值 %s"
          % (sorted({r["time"] for r in rows}), sorted({r["dose"] for r in rows})))

    # ------------------------------------------------- beta2020 侧
    with open(siginfo, encoding="utf-8", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        need = ("sig_id", "distil_ids", "pert_dose", "pert_dose_unit",
                "pert_time", "pert_time_unit", "pert_id", "cell_iname", "pert_type")
        idx = {k: hdr.index(k) for k in need}
        n_trt = 0
        beta_tokens = set()
        beta_token_info = {}
        beta_keys = defaultdict(list)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr) or p[idx["pert_type"]] != "trt_cp":
                continue
            n_trt += 1
            cell, drug = p[idx["cell_iname"]], p[idx["pert_id"]]
            t, d, u, sig = (p[idx["pert_time"]], p[idx["pert_dose"]],
                            p[idx["pert_dose_unit"]], p[idx["sig_id"]])
            for tok in p[idx["distil_ids"]].split("|"):
                if tok:
                    beta_tokens.add(tok)
                    beta_token_info[tok] = (t, d, u)
            beta_keys[(cell, drug)].append((t, d, u, sig))
    print("beta2020 trt_cp 签名 %d；distil 孔位 token 唯一数 %d" % (n_trt, len(beta_tokens)))

    # ------------------------------- 逐项比对（与正文引用的四个数字对应）
    sdst_pairs = {(r["cell"], r["drug"]) for r in rows}
    common = sdst_pairs & set(beta_keys.keys())
    print("SDST 唯一 (cell,drug) 对 %d；beta2020 %d；交集 %d"
          % (len(sdst_pairs), len(beta_keys), len(common)))

    cond_ok = sum(1 for r in rows
                  if r["time"].strip() in TIME_TOKENS
                  and (dose_num(r["dose"]) or -1) == 10.0
                  and r["tokens"])
    paired = sum(1 for r in rows
                 if (r["cell"], r["drug"]) in common
                 and any(t == "24" and u == "uM" and dose_num(d) == 10.0
                         for (t, d, u, _) in beta_keys[(r["cell"], r["drug"])]))
    tok_total = tok_hit = tok_hit_cond = 0
    for r in rows:
        for tok in r["tokens"]:
            tok_total += 1
            if tok in beta_tokens:
                tok_hit += 1
                t, d, u = beta_token_info[tok]
                if t == "24" and u == "uM" and dose_num(d) == 10.0:
                    tok_hit_cond += 1

    out_rows = [
        {"item": "SDST records with 24 h treatment time and 10 uM concentration",
         "matched": cond_ok, "total": n_sdst, "ratio": round(cond_ok / n_sdst, 5)},
        {"item": "beta2020 signatures with identical drug/cell line/time/concentration",
         "matched": paired, "total": n_sdst, "ratio": round(paired / n_sdst, 5)},
        {"item": "SDST original experimental well IDs found in beta2020",
         "matched": tok_hit, "total": tok_total, "ratio": round(tok_hit / tok_total, 5)},
        {"item": "Matched wells with consistent treatment time and concentration",
         "matched": tok_hit_cond, "total": tok_hit, "ratio": round(tok_hit_cond / tok_hit, 5)},
    ]
    for r in out_rows:
        print("  %-62s %s/%s  rate=%s"
              % (r["item"], r["matched"], r["total"], r["ratio"]))

    out = P.out / "S27_sdst_beta2020_condition_audit.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", "matched", "total", "ratio"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print("\n写入:", out)

    EXPECTED = [(78453, 78453), (78453, 78453), (219010, 219066), (218953, 219010)]
    got = [(r["matched"], r["total"]) for r in out_rows]
    if got != EXPECTED:
        print("\n[自证失败] 输出与正文引用数字不一致：")
        for (g, e), r in zip(zip(got, EXPECTED), out_rows):
            if g != e:
                print("   %s: 得到 %s，应为 %s" % (r["item"], g, e))
        raise SystemExit(1)
    print("\n[自证通过] 四项与正文引用数字逐位一致。")


if __name__ == "__main__":
    main()
