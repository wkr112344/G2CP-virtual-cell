# -*- coding: utf-8 -*-
"""
S28 随机错配负对照 —— 正式结果（方案 B，同一细胞系内置换），方法对照（方案 A）。

药物子集
--------
与 `build_S29_paired_bootstrap.py` **完全一致**的整数 ID 碰撞排除规则：
从 l1000_sdst_78453.h5ad 的 pert_idx → pert_id 一对多关系找出受影响药名并剔除，
1,391 → **1,343** 药物。三套评价矩阵、两种模型、两种错配方案都限定在这同一批药物上，
因此本脚本产出的六个「正常匹配 Pearson」与正文表 4 逐位一致。

校验（脚本会先自证）
--------------------
SDST      XPert 0.61990 | TranSiGen 0.60861
beta2020  XPert 0.03249 | TranSiGen 0.03573
dcic2021  XPert 0.01415 | TranSiGen 0.01441
与正文表 4 不符则报错退出，不产出 CSV。

两种错配方案
------------
方案 A（组内位置错配，方法对照）
    先按药物分组，组内记录按位置错配到另一药物组（行对行）。
    不同药物的记录所覆盖的细胞系未必相同 → **细胞系会随行一起改变**。

方案 B（同一细胞系内置换，正式结果）
    置换分组条件 = (细胞系, 模型, 测试折)。组内改用**其他药物**的 978 基因参考向量。
    细胞系与折固定不变，只有药物配对变化。
    **单记录分组**（组内只有一条记录、无别的药物可换）保持原记录不变。
    该规则在输出 CSV 的 `n_single_record_groups` 列中如实记录。

输出
    S28_permutation_scheme_comparison.csv
      scheme, evaluation_matrix, model, observed_pearson,
      null_mean, null_ci95_low, null_ci95_high, observed_exceeds_ci,
      n_single_record_groups, n_drugs, B, seed
"""
import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from config_paths import P, require

warnings.filterwarnings("ignore")

MODELS = ("xpert", "transigen")
FOLDS = range(5)
B = 2000
SEED = 20261011

# 与正文表 4 逐位一致的期望值（用于自证药物子集对齐）
EXPECTED = {
    ("SDST", "XPert"): 0.61990, ("SDST", "TranSiGen"): 0.60861,
    ("beta2020", "XPert"): 0.03249, ("beta2020", "TranSiGen"): 0.03573,
    ("dcic2021", "XPert"): 0.01415, ("dcic2021", "TranSiGen"): 0.01441,
}
TOL = 5e-5


def cat_col(f, name):
    g = f["obs"][name]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
    codes = g["codes"][:]
    out = np.empty(len(codes), dtype=object)
    ok = codes >= 0
    out[ok] = cats[codes[ok]]
    out[~ok] = None
    return out


def pcc_rows(Pm, Y):
    Pc = Pm - Pm.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return (Pc * Yc).sum(axis=1) / (np.sqrt((Pc ** 2).sum(axis=1) * (Yc ** 2).sum(axis=1)) + 1e-12)


def main():
    require(P.h5ad, P.genemap, P.cache_modz / "pairs_meta.npz", P.cache_cd / "pairs_meta.npz")

    internal_col = np.array(json.load(open(P.genemap, encoding="utf-8"))["internal_col"])

    # ---------------- 药物编号碰撞剔除（与 S29 同一规则）----------------
    with h5py.File(P.h5ad, "r") as f:
        pi = f["obs"]["pert_idx"][:]
        pid = cat_col(f, "pert_id")
        cc = f["obs"]["cell_idx"]["codes"][:]
        cin = cat_col(f, "cell_iname")
    smap = pd.DataFrame({"idx": pi, "name": pid}).drop_duplicates()
    bad_idx = set(smap.groupby("idx").name.nunique().pipe(lambda x: x[x > 1]).index.tolist())
    badnames = set(smap[smap.idx.isin(bad_idx)].name.unique())
    print("整数 ID 碰撞涉及药名：%d 个（剔除后应剩 1,343）" % len(badnames))

    i2d = dict(zip(pi.tolist(), pid.tolist()))
    i2c = dict(zip(cc.tolist(), cin.tolist()))

    cache = {}
    for tag, d in (("MODZ", P.cache_modz), ("CD", P.cache_cd)):
        m = np.load(d / "pairs_meta.npz", allow_pickle=True)
        key2row = {(str(c), str(dd)): i for i, (c, dd) in
                   enumerate(zip(m["cell_name"], m["drug_id"]))}
        cache[tag] = {"key2row": key2row, "y": np.load(d / "y.npy", mmap_mode="r")}

    blocks = []
    for mm in MODELS:
        for fo in FOLDS:
            d = P.depmap_bench / "results/main_benchmark" / mm / ("fold_%d" % fo) / "predictions"
            require(d / "test_predictions.npy", d / "test_ground_truth.npy", d / "test_sample_ids.csv")
            Pm = np.load(d / "test_predictions.npy").astype(np.float32)
            GT = np.load(d / "test_ground_truth.npy").astype(np.float32)
            ids = pd.read_csv(d / "test_sample_ids.csv")
            keys, sel, gb, gd = [], [], [], []
            for k, (dv, cv) in enumerate(zip(ids.drug_id, ids.cell_id)):
                dd, c2 = i2d.get(int(dv)), i2c.get(int(cv))
                if dd and c2 and dd not in badnames:          # ← 关键：同一批次剔除
                    rb = cache["MODZ"]["key2row"].get((str(c2), str(dd)))
                    rd = cache["CD"]["key2row"].get((str(c2), str(dd)))
                    if rb is not None and rd is not None:
                        keys.append((str(c2), str(dd)))
                        sel.append(k)
                        gb.append(rb)
                        gd.append(rd)
            if not sel:
                continue
            sel = np.array(sel)
            gb, gd = np.array(gb), np.array(gd)
            blocks.append({
                "model": mm, "fold": fo,
                "drug": np.array([keys[i][1] for i in range(len(keys))], dtype=object),
                "cell": np.array([keys[i][0] for i in range(len(keys))], dtype=object),
                "P": Pm[sel], "SDST": GT[sel],
                "MODZ": np.asarray(cache["MODZ"]["y"][np.ix_(gb, internal_col)], dtype=np.float32),
                "CD": np.asarray(cache["CD"]["y"][np.ix_(gd, internal_col)], dtype=np.float32),
            })

    def flat(field):
        parts = []
        for r in blocks:
            v = r[field]
            nb = len(r["drug"])
            if isinstance(v, str):
                parts.append(np.array([v] * nb, dtype=object))
            elif np.ndim(v) == 0:
                parts.append(np.full(nb, v))
            else:
                parts.append(np.asarray(v))
        return np.concatenate(parts)

    M, F, D, C = flat("model"), flat("fold"), flat("drug"), flat("cell")
    S_all = np.concatenate([r["SDST"] for r in blocks])
    Bz_all = np.concatenate([r["MODZ"] for r in blocks])
    Cd_all = np.concatenate([r["CD"] for r in blocks])
    P_all = np.concatenate([r["P"] for r in blocks])

    n_row = len(D)
    drugs = np.unique(D)
    n_drug = len(drugs)
    print("统一口径：%d 行 / %d 药物" % (n_row, n_drug))
    if n_drug != 1343:
        raise SystemExit("药物数 %d ≠ 1343，未与正文表 4 对齐，停止。" % n_drug)

    prev = {"xpert": np.where(M == "xpert")[0], "transigen": np.where(M == "transigen")[0]}

    def drug_equal(ser):
        dfx = pd.DataFrame({"model": M, "drug": D, "fold": F, "r": ser})
        per = (dfx.groupby(["model", "drug", "fold"])["r"].mean().reset_index()
                  .groupby(["model", "drug"])["r"].mean().reset_index())
        return per.pivot(index="drug", columns="model", values="r").dropna()

    def observed(ref):
        ser = np.empty(n_row)
        for m in MODELS:
            ser[prev[m]] = pcc_rows(P_all[prev[m]], ref[prev[m]])
        piv = drug_equal(ser)
        return {c: float(piv[c].mean()) for c in piv.columns}

    obs_eq = {"SDST": observed(S_all), "MODZ": observed(Bz_all), "CD": observed(Cd_all)}

    # ---------------- 自证：六个正常匹配值 == 正文表 4 ----------------
    mat_name = {"SDST": "SDST", "MODZ": "beta2020", "CD": "dcic2021"}
    bad = []
    print("\n正常匹配 Pearson 与正文表 4 对照：")
    for nm in ("SDST", "MODZ", "CD"):
        for m in MODELS:
            lab = "XPert" if m == "xpert" else "TranSiGen"
            got = obs_eq[nm][m]
            exp = EXPECTED[(mat_name[nm], lab)]
            ok = abs(got - exp) < TOL
            print("  %-9s %-9s 复算 %.5f | 表4 %.5f | %s"
                  % (mat_name[nm], lab, got, exp, "一致" if ok else "不一致"))
            if not ok:
                bad.append((mat_name[nm], lab, got, exp))
    if bad:
        raise SystemExit("正常匹配值与正文表 4 不一致，停止，不产出 CSV：%s" % bad)
    print("→ 六个正常匹配值全部与正文表 4 一致，可继续。\n")

    # ---------------- 方案 A：组内位置错配 ----------------
    rng = np.random.default_rng(SEED)
    order = np.argsort(D, kind="stable")
    Ds = D[order]
    gstart = np.searchsorted(Ds, drugs, "left")
    gend = np.searchsorted(Ds, drugs, "right")
    glen = gend - gstart
    maxlen = int(glen.max())
    gpos = np.zeros((n_drug, maxlen), dtype=np.int64)
    for g in range(n_drug):
        rows = order[gstart[g]:gend[g]]
        gpos[g, :len(rows)] = rows
        gpos[g, len(rows):] = rows[0]
    valid = np.arange(maxlen)[None, :] < glen[:, None]

    # ---------------- 方案 B：同细胞系内置换 ----------------
    grp_key = np.array([f"{C[i]}|{M[i]}|{F[i]}" for i in range(n_row)], dtype=object)
    gk = pd.factorize(grp_key)[0]
    grp_members = [[] for _ in range(gk.max() + 1)]
    for i in range(n_row):
        grp_members[gk[i]].append(i)
    grp_arr = [np.array(v, dtype=np.int64) for v in grp_members]
    single = [v for v in grp_arr if len(v) == 1]
    n_single = len(single)
    print("方案 B 分组：共 %d 组，组内平均 %.2f 条，最大 %d 条；单记录分组 %d 组（保持原记录不变）"
          % (len(grp_arr), n_row / len(grp_arr), max(len(v) for v in grp_arr), n_single))

    def permute_A():
        perm = rng.permutation(n_drug)
        src = gpos[perm]
        newidx = np.empty(n_row, dtype=np.int64)
        newidx[gpos[valid]] = src[valid]
        return newidx

    def permute_B():
        newidx = np.empty(n_row, dtype=np.int64)
        for v in grp_arr:
            k = len(v)
            if k == 1:
                newidx[v[0]] = v[0]              # 单记录分组：保持原记录
            else:
                p = rng.permutation(k)
                if np.any(p == np.arange(k)):    # 保证是「错配」而非原地不动
                    p = np.roll(np.arange(k), 1)
                newidx[v] = v[p]
        return newidx

    rows_out = []
    for scheme, permute_fn in (("A_positional_old", permute_A), ("B_within_cell", permute_B)):
        null = {nm: {m: np.empty(B) for m in MODELS} for nm in ("SDST", "MODZ", "CD")}
        for b in range(B):
            newidx = permute_fn()
            for nm, ref in (("SDST", S_all), ("MODZ", Bz_all), ("CD", Cd_all)):
                rsh = ref[newidx]
                ser = np.empty(n_row)
                for m in MODELS:
                    ser[prev[m]] = pcc_rows(P_all[prev[m]], rsh[prev[m]])
                piv = drug_equal(ser)
                for m in MODELS:
                    null[nm][m][b] = float(piv[m].mean())
        for nm in ("SDST", "MODZ", "CD"):
            for m in MODELS:
                nl = null[nm][m]
                lo, hi = float(np.percentile(nl, 2.5)), float(np.percentile(nl, 97.5))
                o = obs_eq[nm][m]
                rows_out.append({
                    "scheme": scheme,
                    "evaluation_matrix": nm,
                    "model": "XPert" if m == "xpert" else "TranSiGen",
                    "observed_pearson": round(o, 5),
                    "null_mean": round(float(nl.mean()), 5),
                    "null_ci95_low": round(lo, 5),
                    "null_ci95_high": round(hi, 5),
                    "observed_exceeds_ci": bool(o > hi),
                    "n_single_record_groups": n_single,
                    "n_drugs": n_drug,
                    "B": B,
                    "seed": SEED,
                })
                print("%-16s %-9s %-9s 观察 %+.5f | 零均值 %+.5f | 零CI95 [%+.5f,%+.5f] | 超出 %s"
                      % (scheme, nm, m, o, nl.mean(), lo, hi, "是" if o > hi else "否"))

    out = P.out / "S28_permutation_scheme_comparison.csv"
    pd.DataFrame(rows_out).to_csv(out, index=False, encoding="utf-8-sig", lineterminator="\n")
    print("\n写入:", out, "| 药物 %d | B=%d seed=%d | 单记录分组 %d" % (n_drug, B, SEED, n_single))


if __name__ == "__main__":
    main()
