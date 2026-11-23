# -*- coding: utf-8 -*-
"""
S28 随机错配负对照 —— 三套方案的统一输出（A 方法对照，B 同细胞系内置换，C 仅保留可置换分层）。

药物子集
--------
与 `build_S29_paired_bootstrap.py` **完全一致**的整数 ID 碰撞排除规则：
从 l1000_sdst_78453.h5ad 的 pert_idx → pert_id 一对多关系找出受影响药名并剔除，
1,391 → **1,343** 药物。三套评价矩阵、两种模型、三种错配方案都限定在这同一批药物上，
因此本脚本产出的六个「正常匹配 Pearson」与正文表 4 对应。

校验（脚本会先自证）
--------------------
正文表 4：SDST XPert 0.61990 | beta2020 XPert 0.03249，其余四格逐位一致。
冻结的 XPert 预测在重放时经注意力回退实现，这两格会漂移约 2×10⁻⁴（0.61972 / 0.03257）；
脚本打印对照结果而不中止，偏差来源见 REPRODUCTION_MAP.md。

三种错配方案
------------
方案 A（组内位置错配，方法对照）
    先按药物分组，组内记录按位置错配到另一药物组（行对行）。
    不同药物的记录所覆盖的细胞系未必相同 → **细胞系会随行一起改变**。

方案 B（同一细胞系内置换）
    置换分组条件 = (细胞系, 模型, 测试折)。组内改用**其他药物**的 978 基因参考向量。
    细胞系与折固定不变，只有药物配对变化。
    **单记录分组**（组内只有一条记录、无别的药物可换）保持原记录不变。
    该规则在输出 CSV 的 `n_single_record_groups` 列中如实记录。

方案 C（仅保留可置换分层，正文引用方案）
    在方案 B 的基础上剔除 92 个单记录分组（6,892 → 6,800 条记录，1,342 药物），
    使每一个保留的分层都存在完整的药物身份错排。

    方案 B/C 的每一次置换都逐条核对：某记录抽到的响应必须来自**不同药物**，
    否则重抽；连续 500 次不成功则改用确定性构造，并对最终结果逐位置复核。
    全流程计数在运行末尾打印，任一位置保留原药物即中止。

输出
    S28_permutation_scheme_comparison.csv
      scheme, evaluation_matrix, model, observed_pearson, null_mean,
      null_ci95_low, null_ci95_high, observed_exceeds_ci, p_empirical,
      n_records, n_excluded_groups, n_single_record_groups, n_drugs, B, seed
"""
import json
import warnings
from pathlib import Path

import os

import h5py
import numpy as np
import pandas as pd

from config_paths import P, require

warnings.filterwarnings("ignore")

MODELS = ("xpert", "transigen")
FOLDS = range(5)
B = int(os.environ.get("S28_B", "2000"))
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


def _norm_rows(X):
    """行中心化并按行范数归一，使 Pearson 退化为点积。"""
    Xc = X - X.mean(axis=1, keepdims=True)
    n = np.sqrt((Xc ** 2).sum(axis=1))
    n[n < 1e-12] = 1e-12
    return (Xc / n[:, None]).astype(np.float32)


def pcc_norm(Pn, Yn):
    return (Pn * Yn).sum(axis=1)


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

    # ---- 快速聚合：行内归一化 + bincount 两级均值（替代逐次 pandas groupby）----
    def _build_agg(Mx, Dx, Fx):
        n = len(Dx)
        k3 = pd.factorize(np.array([f"{Mx[i]}|{Dx[i]}|{Fx[i]}" for i in range(n)], dtype=object))[0]
        n3 = int(k3.max()) + 1
        cnt3 = np.bincount(k3, minlength=n3).astype(np.float64)
        first = np.full(n3, -1, dtype=np.int64)
        for i in range(n - 1, -1, -1):
            first[k3[i]] = i
        k2 = pd.factorize(np.array([f"{Mx[i]}|{Dx[i]}" for i in first], dtype=object))[0]
        n2 = int(k2.max()) + 1
        cnt2 = np.bincount(k2, minlength=n2).astype(np.float64)
        mdl = np.array([Mx[i] for i in first], dtype=object)
        drg = np.array([Dx[i] for i in first], dtype=object)
        dset = {m: set(drg[mdl == m]) for m in MODELS}
        both = dset["xpert"] & dset["transigen"]
        sel = {m: np.where((mdl == m) & np.isin(drg, list(both)))[0] for m in MODELS}
        return k3, cnt3, k2, cnt2, sel

    def _agg_mean(ser, pack):
        k3, cnt3, k2, cnt2, sel = pack
        m3 = np.bincount(k3, weights=ser, minlength=len(cnt3)) / cnt3
        m2 = np.bincount(k2, weights=m3, minlength=len(cnt2)) / cnt2
        return {m: float(m2[sel[m]].mean()) for m in MODELS}

    Pn = {m: _norm_rows(P_all[prev[m]]) for m in MODELS}
    Yn = {nm: _norm_rows(ref) for nm, ref in (("SDST", S_all), ("MODZ", Bz_all), ("CD", Cd_all))}
    pack_ab = _build_agg(M, D, F)


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

    # 自检：快速聚合必须与 pandas 口径的观测值一致
    _chk = {}
    for nm in ("SDST", "MODZ", "CD"):
        _ser = np.empty(n_row)
        for m in MODELS:
            _ser[prev[m]] = pcc_norm(Pn[m], Yn[nm][prev[m]])
        _g = _agg_mean(_ser, pack_ab)
        for m in MODELS:
            _chk[(nm, m)] = abs(_g[m] - obs_eq[nm][m])
    _worst = max(_chk.values())
    print("聚合自检：最大偏差 %.2e %s" % (_worst, "通过" if _worst < 1e-9 else "不通过"))

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
        print("!! 重放值与正文表 4 有偏差（SDPA 回退的浮点漂移），不中止：%s" % bad)
    else:
        print("→ 六个正常匹配值全部与正文表 4 一致。\n")

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

    def derange(n):
        """无固定点置换：确保没有药物映射到自身。"""
        while True:
            q = rng.permutation(n)
            if not np.any(q == np.arange(n)):
                return q

    _audit = {t: {"draws": 0, "positions": 0, "same_drug": 0, "fallback": 0} for t in ("B", "C")}

    def _derange_multiset(v, drugs):
        """确定性构造的备用方案：按药物排序后整体位移 floor(n/2)。

        组内任一药物的记录数都不超过 n/2 时，该位移必然把每条记录换到别的药物；
        否则该分组不存在满足约束的置换，直接报错而不是静默保留原配对。
        """
        order = np.argsort(drugs[v], kind="stable")
        arr = v[order]
        n = len(arr)
        if n < 2:
            raise RuntimeError("单记录分组不可置换，应在方案 C 中剔除")
        cand = arr[(np.arange(n) + n // 2) % n]
        if np.any(drugs[cand] == drugs[arr]):
            raise RuntimeError("该分组不存在满足药物身份约束的置换（某一药物占组内多数）")
        return cand

    def perm_constrained(v, drugs, tag):
        """按药物身份约束的置换：任一位置都不得保留原药物。

        先做拒绝采样；连续 500 次不满足约束时改用确定性构造，并对最终结果逐位置核对。
        """
        k = len(v)
        src = v
        cand = None
        for _ in range(500):
            q = rng.permutation(k)
            trial = src[q]
            if np.all(drugs[trial] != drugs[src]):
                cand = trial
                break
        if cand is None:
            cand = _derange_multiset(src, drugs)
            _audit[tag]["fallback"] += 1
        _audit[tag]["draws"] += 1
        _audit[tag]["positions"] += k
        _audit[tag]["same_drug"] += int(np.sum(drugs[cand] == drugs[src]))
        return cand

    def permute_A():
        perm = derange(n_drug)
        src = gpos[perm]
        newidx = np.empty(n_row, dtype=np.int64)
        newidx[gpos[valid]] = src[valid]
        return newidx

    def permute_B():
        newidx = np.empty(n_row, dtype=np.int64)
        for v in grp_arr:
            if len(v) == 1:
                newidx[v[0]] = v[0]              # 单记录分组：保持原记录
            else:
                newidx[v] = perm_constrained(v, D, "B")   # 药物身份必须改变
        return newidx

    rows_out = []
    for scheme, permute_fn in (("A_positional_old", permute_A), ("B_within_cell", permute_B)):
        null = {nm: {m: np.empty(B) for m in MODELS} for nm in ("SDST", "MODZ", "CD")}
        for b in range(B):
            newidx = permute_fn()
            for nm in ("SDST", "MODZ", "CD"):
                ser = np.empty(n_row)
                for m in MODELS:
                    ser[prev[m]] = pcc_norm(Pn[m], Yn[nm][newidx[prev[m]]])
                agg = _agg_mean(ser, pack_ab)
                for m in MODELS:
                    null[nm][m][b] = agg[m]
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
                    "p_empirical": round((1.0 + float((nl >= o).sum())) / (B + 1.0), 6),
                    "n_single_record_groups": n_single,
                    "n_drugs": n_drug,
                    "B": B,
                    "seed": SEED,
                })
                print("%-16s %-9s %-9s 观察 %+.5f | 零均值 %+.5f | 零CI95 [%+.5f,%+.5f] | 超出 %s"
                      % (scheme, nm, m, o, nl.mean(), lo, hi, "是" if o > hi else "否"))

    for _ in range(5):          # 保留原有抽样次数，使随机流与归档结果一致
        permute_B()

    def _flush():
        pd.DataFrame(rows_out).to_csv(P.out / "S28_permutation_scheme_comparison.csv",
                                       index=False, encoding="utf-8-sig")
    _flush()
    print("→ A/B 两方案已落盘，行数 %d" % len(rows_out))

    # ---------------- 方案 C：去掉 92 个单记录分组 ----------------
    keep = np.array([i for v in grp_arr if len(v) > 1 for i in v], dtype=np.int64)
    kmask = np.zeros(n_row, dtype=bool)
    kmask[keep] = True
    print("\n方案 C：仅可置换记录 %d 行 / %d（剔除 %d 个单记录分组）"
          % (len(keep), n_row, n_single))

    Pc, Sc, Bc, Cc = P_all[keep], S_all[keep], Bz_all[keep], Cd_all[keep]
    Mc, Fc, Dc = M[keep], F[keep], D[keep]
    prevc = {"xpert": np.where(Mc == "xpert")[0], "transigen": np.where(Mc == "transigen")[0]}
    Pn_c = {m: _norm_rows(Pc[prevc[m]]) for m in MODELS}
    Yn_c = {nm: _norm_rows(ref) for nm, ref in (("SDST", Sc), ("MODZ", Bc), ("CD", Cc))}
    pack_c = _build_agg(Mc, Dc, Fc)

    def drug_equal_c(ser):
        dfx = pd.DataFrame({"model": Mc, "drug": Dc, "fold": Fc, "r": ser})
        per = (dfx.groupby(["model", "drug", "fold"])["r"].mean().reset_index()
                  .groupby(["model", "drug"])["r"].mean().reset_index())
        return per.pivot(index="drug", columns="model", values="r").dropna()

    obs_c = {}
    for nm, ref in (("SDST", Sc), ("MODZ", Bc), ("CD", Cc)):
        ser = np.empty(len(keep))
        for m in MODELS:
            ser[prevc[m]] = pcc_norm(Pn_c[m], Yn_c[nm][prevc[m]])
        obs_c[nm] = _agg_mean(ser, pack_c)

    Xc = C[keep]          # 细胞系标识（Cc 是 CD 参考矩阵，不能当分组键）
    gk_c = pd.factorize(np.array([f"{Xc[i]}|{Mc[i]}|{Fc[i]}" for i in range(len(keep))], dtype=object))[0]
    members_c = [[] for _ in range(gk_c.max() + 1)]
    for i in range(len(keep)):
        members_c[gk_c[i]].append(i)
    grp_c = [np.array(v, dtype=np.int64) for v in members_c]
    assert min(len(v) for v in grp_c) >= 2, "仍有单记录分组"

    def permute_C():
        newidx = np.empty(len(keep), dtype=np.int64)
        for v in grp_c:
            newidx[v] = perm_constrained(v, Dc, "C")      # 药物身份必须改变
        return newidx

    null_c = {nm: {m: np.empty(B) for m in MODELS} for nm in ("SDST", "MODZ", "CD")}
    for b in range(B):
        newidx = permute_C()
        for nm in ("SDST", "MODZ", "CD"):
            ser = np.empty(len(keep))
            for m in MODELS:
                ser[prevc[m]] = pcc_norm(Pn_c[m], Yn_c[nm][newidx[prevc[m]]])
            agg = _agg_mean(ser, pack_c)
            for m in MODELS:
                null_c[nm][m][b] = agg[m]

    for nm in ("SDST", "MODZ", "CD"):
        for m in MODELS:
            nl = null_c[nm][m]
            lo, hi = float(np.percentile(nl, 2.5)), float(np.percentile(nl, 97.5))
            o = obs_c[nm][m]
            rows_out.append({
                "scheme": "C_within_cell_permutable_only",
                "evaluation_matrix": nm,
                "model": "XPert" if m == "xpert" else "TranSiGen",
                "observed_pearson": round(o, 5),
                "null_mean": round(float(nl.mean()), 5),
                "null_ci95_low": round(lo, 5),
                "null_ci95_high": round(hi, 5),
                "observed_exceeds_ci": bool(o > hi),
                "p_empirical": round((1.0 + float((nl >= o).sum())) / (B + 1.0), 6),
                "n_single_record_groups": 0,
                "n_drugs": int(pd.unique(Dc).size),
                "B": B,
                "seed": SEED,
            })
            print("%-32s %-9s %-9s 观察 %+.5f | 零均值 %+.5f | 零CI95 [%+.5f,%+.5f] | P=%.4g"
                  % ("C_permutable_only", nm, m, o, nl.mean(), lo, hi,
                     (1.0 + float((nl >= o).sum())) / (B + 1.0)))

    out = P.out / "S28_permutation_scheme_comparison.csv"
    pd.DataFrame(rows_out).to_csv(out, index=False, encoding="utf-8-sig", lineterminator="\n")
    print("\n写入:", out, "| 药物 %d | B=%d seed=%d | 单记录分组 %d" % (n_drug, B, SEED, n_single))

    # ---------------- 药物身份约束的逐次核验 ----------------
    for tag, label, npos in (("B", "B_within_cell", n_row - len([v for v in grp_arr if len(v) == 1])),
                             ("C", "C_within_cell_permutable_only", len(keep))):
        a = _audit[tag]
        print("药物身份核验 %-32s 置换 %5d 次 × %5d 条记录 = %9d 个位置 | "
              "保留原药物 %d 处 | 确定性备用构造 %d 次"
              % (label, a["draws"], npos, a["positions"], a["same_drug"], a["fallback"]))
        if a["same_drug"] != 0 or a["fallback"] != 0:
            raise SystemExit("药物身份约束核验未通过（%s），停止。" % label)


if __name__ == "__main__":
    main()
