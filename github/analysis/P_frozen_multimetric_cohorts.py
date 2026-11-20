# -*- coding: utf-8 -*-
"""
P_frozen_multimetric_cohorts.py -- fill in the missing "LINCS frozen prediction" artifacts + the big independent-cohort experiment

Deliverables this round (the same heavy data is loaded only once):
  1) raw prediction matrices: beta-trained / dcic-trained frozen prediction matrices over P*=11,275 pairs x 978 landmarks,
     written to .npy (this intermediate matrix was missing from the earlier local submission package).
  2) multi-caliber readings: besides the main-text Pearson, add **per-row Spearman** and **per-row cosine**,
     giving means + drug-cluster bootstrap 95% CIs for each of the level5beta2020 / dcic2021 references,
     plus the paired Δ (reference contrast). Three calibers in parallel => shows the conclusion is not a Pearson-only artifact.
  3) single-case walkthrough: pick representative rows at the median / extreme Δ, giving the raw numeric
     snippets of the prediction vector and the two reference rows + statistics + scores in all three calibers, so a single conclusion can be checked by hand step by step.
  4) non-overlapping cohort replication for the same reference pair: keeping the reference pair fixed,
     split the 2,037 held-out drugs into mutually disjoint cohorts (drugs never cross cohorts),
     and independently recompute Δ and its CI within each cohort, to see whether it reproduces in each independent cohort.

Correctness gate: must first reproduce the main-text M_BB ~ 0.3680 / M_BD ~ 0.0724 (otherwise abort).

Design follows p4c_run_A_2x2.py exactly: C*=64 / D*=20,370 / H=2,037 / P*=11,275,
the last 10% of drugs under RandomState(0) form H; gene axis = 978 landmarks (internal_col);
bootstrap clusters by drug, N_BOOT=2000, seed 12345.

Run: dpb311 python -u P_frozen_multimetric_cohorts.py
"""
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\Users\wkr20\WorkBuddy\2026-09-06-00-53-37"
OUT = r"C:\Users\wkr20\WorkBuddy\Claw\gigascience_v5\results"

D_REB = os.path.join(BASE, "data", "g2cp_cache_rebuilt")   # used only to define the design set P*
D_B2 = os.path.join(BASE, "data", "g2cp_cache_beta_v2")    # beta eval reference (level5beta2020)
D_21 = os.path.join(BASE, "data", "g2cp_cache_2021")       # dcic eval reference (dcic2021)
GENE_MAP = os.path.join(BASE, "_scratch", "v31_mvpa", "93_gene_map_12328_to_978.json")
CKPT_BETA = os.path.join(BASE, "assets", "g2cp_v7_beta_ft.pt")
CKPT_DCIC = os.path.join(BASE, "assets", "g2cp_v7_dcic_ft.pt")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOT = 2000
EPS = 1e-8
N_COHORT_DRUGS_SPLIT = [2, 4, 5]   # split drugs into K mutually disjoint cohorts
_t0 = time.time()


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


# ----------------------------- 1. design set P* (verbatim copy of p4c) -----------------------
log("building the design set ...")
pmB = np.load(os.path.join(D_21, "pairs_meta.npz"), allow_pickle=True)
b_cell = np.array([str(x) for x in pmB["cell_name"]], dtype=object)
b_drug = np.array([str(x) for x in pmB["drug_id"]], dtype=object)

mR = np.load(os.path.join(D_REB, "meta.npz"), allow_pickle=True)
r_cl = np.array([str(x) for x in mR["cl_names"]], dtype=object)
r_dv = np.array([str(x) for x in mR["drug_vocab"]], dtype=object)
r_cn = np.array([r_cl[int(c)] for c in mR["cell"].astype(np.int64)], dtype=object)
r_dn = np.array([r_dv[int(k)] for k in mR["key"].astype(np.int64)], dtype=object)

Cstar = sorted(set(r_cn.tolist()) & set(b_cell.tolist()))
Dstar = sorted(set(r_dn.tolist()) & set(b_drug.tolist()))
perts = sorted(Dstar)
n_te = int(len(perts) * 0.9)
perm = np.random.RandomState(0).permutation(len(perts))
H = set(perts[i] for i in perm[n_te:])
Cs_set = set(Cstar)
r_pairs = set(zip(r_cn.tolist(), r_dn.tolist()))
b_pairs = set(zip(b_cell.tolist(), b_drug.tolist()))
Pstar = sorted(p for p in (r_pairs & b_pairs) if p[0] in Cs_set and p[1] in H)
nP = len(Pstar)
pidx = {p: i for i, p in enumerate(Pstar)}
log("design C*=%d D*=%d H=%d P*=%d" % (len(Cstar), len(Dstar), len(H), nP))
assert (len(Cstar), len(Dstar), len(H), nP) == (64, 20370, 2037, 11275), "design != archived p4c caliber; aborting"

P_cells = np.array([p[0] for p in Pstar], dtype=object)
P_drugs = np.array([p[1] for p in Pstar], dtype=object)

# ----------------------------- 2. 978 landmark gene axis ----------------------------
gm = json.load(io.open(GENE_MAP, encoding="utf-8"))
internal_col = np.asarray(gm["internal_col"], dtype=np.int64)
assert internal_col.size == 978 and len(set(internal_col.tolist())) == 978
gene_symbols = None
for k in ("symbols", "genes", "gene_symbols", "landmark_symbols"):
    if isinstance(gm.get(k), list) and len(gm[k]) == 978:
        gene_symbols = np.array([str(x) for x in gm[k]], dtype=object)
        break
log("gene axis 978 (internal_col); symbol table %s" % ("available" if gene_symbols is not None else "unavailable (use column index)"))


# ----------------------------- 3. target matrices T_A / T_B -> 978 -----------------------
def pair_level_978(d, cell, drug, tag):
    """Select rows directly at pair level and slice to the 978 columns (avoiding a 12328 intermediate matrix)."""
    Y = np.load(os.path.join(d, "y.npy"), mmap_mode="r")
    sel = [i for i in range(len(cell)) if (cell[i], drug[i]) in pidx]
    tgt = np.array([pidx[(cell[i], drug[i])] for i in sel], dtype=np.int64)
    T = np.zeros((nP, 978), dtype=np.float32)
    CH = 5000
    for s in range(0, len(sel), CH):
        blk = np.asarray(sel[s:s + CH], dtype=np.int64)
        v = np.asarray(Y[blk], dtype=np.float32)
        T[tgt[s:s + CH]] = v[:, internal_col]
        del v
    del Y
    log("  %s: %d rows matched -> T%s" % (tag, len(sel), T.shape))
    return T


pmB2 = np.load(os.path.join(D_B2, "pairs_meta.npz"), allow_pickle=True)
b2_cell = np.array([str(x) for x in pmB2["cell_name"]], dtype=object)
b2_drug = np.array([str(x) for x in pmB2["drug_id"]], dtype=object)

log("reading T_A = cache_beta_v2 (level5beta2020) ...")
T_A = pair_level_978(D_B2, b2_cell, b2_drug, "A")
log("reading T_B = cache_2021 (dcic2021) ...")
T_B = pair_level_978(D_21, b_cell, b_drug, "B")

# ----------------------------- 4. frozen prediction matrices -------------------------------
ck0 = torch.load(CKPT_BETA, map_location="cpu", weights_only=False)
cl_names = [str(x) for x in ck0["cl_names"]]
drugv = [str(x) for x in ck0["drug_vocab"]]
cpos = {c: i for i, c in enumerate(cl_names)}
dpos = {d: i for i, d in enumerate(drugv)}
fps = np.load(os.path.join(D_21, "drug_fps.npy"))
drug_idx = np.array([dpos[p[1]] for p in Pstar], dtype=np.int64)
cell_idx = np.array([cpos[p[0]] for p in Pstar], dtype=np.int64)


def build_net(sd):
    K_CP_W = "cp.weight" if "cp.weight" in sd else "cp_lin.weight"
    K_CP_B = "cp.bias" if "cp.bias" in sd else "cp_lin.bias"
    K_CELL = "cell.weight" if "cell.weight" in sd else "cell_emb.weight"
    G = sd["head.3.weight"].shape[0]
    NCELL = sd[K_CELL].shape[0]
    assert G == 12328, "G=%d should equal 12328" % G

    class Net(nn.Module):
        """Layer-for-layer identical to p4c_run_A_2x2.py: head = LayerNorm(544) -> Linear(544,1024) -> GELU -> Linear(1024,G)
        corresponding to state_dict keys head.{0,1,3} (GELU occupies head.2 with no parameters). Do not remove LayerNorm."""
        def __init__(self):
            super().__init__()
            self.cp = nn.Linear(2048, 512)
            self.cell = nn.Embedding(NCELL, 32)
            self.head = nn.Sequential(nn.LayerNorm(544), nn.Linear(544, 1024), nn.GELU(),
                                      nn.Linear(1024, G))
            own = self.state_dict()
            own["cp.weight"] = sd[K_CP_W]
            own["cp.bias"] = sd[K_CP_B]
            own["cell.weight"] = sd[K_CELL]
            for k in ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias",
                      "head.3.weight", "head.3.bias"]:
                own[k] = sd[k]
            self.load_state_dict(own)

        def forward(self, fp, c):
            return self.head(torch.cat([F.normalize(self.cp(fp), dim=-1), self.cell(c)], 1))

    return Net().to(DEV).eval()


def predict(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = build_net(ck["net"])
    outs = []
    with torch.no_grad():
        for s in range(0, nP, 2048):
            o = net(torch.from_numpy(fps[drug_idx[s:s + 2048]]).to(DEV),
                    torch.from_numpy(cell_idx[s:s + 2048]).to(DEV)).cpu().numpy()
            outs.append(o[:, internal_col].astype(np.float32))
            del o
    del net
    return np.concatenate(outs, 0)


log("predicting beta-trained ...")
P_beta = predict(CKPT_BETA)
log("predicting dcic-trained ...")
P_dcic = predict(CKPT_DCIC)
log("P_beta %s  P_dcic %s" % (P_beta.shape, P_dcic.shape))

os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, "frozen_pred_matrix_beta_trained_11275x978.npy"), P_beta)
np.save(os.path.join(OUT, "frozen_pred_matrix_dcic_trained_11275x978.npy"), P_dcic)
log("both frozen prediction matrices written -> %s" % OUT)


# ----------------------------- 5. per-row metrics in three calibers -------------------------
def row_pearson(Pm, T):
    a = Pm - Pm.mean(1, keepdims=True)
    b = T - T.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + EPS)


def row_rank(X):
    """Per-row ranks (1..n), vectorized; ties use ordinal (continuous floats rarely tie, and tied rows are nan-correlated anyway)."""
    n = X.shape[1]
    order = np.argsort(X, axis=1, kind="mergesort")
    ranks = np.empty(X.shape, dtype=np.float64)
    ar = np.arange(n, dtype=np.float64)
    np.put_along_axis(ranks, order, np.broadcast_to(ar, (X.shape[0], n)), axis=1)
    return ranks + 1.0


def row_spearman(Pm, T):
    return row_pearson(row_rank(Pm.astype(np.float64)), row_rank(T.astype(np.float64)))


def row_cosine(Pm, T):
    num = (Pm * T).sum(1)
    den = np.sqrt((Pm * Pm).sum(1) * (T * T).sum(1))
    return num / (den + EPS)


METRICS = [("pearson", row_pearson), ("spearman", row_spearman), ("cosine", row_cosine)]

udrug = np.unique(drug_idx)
rng_master = np.random.default_rng(12345)


def drug_group(v):
    """per-row -> per-drug mean"""
    out = np.empty(len(udrug), dtype=np.float64)
    for j, g in enumerate(udrug):
        m = drug_idx == g
        out[j] = np.nanmean(v[m])
    return out


def ci_of(per_drug, rng, n=N_BOOT):
    b = np.array([np.nanmean(per_drug[rng.integers(0, len(per_drug), len(per_drug))])
                  for _ in range(n)])
    return float(np.nanmean(per_drug)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


log("computing per-row metrics in three calibers ...")
MET = {}   # MET[predictor][ref][metric] = per-row array
for pname, Pm in [("beta_trained", P_beta), ("dcic_trained", P_dcic)]:
    MET[pname] = {}
    for rname, T in [("A_level5beta2020", T_A), ("B_dcic2021", T_B)]:
        MET[pname][rname] = {}
        for mname, fn in METRICS:
            MET[pname][rname][mname] = fn(Pm, T).astype(np.float64)
            log("  %-13s x %-18s %-8s done" % (pname, rname, mname))

# ---------- multi-caliber summary table ----------
summary = {"schema": "lincs_frozen_multimetric/v1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "design": {"Cstar_cells": 64, "Dstar_drugs": 20370, "heldout_drugs": 2037,
                      "eval_pairs": nP, "gene_axis": "978 landmark",
                      "ckpt_beta": "g2cp_v7_beta_ft.pt", "ckpt_dcic": "g2cp_v7_dcic_ft.pt",
                      "n_boot": N_BOOT, "boot_seed": 12345, "boot_unit": "drug"},
           "metrics": {}}
rows_out = []
for pname in ["beta_trained", "dcic_trained"]:
    for mname, _ in METRICS:
        rng = np.random.default_rng(12345)
        a = MET[pname]["A_level5beta2020"][mname]
        b = MET[pname]["B_dcic2021"][mname]
        ma = ci_of(drug_group(a), rng)
        mb = ci_of(drug_group(b), rng)
        # paired Δ: per-row difference -> per-drug -> bootstrap
        rng2 = np.random.default_rng(12345)
        dpyr = np.array([np.nanmean((a - b)[drug_idx == g]) for g in udrug])
        md = ci_of(dpyr, rng2)
        summary["metrics"].setdefault(pname, {})[mname] = {
            "A": [round(x, 6) for x in ma], "B": [round(x, 6) for x in mb],
            "delta_A_minus_B": [round(x, 6) for x in md],
            "ci_excludes_zero": bool(md[1] * md[2] > 0)}
        rows_out.append(dict(predictor=pname, metric=mname,
                             A_mean=ma[0], A_lo=ma[1], A_hi=ma[2],
                             B_mean=mb[0], B_lo=mb[1], B_hi=mb[2],
                             delta=md[0], delta_lo=md[1], delta_hi=md[2],
                             ci_excludes_zero=bool(md[1] * md[2] > 0)))

import csv
with open(os.path.join(OUT, "frozen_multimetric_three_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)

log("=" * 100)
log("three-caliber readings (per-drug mean [95%CI], drug-cluster bootstrap 2000, seed 12345)")
for r in rows_out:
    log("  %-13s %-9s A=%+.4f[%+.4f,%+.4f]  B=%+.4f[%+.4f,%+.4f]  Δ=%+.4f[%+.4f,%+.4f] %s"
        % (r["predictor"], r["metric"], r["A_mean"], r["A_lo"], r["A_hi"],
           r["B_mean"], r["B_lo"], r["B_hi"], r["delta"], r["delta_lo"], r["delta_hi"],
           "significant" if r["ci_excludes_zero"] else "not significant"))
log("=" * 100)

# ---------- correctness gate: must reproduce the main-text diagonal ----------
mb_ = summary["metrics"]["beta_trained"]["pearson"]["A"][0]
md_ = summary["metrics"]["beta_trained"]["pearson"]["B"][0]
log("* reproduction check pearson: M_BB=%.4f (expected ~0.3680)  M_BD=%.4f (expected ~0.0724)" % (mb_, md_))
summary["repro_check"] = {"M_BB": round(mb_, 6), "M_BD": round(md_, 6),
                          "expected": [0.3680, 0.0724],
                          "pass": bool(abs(mb_ - 0.3680) < 0.005 and abs(md_ - 0.0724) < 0.005)}
if not summary["repro_check"]["pass"]:
    log("!! main-text diagonal not reproduced; downstream conclusions unreliable -- aborting")
    json.dump(summary, io.open(os.path.join(OUT, "frozen_multimetric_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    raise SystemExit(2)
log("reproduction passed")


# ───────────────────────────── 6. single-case walkthrough ────────────────────────
log("picking single cases ...")
d_row = MET["beta_trained"]["A_level5beta2020"]["pearson"] - MET["beta_trained"]["B_dcic2021"]["pearson"]
ok_rows = np.where(np.isfinite(d_row))[0]
i_med = int(ok_rows[np.argsort(np.abs(d_row[ok_rows] - np.median(d_row[ok_rows])))[0]])
i_min = int(ok_rows[np.argmin(d_row[ok_rows])])
i_max = int(ok_rows[np.argmax(d_row[ok_rows])])


def case_detail(i, label):
    pb, tb, pd_, td = P_beta[i], T_A[i], T_B[i], None
    show_cols = list(range(min(24, 978)))
    gz = (gene_symbols[show_cols].tolist() if gene_symbols is not None else show_cols)
    d = {
        "row_index": int(i), "cell": str(P_cells[i]), "drug": str(P_drugs[i]), "label": label,
        "prediction_stats": {"mean": float(pb.mean()), "std": float(pb.std()),
                             "min": float(pb.min()), "max": float(pb.max())},
        "ref_A_level5beta2020_stats": {"mean": float(T_A[i].mean()), "std": float(T_A[i].std()),
                                       "min": float(T_A[i].min()), "max": float(T_A[i].max())},
        "ref_B_dcic2021_stats": {"mean": float(T_B[i].mean()), "std": float(T_B[i].std()),
                                 "min": float(T_B[i].min()), "max": float(T_B[i].max())},
        "first_24_genes": {
            "gene": [str(x) for x in gz],
            "internal_col": [int(internal_col[c]) for c in show_cols],
            "prediction": [round(float(pb[c]), 6) for c in show_cols],
            "ref_A_beta2020": [round(float(T_A[i][c]), 6) for c in show_cols],
            "ref_B_dcic2021": [round(float(T_B[i][c]), 6) for c in show_cols]},
        "row_scores_three_metrics": {
            m: {"A_beta2020": float(MET["beta_trained"]["A_level5beta2020"][m][i]),
                "B_dcic2021": float(MET["beta_trained"]["B_dcic2021"][m][i]),
                "delta_A_minus_B": float(MET["beta_trained"]["A_level5beta2020"][m][i]
                                        - MET["beta_trained"]["B_dcic2021"][m][i])}
            for m, _ in METRICS},
    }
    return d


cases = [case_detail(i_med, "median Δ (typical)"), case_detail(i_min, "min Δ"), case_detail(i_max, "max Δ")]
summary["single_case_walkthrough"] = cases
json.dump(cases, io.open(os.path.join(OUT, "frozen_single_case_walkthrough.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
for c in cases:
    cs = c["row_scores_three_metrics"]
    log("  case[%s] cell=%s drug=%s row=%d | pearson Δ=%+.4f spearman Δ=%+.4f cosine Δ=%+.4f"
        % (c["label"], c["cell"], c["drug"], c["row_index"],
           cs["pearson"]["delta_A_minus_B"], cs["spearman"]["delta_A_minus_B"],
           cs["cosine"]["delta_A_minus_B"]))

# per-sample three-caliber table (for checking; not many rows)
with open(os.path.join(OUT, "frozen_perrow_three_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    hdr = ["row", "cell", "drug"] + ["%s_%s" % (m, r) for m, _ in METRICS
                                     for r in ("A_beta2020", "B_dcic2021")]
    w = csv.writer(f); w.writerow(hdr)
    for i in range(nP):
        w.writerow([i, P_cells[i], P_drugs[i]]
                   + [float(MET["beta_trained"][("A_level5beta2020" if r == "A_beta2020" else "B_dcic2021")][m][i])
                      for m, _ in METRICS for r in ("A_beta2020", "B_dcic2021")])
log("per-row three-caliber table written")


# ───────────────────────────── 7. non-overlapping cohort replication ─────────────
log("non-overlapping cohort replication (same reference pair) ...")
drugs_unique = np.array(sorted(set(P_drugs.tolist())), dtype=object)
log("  P* involves %d drugs" % len(drugs_unique))

cohort_rows = []
summary["cohort_replication"] = {}
for K in N_COHORT_DRUGS_SPLIT:
    rngk = np.random.RandomState(20260920 + K)
    perm_k = rngk.permutation(len(drugs_unique))
    belong = np.empty(len(drugs_unique), dtype=object)
    for bi, di in enumerate(perm_k):
        belong[di] = int(bi % K)
    d2c = {d: belong[i] for i, d in enumerate(drugs_unique)}
    row_cohort = np.array([d2c[d] for d in P_drugs.tolist()], dtype=np.int64)

    rec = {"K": K, "per_cohort": []}
    ok = True
    for c in range(K):
        mask = row_cohort == c
        dgs = np.array(sorted(set(np.array(P_drugs.tolist(), dtype=object)[mask].tolist())), dtype=object)
        rngc = np.random.default_rng(12345 + c * 17 + K)
        ent = {"cohort": int(c), "n_pairs": int(mask.sum()), "n_drugs": int(len(dgs))}
        for mname, _ in METRICS:
            a = MET["beta_trained"]["A_level5beta2020"][mname][mask]
            b = MET["beta_trained"]["B_dcic2021"][mname][mask]
            dgpos = {d: j for j, d in enumerate(dgs)}
            arr = np.empty(len(dgs), dtype=np.float64)
            for j, d in enumerate(dgs):
                arr[j] = np.nanmean((a - b)[np.array(P_drugs.tolist(), dtype=object)[mask] == d])
            mm, lo, hi = ci_of(arr, rngc)
            ent[mname] = {"delta": round(mm, 6), "ci95": [round(lo, 6), round(hi, 6)],
                          "excludes_zero": bool(lo * hi > 0)}
        rec["per_cohort"].append(ent)
        cohort_rows.append(dict(K=K, cohort=c, n_pairs=int(mask.sum()), n_drugs=int(len(dgs)),
                                delta_pearson=ent["pearson"]["delta"],
                                lo_pearson=ent["pearson"]["ci95"][0], hi_pearson=ent["pearson"]["ci95"][1],
                                delta_spearman=ent["spearman"]["delta"],
                                delta_cosine=ent["cosine"]["delta"],
                                pearson_excludes_zero=ent["pearson"]["excludes_zero"]))
        ok = ok and ent["pearson"]["excludes_zero"]
    # cross-cohort consistency / heterogeneity
    deltas = np.array([e["pearson"]["delta"] for e in rec["per_cohort"]], dtype=np.float64)
    low = np.array([e["pearson"]["ci95"][0] for e in rec["per_cohort"]], dtype=np.float64)
    high = np.array([e["pearson"]["ci95"][1] for e in rec["per_cohort"]], dtype=np.float64)
    # I^2-like heterogeneity (infer se from each cohort's CI width)
    se = (high - low) / (2 * 1.959964)
    wts = 1.0 / np.maximum(se ** 2, 1e-12)
    pooled = float((wts * deltas).sum() / wts.sum())
    Q = float((wts * (deltas - pooled) ** 2).sum())
    dfree = max(len(deltas) - 1, 1)
    I2 = float(max(0.0, (Q - dfree) / max(Q, 1e-12)) * 100.0)
    rec["consistency"] = {
        "all_cohorts_same_sign": bool(np.all(deltas > 0) or np.all(deltas < 0)),
        "all_cohorts_ci_exclude_zero": bool(ok),
        "n_cohorts_significant": int(sum(e["pearson"]["excludes_zero"] for e in rec["per_cohort"])),
        "median_delta": round(float(np.median(deltas)), 6),
        "range_delta": [round(float(deltas.min()), 6), round(float(deltas.max()), 6)],
        "fixed_effect_pooled": round(pooled, 6),
        "Q": round(Q, 4), "I2_pct": round(I2, 2)}
    summary["cohort_replication"]["K%d" % K] = rec
    log("  K=%d: per-cohort Δ(pearson)=%s  all CIs exclude 0=%s  signs agree=%s  pooled=%.4f  I2=%.1f%%"
        % (K, np.round(deltas, 4).tolist(), ok, rec["consistency"]["all_cohorts_same_sign"], pooled, I2))

with open(os.path.join(OUT, "frozen_cohort_replication.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(cohort_rows[0].keys()))
    w.writeheader(); w.writerows(cohort_rows)

summary["elapsed_s"] = round(time.time() - _t0, 1)
json.dump(summary, io.open(os.path.join(OUT, "frozen_multimetric_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
log("saved summary -> %s | %.0fs" % (os.path.join(OUT, "frozen_multimetric_summary.json"), time.time() - _t0))
