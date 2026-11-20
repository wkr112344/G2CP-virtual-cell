# -*- coding: utf-8 -*-
"""
Q_depmap_disjoint_cohorts.py -- DepMap independent-cohort replication (training target CERES vs Chronos 2x2)

Using the existing 5-seed per-cell results (136 test cells, seeds 0-4), split the cells into **mutually disjoint** cohorts,
and independently recompute within each cohort:
  - the training-target 2x2 four cells CC/CH/HC/HH and interaction I=(CC-CH)-(HC-HH)
  - the CERES arm's own-target vs cross-target contrast (CC-CH), and the Chronos arm's (HH-HC)
the bootstrap is **hierarchical (first resample seeds, then resample the cohort cells within that seed)**, 5000 times.

Correctness gate: on the full sample (136 cells) it must reproduce the published interaction ~ 0.0563 and fall inside its CI [0.0558,0.0572].

Output -> gigascience_v5/results/
  depmap_cohort_replication.csv   per-cohort results
  depmap_cohort_summary.json      with consistency/heterogeneity
"""
import csv
import io
import json
import os
import time

import numpy as np

GH = r"C:\Users\wkr20\Desktop\depmap_second_corpus\GH_second_predictor"
OUT = r"C:\Users\wkr20\WorkBuddy\Claw\gigascience_v5\results"
N_SEED = 5
N_BOOT = 5000
BOOT_SEED = 20260920
COHORT_K = [2, 4]

_t0 = time.time()


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


# --------------------------- load and align the 5-seed per-cell results ------------------------
log("loading the 5-seed per-cell results ...")
S = [np.load(os.path.join(GH, "results", "seed%d_percell.npz" % k), allow_pickle=True)
     for k in range(N_SEED)]
cells = [str(x) for x in S[0]["cells"]]
nC = len(cells)


def aligned(s, key):
    cs = [str(x) for x in s["cells"]]
    pos = {c: i for i, c in enumerate(cs)}
    assert set(cs) == set(cells), "seed cell sets differ"
    return np.array([s[key][pos[c]] for c in cells], dtype=np.float64)


CC = np.stack([aligned(s, "CC") for s in S])      # (5, nC)
CH = np.stack([aligned(s, "CH") for s in S])
HC = np.stack([aligned(s, "HC") for s in S])
HH = np.stack([aligned(s, "HH") for s in S])
INTn = np.stack([aligned(s, "interaction") for s in S])
log("  cells=%d seeds=%d" % (nC, N_SEED))


def interaction_from(rows, seeds=None):
    """Point estimate exactly matching the original aggregate_GH_multiseed.py:
       interaction = mean_over_seeds( median_over_cells( per-cell interaction ) )
       (note: their within-unit aggregation uses **median**, not mean; the difference comes from the skewed per-cell interaction distribution.)"""
    rows = np.asarray(rows, dtype=np.int64)
    sl = np.arange(N_SEED) if seeds is None else np.asarray(seeds, dtype=np.int64)
    per_seed = np.array([np.nanmedian(INTn[s][rows]) for s in sl], dtype=np.float64)
    return float(np.nanmean(per_seed)), per_seed


def hier_boot(rows, nboot=N_BOOT, seed=BOOT_SEED):
    """Hierarchical bootstrap in the original script's caliber: each rep draws 1 seed (with replacement),
       then resamples cells within that seed according to the cohort size, taking the **median**."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows, dtype=np.int64)
    nc = len(rows)
    out = np.empty(nboot, dtype=np.float64)
    for b in range(nboot):
        s = int(rng.integers(0, N_SEED))
        iv = INTn[s][rows]
        iv = iv[~np.isnan(iv)]
        pick = iv[rng.integers(0, len(iv), len(iv))]
        out[b] = np.nanmedian(pick)
    return out


# --------------------------- gate: the full sample must reproduce -------------------------------
allrows = np.arange(nC, dtype=np.int64)


def mean_ci(v):
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


full_point, full_per_seed = interaction_from(allrows)
full_boot = hier_boot(allrows)
fm, flo, fhi = mean_ci(full_boot)
log("=" * 96)
log("gate: full-sample interaction = %.6f  bootstrap CI [%+.6f, %+.6f]"
    % (full_point, flo, fhi))
log("      published mean_interaction=0.056317, hierarchical CI=[0.055806, 0.057157]")
log("      per-seed = %s  (published min 0.056096 / max 0.056554)"
    % np.round(full_per_seed, 6).tolist())
gate = bool(abs(full_point - 0.056317) < 1e-4
            and abs(flo - 0.055806) < 5e-4 and abs(fhi - 0.057157) < 5e-4)
log("reproduction check = %s" % ("passed (caliber matches the original aggregate_GH_multiseed.py)" if gate else "failed"))
log("=" * 96)
if not gate:
    log("!! not reproduced; aborting")
    raise SystemExit(2)

# full-sample four cells
four = {k: float(np.stack([aligned(s, k) for s in S]).mean()) for k in ("CC", "CH", "HC", "HH")}
log("full-sample four cells: " + "  ".join("%s=%.4f" % (k, v) for k, v in four.items()))

# --------------------------- cohort split and replication ---------------------------------------
summary = {"schema": "depmap_disjoint_cohort/v1",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "source": {"per_cell": "GH_second_predictor/results/seed{k}_percell.npz",
                      "n_seeds": N_SEED, "n_test_cells": nC,
                      "bootstrap": "hierarchical (seed -> cell), n=%d, seed=%d" % (N_BOOT, BOOT_SEED)},
           "full_sample": {"interaction": round(full_point, 6),
                           "ci95": [round(flo, 6), round(fhi, 6)],
                           "four_cells": {k: round(v, 6) for k, v in four.items()},
                           "repro_gate_pass": gate},
           "cohorts": {}}
rows_out = []

for K in COHORT_K:
    rngk = np.random.RandomState(20260920 + K)
    perm = rngk.permutation(nC)
    belong = np.empty(nC, dtype=np.int64)
    for bi, ci_ in enumerate(perm):
        belong[ci_] = bi % K
    rec = {"K": K, "per_cohort": []}
    log("K=%d split: about %d cells per cohort" % (K, int(np.bincount(belong).mean())))
    for c in range(K):
        crow = np.where(belong == c)[0].astype(np.int64)
        bsamp = hier_boot(crow, seed=BOOT_SEED + 17 * c + K)
        m, lo, hi = mean_ci(bsamp)
        cc = float(CC[:, crow].mean()); ch = float(CH[:, crow].mean())
        hc = float(HC[:, crow].mean()); hh = float(HH[:, crow].mean())
        point, point_perseed = interaction_from(crow)          # original GH caliber (per-seed cell median)
        point_mean = float((cc - ch) - (hc - hh))              # control: four-cell mean caliber
        # consistency: the cellwise mean of the per-cell interaction should also give the same sign
        cellwise = float(((CC[:, crow] - CH[:, crow]) - (HC[:, crow] - HH[:, crow])).mean())
        ent = {"cohort": int(c), "n_cells": int(len(crow)),
               "CC": round(cc, 6), "CH": round(ch, 6), "HC": round(hc, 6), "HH": round(hh, 6),
               "interaction": round(point, 6),
               "ci95": [round(lo, 6), round(hi, 6)],
               "excludes_zero": bool(lo * hi > 0),
               "interaction_fourcell_mean_convention": round(point_mean, 6),
               "interaction_cellwise_mean_convention": round(cellwise, 6),
               "per_seed_interaction": [round(float(x), 6) for x in point_perseed],
               "same_direction_within_cohort": bool(np.all(point_perseed > 0) or np.all(point_perseed < 0)),
               "ceres_arm_own_minus_cross_CC_minus_CH": round(cc - ch, 6),
               "chronos_arm_own_minus_cross_HH_minus_HC": round(hh - hc, 6)}
        rec["per_cohort"].append(ent)
        rows_out.append(dict(K=K, cohort=c, n_cells=int(len(crow)),
                             CC=ent["CC"], CH=ent["CH"], HC=ent["HC"], HH=ent["HH"],
                             interaction=ent["interaction"],
                             ci95_low=ent["ci95"][0], ci95_high=ent["ci95"][1],
                             excludes_zero=ent["excludes_zero"],
                             interaction_mean_convention=ent["interaction_fourcell_mean_convention"],
                             all_seeds_same_direction=ent["same_direction_within_cohort"],
                             CC_minus_CH=ent["ceres_arm_own_minus_cross_CC_minus_CH"],
                             HH_minus_HC=ent["chronos_arm_own_minus_cross_HH_minus_HC"]))
        log("  cohort %d (n=%d): I=%+.6f [%+.4f,%+.4f] (original caliber) | mean caliber I=%+.6f | CC−CH=%+.4f HH−HC=%+.4f | %s"
            % (c, len(crow), point, lo, hi, point_mean, cc - ch, hh - hc,
               "significant" if ent["excludes_zero"] else "not significant"))

    # consistency and heterogeneity
    est = np.array([e["interaction"] for e in rec["per_cohort"]], dtype=np.float64)
    lo_ = np.array([e["ci95"][0] for e in rec["per_cohort"]], dtype=np.float64)
    hi_ = np.array([e["ci95"][1] for e in rec["per_cohort"]], dtype=np.float64)
    se = (hi_ - lo_) / (2 * 1.959964)
    wts = 1.0 / np.maximum(se ** 2, 1e-12)
    pooled = float((wts * est).sum() / wts.sum())
    Q = float((wts * (est - pooled) ** 2).sum())
    dfree = max(len(est) - 1, 1)
    I2 = float(max(0.0, (Q - dfree) / max(Q, 1e-12)) * 100.0)
    rec["consistency"] = {
        "all_cohorts_same_sign": bool(np.all(est > 0) or np.all(est < 0)),
        "all_cohorts_ci_exclude_zero": bool(all(e["excludes_zero"] for e in rec["per_cohort"])),
        "n_cohorts_significant": int(sum(e["excludes_zero"] for e in rec["per_cohort"])),
        "cohort_estimates": [round(float(x), 6) for x in est],
        "range": [round(float(est.min()), 6), round(float(est.max()), 6)],
        "fixed_effect_pooled": round(pooled, 6),
        "Q": round(Q, 4), "I2_pct": round(I2, 2),
        "vs_full_sample": round(float(pooled - full_point), 6)}
    summary["cohorts"]["K%d" % K] = rec
    log("  -> K=%d consistency: same sign=%s allCIsExclude0=%s pooled=%.6f I2=%.1f%%"
        % (K, rec["consistency"]["all_cohorts_same_sign"],
           rec["consistency"]["all_cohorts_ci_exclude_zero"], pooled, I2))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "depmap_cohort_replication.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)
summary["elapsed_s"] = round(time.time() - _t0, 1)
json.dump(summary, io.open(os.path.join(OUT, "depmap_cohort_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
log("saved -> %s | %.0fs" % (os.path.join(OUT, "depmap_cohort_summary.json"), time.time() - _t0))
