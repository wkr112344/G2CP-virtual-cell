# -*- coding: utf-8 -*-
"""
R_big_independent_cohort.py -- a large experiment centered on "independent cohorts" (unified across two ecosystems)

Treating **cohort independence** itself as the core design variable, executed uniformly on two data ecosystems, tested across orthogonal split axes:

  LINCS (reference-product comparison: level5beta2020 vs dcic2021, frozen beta-trained prediction)
    axis 1: drug-based mutually disjoint cohorts (K=2,4,5) -- existing
    axis 2: cell-line-based mutually disjoint cohorts (K=2,3,4) -- new in this script (orthogonal axis)
    three calibers in parallel: Pearson / Spearman / cosine
    the aggregation unit is still "drug" (matching the main text): first average Δ within a panel per drug, then bootstrap by drug

  DepMap (training-target comparison: CERES vs Chronos, VAE-DeepDEP 5 seeds)
    axis: cell-line-based mutually disjoint cohorts (K=2,4) -- existing
    per-seed cellwise median → mean over seeds；hierarchical bootstrap (seed→cell)

Deliverables (unified):
  results/big_experiment_independent_cohorts.csv   per-cohort (both ecosystems) table
  results/big_experiment_independent_cohorts.json  summary + replication metrics
  figures/big_experiment_independent_cohorts.png   two-ecosystem forest plot
"""
import csv
import io
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = r"C:\Users\wkr20\WorkBuddy\Claw\gigascience_v5\results"
FIG = r"C:\Users\wkr20\WorkBuddy\Claw\gigascience_v5\results\for_figures\figures"
N_BOOT = 2000
BOOT_SEED = 12345
_t0 = time.time()


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def ci_of(per_drug, rng, n=N_BOOT):
    b = np.array([np.nanmean(per_drug[rng.integers(0, len(per_drug), len(per_drug))])
                  for _ in range(n)])
    return float(np.nanmean(per_drug)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def relabel(lines):
    """Original CSV header names -> the three-caliber Δ column names"""
    return None


# ==========================  LINCS section  ==========================
log("loading the LINCS per-row three-caliber table ...")
per = []
with open(os.path.join(RES, "frozen_perrow_three_metrics.csv"), encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for r in rd:
        per.append(r)
log("  rows=%d" % len(per))
cells = np.array([r["cell"] for r in per], dtype=object)
drugs = np.array([r["drug"] for r in per], dtype=object)
METRICS = ["pearson", "spearman", "cosine"]
DELTA = {}
for m in METRICS:
    a = np.array([float(r["%s_A_beta2020" % m]) for r in per], dtype=np.float64)
    b = np.array([float(r["%s_B_dcic2021" % m]) for r in per], dtype=np.float64)
    DELTA[m] = a - b

METRIC_KEY = {"pearson": "delta_pearson", "spearman": "delta_spearman", "cosine": "delta_cosine"}


def lincs_cohort(mask, metric, tag_extra=""):
    """Within a cohort: first aggregate Δ per drug, then bootstrap by drug (matching the main-text aggregation unit)"""
    dg = np.unique(drugs[mask])
    dpos = {d: j for j, d in enumerate(dg)}
    arr = np.empty(len(dg), dtype=np.float64)
    for d in dg:
        arr[dpos[d]] = np.nanmean(DELTA[metric][mask & (drugs == d)])
    mm, lo, hi = ci_of(arr, np.random.default_rng(BOOT_SEED))
    return mm, lo, hi, len(dg), int(mask.sum())


rows_out = []
lincs_summary = {}

# ---- axis 2: cell-line-based mutually disjoint cohorts (orthogonal axis; new in this script) ----
uniq_cells = np.array(sorted(set(cells.tolist())), dtype=object)
log("%d LINCS cell lines; running cell-based independent cohorts ..." % len(uniq_cells))
lincs_summary["axis_cell"] = {}
for K in [2, 3, 4]:
    rngk = np.random.RandomState(31337 + K)
    perm = rngk.permutation(len(uniq_cells))
    belong = np.empty(len(uniq_cells), dtype=np.int64)
    for bi, ci_ in enumerate(perm):
        belong[ci_] = bi % K
    c2g = {c: belong[i] for i, c in enumerate(uniq_cells)}
    row_g = np.array([c2g[c] for c in cells.tolist()], dtype=np.int64)
    rec = {"K": K, "per_cohort": []}
    for c in range(K):
        mask = row_g == c
        ent = {"cohort": int(c)}
        for m in METRICS:
            mm, lo, hi, nd, npair = lincs_cohort(mask, m)
            ent[m] = {"delta": round(mm, 6), "ci95": [round(lo, 6), round(hi, 6)],
                      "excludes_zero": bool(lo * hi > 0)}
            rows_out.append(dict(ecosystem="LINCS", split_axis="cell_line", K=K, cohort=c,
                                 metric=m, n_pairs=int(npair), n_drugs=int(nd),
                                 estimate=round(mm, 6), ci95_low=round(lo, 6), ci95_high=round(hi, 6),
                                 excludes_zero=bool(lo * hi > 0),
                                 estimand="reference contrast Δ=PCC(β2020)−PCC(dcic2021)"))
        rec["per_cohort"].append(ent)
    est = np.array([e["pearson"]["delta"] for e in rec["per_cohort"]], dtype=np.float64)
    rec["consistency"] = {
        "all_same_sign": bool(np.all(est > 0) or np.all(est < 0)),
        "all_ci_exclude_zero": bool(all(e["pearson"]["excludes_zero"] for e in rec["per_cohort"])),
        "estimates_pearson": [round(float(x), 6) for x in est],
        "median_pearson": round(float(np.median(est)), 6),
        "range_pearson": [round(float(est.min()), 6), round(float(est.max()), 6)]}
    lincs_summary["axis_cell"]["K%d" % K] = rec
    log("  K=%d Δ(pearson)=%s all same sign=%s all CIs exclude 0=%s"
        % (K, np.round(est, 4).tolist(), rec["consistency"]["all_same_sign"],
           rec["consistency"]["all_ci_exclude_zero"]))

# ---- axis 1: drug-based (reuse existing results; include in the unified table) ----
log("merging in the LINCS drug-based cohort results (existing) ...")
with open(os.path.join(RES, "frozen_cohort_replication.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        K = int(r["K"])
        for m in METRICS:
            if r["n_drugs"] == "0":
                continue
            rows_out.append(dict(ecosystem="LINCS", split_axis="drug", K=K, cohort=int(r["cohort"]),
                                 metric=m, n_pairs=int(r["n_pairs"]), n_drugs=int(r["n_drugs"]),
                                 estimate=float(r[METRIC_KEY[m]]), ci95_low=float(r["lo_pearson"] if m == "pearson" else "0"),
                                 ci95_high=float(r["hi_pearson"] if m == "pearson" else "0"),
                                 excludes_zero=(r["pearson_excludes_zero"] == "True") if m == "pearson" else None,
                                 estimand="reference contrast Δ=PCC(β2020)−PCC(dcic2021)"))
lincs_summary["note_drug_axis"] = ("drug-axis CIs are written per row into the existing summary json;"
                                   "this unified table carries only the pearson CI for the drug axis (spearman/cosine CIs see "
                                   "frozen_multimetric_summary.json cohort_replication）")

# ==========================  DepMap section  ==========================
log("merging in the DepMap cell-line-based cohort results ...")
with open(os.path.join(RES, "depmap_cohort_replication.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows_out.append(dict(ecosystem="DepMap", split_axis="cell_line", K=int(r["K"]),
                             cohort=int(r["cohort"]), metric="pearson",
                             n_pairs=int(r["n_cells"]), n_drugs="",
                             estimate=float(r["interaction"]),
                             ci95_low=float(r["ci95_low"]), ci95_high=float(r["ci95_high"]),
                             excludes_zero=(r["excludes_zero"] == "True"),
                             estimand="2×2 interaction I=(CC−CH)−(HC−HH)"))

with open(os.path.join(RES, "frozen_multimetric_summary.json"), encoding="utf-8") as f:
    lj = json.load(f)
with open(os.path.join(RES, "depmap_cohort_summary.json"), encoding="utf-8") as f:
    dj = json.load(f)

os.makedirs(FIG, exist_ok=True)
with open(os.path.join(RES, "big_experiment_independent_cohorts.csv"), "w", newline="", encoding="utf-8") as f:
    keys = ["ecosystem", "split_axis", "K", "cohort", "metric", "n_pairs", "n_drugs",
            "estimate", "ci95_low", "ci95_high", "excludes_zero", "estimand"]
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows_out)

# ==========================  replication-metric summary  ==========================
def counts(sub):
    if not sub:
        return dict(n=0, same_sign=None, all_excl=None)
    e = np.array([r["estimate"] for r in sub], dtype=np.float64)
    excl = [r["excludes_zero"] for r in sub]
    return dict(n=len(sub), same_sign=bool(np.all(e > 0) or np.all(e < 0)),
                all_excl=bool(all(x is True for x in excl)),
                n_excl=int(sum(1 for x in excl if x is True)),
                median=round(float(np.median(e)), 6),
                range=[round(float(e.min()), 6), round(float(e.max()), 6)])


one_per_split = {}
for r in rows_out:
    key = (r["ecosystem"], r["split_axis"], r["K"], r["cohort"])
    if r["metric"] == "pearson" and (key not in one_per_split):
        one_per_split[key] = r
pear = list(one_per_split.values())

big = {
    "schema": "big_experiment_independent_cohort/v1",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "question": "Taking cohort independence as the core design variable: can the same (paired) reference/target contrast be reproduced within each mutually disjoint independent cohort?",
    "reference": {
        "LINCS_full_sample_pearson": lj["metrics"]["beta_trained"]["pearson"]["delta_A_minus_B"],
        "DepMap_full_sample_interaction": dj["full_sample"]["interaction"]},
    "repro_gates": {
        "LINCS_reproduced_maintext": bool(lj["repro_check"]["pass"]),
        "DepMap_reproduced_published": bool(dj["full_sample"]["repro_gate_pass"])},
    "by_view": {
        "LINCS_cell_axis": counts([r for r in pear if r["ecosystem"] == "LINCS" and r["split_axis"] == "cell_line"]),
        "DepMap_cell_axis": counts([r for r in pear if r["ecosystem"] == "DepMap"]),
        "LINCS_drug_axis": counts([r for r in pear if r["ecosystem"] == "LINCS" and r["split_axis"] == "drug"]),
        "ALL": counts(pear)},
    "lincs_cell_axis_detail": lincs_summary["axis_cell"],
    "elapsed_s": round(time.time() - _t0, 1)}
json.dump(big, io.open(os.path.join(RES, "big_experiment_independent_cohorts.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

log("=" * 96)
log("replication metrics (pearson caliber, one representative per split)")
for k, v in big["by_view"].items():
    log("  %-18s n=%-3d same-sign=%-5s allCIsExclude0=%-5s median=%.6f range=%s"
        % (k, v["n"], v["same_sign"], v["all_excl"], v.get("median", float("nan")), v.get("range")))
log("  reference: LINCS full-sample Δ=%.6f  |  DepMap full-sample I=%.6f"
    % (big["reference"]["LINCS_full_sample_pearson"][0], big["reference"]["DepMap_full_sample_interaction"]))
log("=" * 96)

# ==========================  figure: two-ecosystem forest  ==========================
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
for ax, eco, title in [(axes[0], "LINCS",
                        "LINCS: reference contrast Δ (β2020 − dcic2021)\nfrozen β-trained prediction"),
                       (axes[1], "DepMap",
                        "DepMap: 2×2 interaction (CERES vs Chronos)\nVAE-DeepDEP, 5 seeds")]:
    sub = [r for r in pear if r["ecosystem"] == eco]
    ys = np.arange(len(sub))
    lab = []
    for r in sub:
        lab.append("%s K=%d c%d (n=%s)" % ({"cell_line": "cell", "drug": "drug"}[r["split_axis"]],
                                           r["K"], r["cohort"], r["n_pairs"]))
    ax.errorbar([r["estimate"] for r in sub], ys,
                xerr=[[max(0.0, r["estimate"] - r["ci95_low"]) for r in sub],
                      [max(0.0, r["ci95_high"] - r["estimate"]) for r in sub]],
                fmt="o", color="#2c7fb8", capsize=3, ecolor="gray")
    ref = (big["reference"]["LINCS_full_sample_pearson"][0] if eco == "LINCS"
           else big["reference"]["DepMap_full_sample_interaction"])
    ax.axvline(ref, color="#1b9e77", lw=1.2, ls="--", label="full-sample estimate")
    ec = "black" if eco == "LINCS" else "black"
    if eco == "DepMap":
        ax.axvline(0, color="r", lw=0.9, ls=":")
    else:
        ax.axvline(0, color="r", lw=0.9, ls=":")
    ax.set_yticks(ys); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("estimate (95% CI)"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
fig.tight_layout()
p = os.path.join(FIG, "big_experiment_independent_cohorts.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
log("saved figure -> %s" % p)
log("saved -> %s | %.0fs" % (os.path.join(RES, "big_experiment_independent_cohorts.json"), time.time() - _t0))
