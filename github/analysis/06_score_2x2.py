# 06 Compute the 2x2 scores (pre-registered caliber, main criterion)
#   space: log2(TPM+1), all common genes
#   score: per test sample, across-gene Pearson (primary) / Spearman (secondary) between the prediction vector and the sample vector
#   four cells: rRR (RNASeQC predictor x RNASeQC ref)  rRE (RNASeQC pred x RSEM ref)
#          rER(RSEM pred × RNASeQC ref)        rEE(RSEM pred × RSEM ref)
#   interaction = (rRR - rRE) - (rER - rEE)
# Output results/per_sample_scores.csv + results/score_summary.json
import os, sys, json, time
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, log, p

cfg = load_config()
grid = json.load(open(p("metadata", "common_grid.json")))
samples = grid["samples"]
splits = json.load(open(p("metadata", "splits.json")))
tmap = json.load(open(p("metadata", "sample_tissue_map.json")))
lab = cfg["analysis"]["tissue_label"]

TM = {k: np.load(p("data_processed", "tissue_mean_%s.npz" % k), allow_pickle=False)
      for k in ["rnaseqc", "rsem"]}
def pred_vec(prod, tissue):
    z = TM[prod]
    ts = list(z["tissues"])
    if tissue in ts:
        return z["means"][ts.index(tissue)]
    return z["fallback"]

A = np.load(p("data_processed", "rnaseqc.npy"), mmap_mode="r")     # (S, NG) TPM
B = np.load(p("data_processed", "rsem_gene.npy"), mmap_mode="r")
sidx = {s: i for i, s in enumerate(samples)}

rows = []
t0 = time.time()
te = splits["test_samples"]
log("%d test samples; scoring sample by sample" % len(te))
for i, s in enumerate(te):
    ia, ib = sidx[s], sidx[s]
    tissue = tmap[s][lab]
    # per product: the log2 sample vector and the log2 tissue-mean prediction vector
    va = np.log2(np.asarray(A[ia], dtype=np.float64) + 1.0)
    vb = np.log2(np.asarray(B[ib], dtype=np.float64) + 1.0)
    pa = pred_vec("rnaseqc", tissue).astype(np.float64)
    pb = pred_vec("rsem", tissue).astype(np.float64)

    def pear(u, v):
        u = u - u.mean(); v = v - v.mean()
        d = np.sqrt((u*u).sum() * (v*v).sum())
        return float((u*v).sum()/d) if d > 0 else np.nan
    def spear(u, v):
        return pear(rankdata(u), rankdata(v))

    r = {"sample": s, "donor": tmap[s]["donor"], "tissue": tissue,
         "rRR": pear(pa, va), "rRE": pear(pa, vb),
         "rER": pear(pb, va), "rEE": pear(pb, vb)}
    r["interaction_pearson"] = (r["rRR"] - r["rRE"]) - (r["rER"] - r["rEE"])
    r.update({"sp_rRR": spear(pa, va), "sp_rRE": spear(pa, vb),
              "sp_rER": spear(pb, va), "sp_rEE": spear(pb, vb)})
    r["interaction_spearman"] = (r["sp_rRR"] - r["sp_rRE"]) - (r["sp_rER"] - r["sp_rEE"])
    rows.append(r)
    if (i+1) % 200 == 0:
        log("  %d/%d, %.0fs" % (i+1, len(te), time.time()-t0))

import csv
cols = list(rows[0].keys())
with open(p("results", "per_sample_scores.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

import statistics as st
summ = {}
for k in ["rRR", "rRE", "rER", "rEE", "interaction_pearson",
          "sp_rRR", "sp_rRE", "sp_rER", "sp_rEE", "interaction_spearman"]:
    v = [r[k] for r in rows if r[k] == r[k]]
    summ[k] = {"median": st.median(v), "mean": st.mean(v), "n": len(v)}
summ["n_test_samples"] = len(rows)
summ["n_test_donors"] = len({r["donor"] for r in rows})
json.dump(summ, open(p("results", "score_summary.json"), "w"), indent=2)
log("PASS -> results/per_sample_scores.csv + score_summary.json")
log("median interaction: pearson=%+.6f spearman=%+.6f"
    % (summ["interaction_pearson"]["median"], summ["interaction_spearman"]["median"]))
