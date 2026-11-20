# cell-bootstrap CI for the interaction (5000x)
#   interaction = (median_CC - median_CH) - (median_HC - median_HH); each replicate independently resamples test cells
#   two calibers, raw and per-gene residualized; objects: Ridge / Frozen / Permuted
import csv, json, os
import numpy as np

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(B, "results", "per_cell_metrics.csv"), encoding="utf-8")))
N_BOOT, SEED = 5000, 20260930

def cell_arr(model, ev, key):
    d = {}
    for r in rows:
        if r["model"] == model and r["eval"] == ev:
            d[r["cell"]] = float(r[key])
    return d

out = {}
for key, kname in [("r_raw", "raw"), ("r_shift", "shift"), ("r_zres", "zres")]:
    obj = {}
    for name, (a_cer, a_chr) in {"Ridge": ("ridge_CERES", "ridge_Chronos"),
                                 "Frozen": ("frozen_CERES", "frozen_Chronos"),
                                 "Permuted": ("ridge_perm", "ridge_perm")}.items():
        CC = cell_arr(a_cer, "CERES", key); CH = cell_arr(a_cer, "Chronos", key)
        HC = cell_arr(a_chr, "CERES", key); HH = cell_arr(a_chr, "Chronos", key)
        cells = sorted(set(CC) & set(CH) & set(HC) & set(HH))
        A = np.array([[CC[c] for c in cells], [CH[c] for c in cells],
                      [HC[c] for c in cells], [HH[c] for c in cells]])
        point = float((np.median(A[0]) - np.median(A[1])) - (np.median(A[2]) - np.median(A[3])))
        rng = np.random.default_rng(SEED)
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            p = rng.integers(0, len(cells), len(cells))
            boot[b] = (np.median(A[0][p]) - np.median(A[1][p])) - (np.median(A[2][p]) - np.median(A[3][p]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        obj[name] = {"point": point, "ci95": [float(lo), float(hi)],
                     "excludes_zero": bool(lo * hi > 0), "n_cells": len(cells)}
        print("%-6s %-5s I=%+.4f CI[%+.4f,%+.4f] excludes0=%s"
              % (kname, name, point, lo, hi, lo * hi > 0))
    out[kname] = obj
json.dump(out, open(os.path.join(B, "results", "interaction_bootstrap.json"), "w"), indent=2)
print("saved results/interaction_bootstrap.json")
