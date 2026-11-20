# Three-panel figure: positive-control loop
#   A: residualized by-cell correlation（gene-mean baseline / permuted Ridge / real Ridge / frozen）
#   B: gene-wise across-cell correlation (same four objects)
#   C: 2x2 heatmap for Ridge and Frozen (raw and residualized calibers) + interaction
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(B, "results", "positive_control_summary.json")))
sb = S["bootstrap"]; grid = S["grid"]
IB = json.load(open(os.path.join(B, "results", "interaction_bootstrap.json")))

fig = plt.figure(figsize=(11, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.28)

# ---------- Panel A: residualized by-cell ----------
ax = fig.add_subplot(gs[0])
items = [("gene-mean\nbaseline", sb["r_shift_mean_baseline"], "#7f7f7f"),
         ("permuted\nRidge", sb["r_shift_perm"], "#7f7f7f"),
         ("expression\nRidge", sb["r_shift_ridge_CERES"], "#2ca02c"),
         ("frozen\nG2CP", sb["r_shift_frozen_CERES"], "#d62728")]
for i, (nm, v, c) in enumerate(items):
    pt = v["point"]; lo, hi = v["ci95"]
    sig = v["excludes_zero"]
    ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt="o", capsize=5,
                color=c, ms=8, lw=2)
    ax.text(hi + 0.006, i, "%+.3f%s" % (pt, "  ✓" if sig else ""), va="center", fontsize=9,
            color="black" if sig else "#666666")
ax.axvline(0, color="red", ls="--", lw=1)
ax.set_yticks(range(len(items)), [it[0] for it in items], fontsize=9)
ax.set_xlim(-0.03, 0.30)
ax.set_xlabel("by-cell correlation, per-gene residualized\n(log2 dependency, CERES eval)", fontsize=9)
ax.set_title("A  Residualized by-cell correlation\n(residualization does not kill real signal)", fontsize=10)

# ---------- Panel B: across-cell ----------
ax = fig.add_subplot(gs[1])
items2 = [("gene-mean\nbaseline", None, "#7f7f7f"),
          ("permuted\nRidge", sb["across_cell_ridge_perm"], "#7f7f7f"),
          ("expression\nRidge", sb["across_cell_ridge_CERES"], "#2ca02c"),
          ("frozen\nG2CP", sb["across_cell_frozen_CERES"], "#d62728")]
for i, (nm, v, c) in enumerate(items2):
    if v is None:
        ax.plot(0, i, "x", color=c, ms=9, mew=2)
        ax.text(0.006, i, "0 (constant predictor)", va="center", fontsize=8, color="#666666")
        continue
    pt = v["point"]; lo, hi = v["ci95"]
    ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt="o", capsize=5, color=c, ms=8, lw=2)
    ax.text(hi + 0.004, i, "%+.3f%s" % (pt, "  ✓" if v["excludes_zero"] else ""),
            va="center", fontsize=9, color="black" if v["excludes_zero"] else "#666666")
ax.axvline(0, color="red", ls="--", lw=1)
ax.set_yticks(range(len(items2)), [it[0] for it in items2], fontsize=9)
ax.set_xlim(-0.02, 0.16)
ax.set_xlabel("gene-wise across-cell correlation\n(per-gene offsets removed)", fontsize=9)
ax.set_title("B  Gene-wise across-cell correlation\n(cell-state information survives)", fontsize=10)


out = os.path.join(B, "figures", "panel_AB_positive_control.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
