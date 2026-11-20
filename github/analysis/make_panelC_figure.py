# Panel C on its own: 2x2 for Ridge and Frozen (raw / per-gene residualized), with 5000x CIs on the interaction
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(B, "results", "positive_control_summary.json")))
grid = S["grid"]
IB = json.load(open(os.path.join(B, "results", "interaction_bootstrap.json")))

cols = [("raw", "raw", 0.95), ("per-gene residualized", "shift", 0.32)]
rows = [("expression-PCA→Ridge\n(positive control)", "Ridge", "#2ca02c"),
        ("frozen G2CP\n(current model)", "Frozen", "#d62728")]

fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0))
for ci, (ctitle, ckey, vmax) in enumerate(cols):
    for ri, (rtitle, rkey, _) in enumerate(rows):
        ax = axes[ri, ci]
        g = grid[rkey]
        M = np.array([[g["CC_" + ckey], g["CH_" + ckey]], [g["HC_" + ckey], g["HH_" + ckey]]])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, "%+.4f" % M[i, j], ha="center", va="center", fontsize=14,
                        color="white" if abs(M[i, j]) > 0.55 * vmax else "black")
        ax.set_xticks([0, 1], ["CERES\nreference", "Chronos\nreference"], fontsize=9)
        ax.set_yticks([0, 1], ["CERES-trained\npredictor", "Chronos-trained\npredictor"], fontsize=9)
        ib = IB[ckey][rkey]
        ax.set_title("%s | %s\ninteraction = %+.4f  [%+.4f, %+.4f]  %s"
                     % (rtitle.split("\n")[0], ctitle, ib["point"], ib["ci95"][0], ib["ci95"][1],
                        "excludes 0 ✓" if ib["excludes_zero"] else "crosses 0 ✗"),
                     fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

fig.suptitle("DepMap 2×2: reference-matching interaction survives residualization only for the model with real cell-state signal\n"
             "(identical train/test split: 636/136 cells, seed 42; 5000× cell bootstrap)", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(B, "figures", "fig_panelC_2x2_heatmap.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
