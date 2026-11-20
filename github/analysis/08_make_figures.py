# 08 Make figures (template ready; numbers auto-filled from results/)
#   fig1: RNASeQC vs RSEM (distribution of per-sample score agreement + per-gene agreement between the two references)
#   fig2: 2x2 heatmap + interaction point estimate and bootstrap CI
import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, log, p

cfg = load_config()
rows = list(csv.DictReader(open(p("results", "per_sample_scores.csv"), encoding="utf-8")))
boot = json.load(open(p("results", "bootstrap.json")))
grid = json.load(open(p("metadata", "common_grid.json")))

# --- fig1: per-sample reference agreement (distribution of across-gene Pearson vs the two references) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, pre, ttl in [(axes[0], "", "Pearson (primary)"), (axes[1], "sp_", "Spearman (secondary)")]:
    for key, c, lbl in [(pre+"rRR", "#1f77b4", "RNASeQC pred × RNASeQC"),
                        (pre+"rRE", "#aec7e8", "RNASeQC pred × RSEM"),
                        (pre+"rER", "#d62728", "RSEM pred × RNASeQC"),
                        (pre+"rEE", "#ff9896", "RSEM pred × RSEM")]:
        v = np.array([float(r[key]) for r in rows])
        ax.hist(v, bins=60, alpha=0.45, color=c, label=lbl)
    ax.set_xlabel("per-sample across-gene correlation (log2 TPM+1)")
    ax.set_ylabel("# test samples"); ax.set_title(ttl); ax.legend(fontsize=7)
fig.suptitle("GTEx v11 (%s): reference-product sensitivity of tissue-mean predictors" % cfg["release"])
fig.tight_layout(); fig.savefig(p("figures", "fig1_score_distributions.png"), dpi=200)
log("figures/fig1_score_distributions.png")

# --- fig2: 2×2 heatmap + interaction CI ---
cells = [("rRR", "RNASeQC\npred"), ("rRE", ""), ("rER", "RSEM\npred"), ("rEE", "")]
mats = {"rRR": np.array([float(r["rRR"]) for r in rows]),
        "rRE": np.array([float(r["rRE"]) for r in rows]),
        "rER": np.array([float(r["rER"]) for r in rows]),
        "rEE": np.array([float(r["rEE"]) for r in rows])}
med = {k: float(np.median(v)) for k, v in mats.items()}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
im = ax.imshow([[med["rRR"], med["rRE"]], [med["rER"], med["rEE"]]], cmap="RdBu_r",
               vmin=-1, vmax=1)
ax.set_xticks([0, 1], ["RNASeQC\nreference", "RSEM\nreference"])
ax.set_yticks([0, 1], ["RNASeQC\npredictor", "RSEM\npredictor"])
for (i, j), k in [((0, 0), "rRR"), ((0, 1), "rRE"), ((1, 0), "rER"), ((1, 1), "rEE")]:
    ax.text(j, i, "%+.4f" % med[k], ha="center", va="center", fontsize=13,
            color="white" if abs(med[k]) > 0.5 else "black")
ax.set_title("median per-sample score")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1]
ms = []
for tag, key in [("Pearson", "pearson"), ("Spearman", "spearman")]:
    b = boot[key]
    ms.append((tag, b["point_median"], b["ci95"][0], b["ci95"][1], b["excludes_zero"]))
for i, (tag, pt, lo, hi, ex) in enumerate(ms):
    ax.errorbar(pt, i, xerr=[[pt-lo], [hi-pt]], fmt="o", capsize=4,
                color="#d62728" if ex else "#7f7f7f")
    ax.text(hi + 0.002, i, "CI[%+.4f, %+.4f]%s" % (lo, hi, "  (excl. 0)" if ex else ""), va="center", fontsize=8)
ax.axvline(0, color="red", ls="--", lw=1)
ax.set_yticks([0, 1], [m[0] for m in ms])
ax.set_xlabel("interaction  = (rRR − rRE) − (rER − rEE)")
ax.set_title("2×2 interaction, donor-bootstrap %d×" % cfg["analysis"]["n_bootstrap"])
fig.tight_layout(); fig.savefig(p("figures", "fig2_2x2_interaction.png"), dpi=200)
log("figures/fig2_2x2_interaction.png")
