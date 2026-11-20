"""Rebuild the 55point version of Figure 4A (v2): remove arrow annotations, move the legend below the axes to avoid overlapping the curves.

Data: final Supplementary Table S17 (full-set within-beta / within-dcic means and 2.5/97.5 quantiles) + observed cross-reference overlap.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 600})
BLUE, ORANGE, GREEN = "#2F6E9E", "#C65C27", "#337F66"
SRC = Path("_incoming_55point/GigaScience_styleclean_final/supplementary")
OUT = Path("gigascience_aligned_55point_20261001/panels_standalone")
OUT.mkdir(parents=True, exist_ok=True)

s17 = pd.read_csv(SRC / "S17_within_reference_rank_stability.csv")
s17 = s17[s17.analysis_set == "full"].sort_values("k")
K = s17.k.to_numpy(float)
obs = s17.cross_reference_overlap.to_numpy(float) / K
mb = s17.within_beta_mean.to_numpy() / K
lb, hb = s17.within_beta_p2_5.to_numpy() / K, s17.within_beta_p97_5.to_numpy() / K
md = s17.within_dcic_mean.to_numpy() / K
ld, hd = s17.within_dcic_p2_5.to_numpy() / K, s17.within_dcic_p97_5.to_numpy() / K

fig = plt.figure(figsize=(135 / 25.4, 108 / 25.4))
ax = fig.add_axes([0.155, 0.345, 0.815, 0.545])
ax.errorbar(K, mb, yerr=[mb - lb, hb - mb], color=ORANGE, lw=1.2, marker="s", ms=3.5,
            capsize=2, elinewidth=0.7, label="Within beta2020 pairs")
ax.errorbar(K, md, yerr=[md - ld, hd - md], color=GREEN, lw=1.2, marker="^", ms=3.5,
            capsize=2, elinewidth=0.7, label="Within dcic2021 pairs")
ax.plot(K, obs, color=BLUE, lw=1.6, marker="o", ms=4, label="Observed cross-reference")
ax.set(xscale="log", xlabel="List size, $k$", ylabel="Top-k retention", ylim=(0, 1.02))
ax.set_xticks([10, 20, 50, 100, 204, 500, 1000], ["10", "20", "50", "100", "204", "500", "1000"])
ax.set_title("A  Cross-reference retention vs within-reference stability",
             loc="left", fontweight="bold", fontsize=9)
ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, -0.27), columnspacing=1.6, handletextpad=0.5)
fig.savefig(OUT / "Figure4_A_retention_vs_within_reference_stability.pdf")
fig.savefig(OUT / "Figure4_A_retention_vs_within_reference_stability.png")
print("Figure 4A (v2) rebuilt: arrows removed, legend moved below the axes")
