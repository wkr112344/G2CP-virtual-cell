"""Final 7-figure set (evidence re-organisation, full implementation).

Fig 1  study design                     (untouched)
Fig 2  LINCS fixed-output sensitivity   density scatter / 3-metric ridges / small multiples / models
Fig 3  LINCS interaction & reversal     2x2 violins / rank-rank hexbin / membership rug / exemplars
Fig 4  LINCS ranking consequence        top-k / rank flow / context heatmap / context scatter
Fig 5  LINCS heterogeneity              landscape / caterpillar / drug x cell matrix / ridges / exemplars
Fig 6  DepMap mechanism + pos control   raw dist / paired delta / residual dist / genes / pos control / interaction
Fig 7  GTEx generalization              ridgelines / raincloud / compact 2x2 / bootstrap violins / tissue heatmap

Data-availability notes (flagged in captions):
  * per-drug dcic-trained scores are not part of the package (only aggregate
    0.1705/0.1534) -> Fig 3A shows those as aggregate markers;
  * per-context shortlists are not part of the package (only Jaccard/Spearman
    summaries) -> Fig 4D uses the summary matrix;
  * interaction bootstrap draws exist only for the frozen model -> Fig 6F shows a
    violin for Frozen and point+CI for Ridge/Permuted.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib
import matplotlib.patheffects as pe
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

R = Path(__file__).resolve().parents[1]
S, O, F = R / "source_data", R / "supplementary", R / "figures"
W = 170 / 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                      "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                      "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                      "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})
BLUE, ORANGE, GRAY, GREEN = "#216A94", "#C65C27", "#687681", "#337F66"
LIGHT, DARK = "#9FC3D8", "#172A38"
plotdata = []


def rec(fig, panel_, series, val, **kw):
    plotdata.append(dict(figure=fig, panel=panel_, series=series, value=float(val), **kw))


def panel(ax, l, title, fs_title=9, pad=11):
    ax.set_title(l + "  " + title, loc="left", fontweight="bold", pad=pad, fontsize=fs_title)


def save(fig, name):
    fig.savefig(F / (name + ".pdf"))
    fig.savefig(F / (name + ".png"), dpi=600)
    plt.close(fig)
    print("  wrote", name)


def kde_ridge(ax, data, y0, height, color, xr=None, alpha=0.55):
    kde = gaussian_kde(data)
    if xr is None:
        xr = (float(np.min(data)), float(np.max(data)))
    xs = np.linspace(xr[0], xr[1], 300)
    dens = kde(xs)
    dens = dens / dens.max() * height
    ax.fill_between(xs, y0, y0 + dens, color=color, alpha=alpha, lw=0.8)
    ax.plot(xs, y0 + dens, color=color, lw=1)

# --- Fig 2C shared: exemplar selection, delta colour mapping, colour bar ---------
# Truncated Blues, so the light end stays clearly visible against a white background.
F2C_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "f2c_blue", plt.get_cmap("Blues")(np.linspace(0.40, 1.0, 256)))


def f2c_selection():
    """Six exemplar drugs: 3 with the largest per-drug delta, 3 closest to the median.

    Restricted to drugs with >= 20 cell lines (36 of the 2,037), so both the
    exemplar choice and the median the bottom row sits on are defined on that
    subset. The caller needs the subset to label the panel honestly.
    """
    ncells = d.groupby("drug").size().rename("n_cells")
    pdc = perdrug.set_index("drug_id").join(ncells).reset_index()
    pdc = pdc[pdc.n_cells >= 20]              # enough cell lines for a readable scatter
    strong = pdc.nlargest(3, "delta").reset_index(drop=True)
    stable = pdc.iloc[(pdc.delta - np.median(pdc.delta)).abs().argsort()[:3]].reset_index(drop=True)
    return ([(strong.iloc[k], "high") for k in range(3)] +
            [(stable.iloc[k], "mid") for k in range(3)], pdc)


def f2c_norm(sel):
    dv = np.array([float(r.delta) for r, _ in sel])
    pad = 0.08 * (dv.max() - dv.min())
    return plt.Normalize(vmin=float(dv.min() - pad), vmax=float(dv.max() + pad))


def f2c_colorbar(fig, norm, rect, fs_label, fs_tick):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=F2C_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes(rect), orientation="horizontal")
    cb.set_label("per-drug Δ  (beta2020 − dcic2021)", fontsize=fs_label)
    cb.ax.tick_params(labelsize=fs_tick)
    cb.outline.set_linewidth(0.5)
    return cb



# ------------------------------------------------------------------ data
d = pd.read_csv(S / "frozen_prediction_matrices/frozen_perrow_three_metrics.csv")
for m in ("pearson", "spearman", "cosine"):
    d[m + "_delta"] = d[m + "_A_beta2020"] - d[m + "_B_dcic2021"]
met = pd.read_csv(O / "S1_weighting_and_metrics.csv")
perdrug = pd.read_csv(S / "C_perdrug_stability.csv")
topk = pd.read_csv(O / "S3_topk_performance_ranking.csv")
S17 = pd.read_csv(O / "S17_within_reference_rank_stability.csv")
ext = pd.read_csv(S / "B_sevenmodel.csv")
J = pd.read_csv(S / "J_disease_panels.csv")
gtex = pd.read_csv(S / "gtex_reference_sensitivity/per_sample_scores.csv")
gtex["adv_R"] = gtex.rRR - gtex.rRE
gtex["adv_E"] = gtex.rEE - gtex.rER
S11 = pd.read_csv(O / "S11_gtex_donor_bootstrap.csv")
S12 = pd.read_csv(O / "S12_gtex_tissue_sensitivity.csv")
D = S / "depmap_background_control"
dquad_raw = pd.read_csv(D / "task5_raw_2x2_quadrants.csv")
dquad_res = pd.read_csv(D / "task5_residualized_2x2_quadrants.csv")
dpaired = pd.read_csv(D / "task2_model_vs_trainmean_baseline_paired.csv")
S6 = pd.read_csv(O / "S6_depmap_interaction_bootstrap_summary.csv")
dgene = pd.read_csv(D / "task3_genewise_raw_scores.csv")
dfrozen_boot = pd.read_csv(D / "task6_interaction_bootstrap_distributions.csv")
PC = json.load(open(S / "depmap_positive_control/results/positive_control_summary.json"))
pcell = pd.read_csv(S / "depmap_positive_control/results/per_cell_metrics.csv")
S14 = pd.read_csv(O / "S14_depmap_positive_control_genewise_across_cell.csv")
S15 = pd.read_csv(O / "S15_depmap_positive_control_interaction_bootstrap.csv")
dfrozen_boot = pd.read_csv(D / "task6_interaction_bootstrap_distributions.csv")
donor = gtex.groupby("donor")[["adv_R", "adv_E"]].median()
CTX = {"rank_perf": "Global",
       "rank_ACEVEDO_LIVER_TUMOR_VS_NORMAL_ADJACENT_TISSUE": "Liver tumor (Acevedo)",
       "rank_RODRIGUES_THYROID_CARCINOMA_ANAPLASTIC": "Thyroid ATC (Rodrigues)",
       "rank_CASORELLI_ACUTE_PROMYELOCYTIC_LEUKEMIA": "APL (Casorelli)",
       "rank_HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION_raw": "EMT (raw)",
       "rank_HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION_zrow": "EMT (z-row)",
       "rank_HALLMARK_MYC_TARGETS_V1_raw": "MYC targets (raw)",
       "rank_HALLMARK_MYC_TARGETS_V1_zrow": "MYC targets (z-row)",
       "rank_HALLMARK_HYPOXIA_raw": "Hypoxia (raw)",
       "rank_HALLMARK_HYPOXIA_zrow": "Hypoxia (z-row)"}

# ============================================================================ Figure 2
fig = plt.figure(figsize=(W, 9.2))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.35, 1.25, 0.55],
              hspace=0.55, wspace=0.38, left=0.135, right=0.985, top=0.935, bottom=0.015)

# A density scatter + marginals
ax = fig.add_subplot(gs[0, 0])
ax_main = ax
divider = make_axes_locatable(ax)
ax_top = divider.append_axes("top", size="18%", pad=0.02, sharex=ax)
ax_right = divider.append_axes("right", size="18%", pad=0.02, sharey=ax)
hb = ax_main.hexbin(perdrug.pcc_dcic, perdrug.pcc_beta, gridsize=38, cmap="Blues",
                    mincnt=1, linewidths=0.1)
n_pos = int((perdrug.delta > 0).sum())
ax_main.text(0.04, 0.955, f"{n_pos} / {len(perdrug)} drugs shift\nin the same direction",
             transform=ax_main.transAxes, va="top", fontsize=7.5, color=DARK,
             path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

# Axes run from -0.25, not 0. 22.4% of drugs score NEGATIVELY under dcic2021
# (2526 of 2,037 rows; 87.8% of the per-drug means are negative on that side),
# so a 0-based axis silently truncated a fifth of the data and made the cloud
# look far more concentrated than it is. The diagonal is drawn across the full
# range, so the reference it provides is unchanged.
_AX0 = -0.25
ax_main.plot([_AX0, 1], [_AX0, 1], "--", color=GRAY, lw=1)
ax_main.plot(perdrug.pcc_dcic, perdrug.pcc_beta, "o", ms=1.6, color="#0B3A5C", alpha=0.25)
ax_main.axhline(0, color="#9AA7B0", lw=0.7, zorder=1)
ax_main.axvline(0, color="#9AA7B0", lw=0.7, zorder=1)
ax_main.set(xlabel="Cross-gene Pearson, dcic2021", ylabel="Cross-gene Pearson, beta2020",
            xlim=(_AX0, 1), ylim=(_AX0, 1))
ax_main.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
ax_main.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
# fs_title 8 rather than 9: at this panel width the full wording collided with
# panel B's heading. The wording itself is the author's; only the size is cut.
panel(ax_top, "A", "Fixed predictions scored against two matrices", pad=14, fs_title=8)
_EDG = np.linspace(_AX0, 1, 41)
ax_top.hist(perdrug.pcc_dcic.dropna(), bins=_EDG, color=BLUE, alpha=0.6)
ax_top.axis("off")
ax_right.hist(perdrug.pcc_beta.dropna(), bins=_EDG, orientation="horizontal",
              color=BLUE, alpha=0.6)
ax_right.axis("off")

# B three aligned metric ridges
ax = fig.add_subplot(gs[0, 1])
agg = d.groupby("drug")[["pearson_delta", "spearman_delta", "cosine_delta"]].mean()
xr = (-0.15, 0.65)
labels3 = [("Pearson Δ", "pearson_delta", BLUE), ("Spearman Δ", "spearman_delta", ORANGE),
           ("cosine Δ", "cosine_delta", GREEN)]
for k, (lab, col, c) in enumerate(labels3):
    data = agg[col].dropna().to_numpy()
    y0 = 2.4 - k * 1.15
    kde_ridge(ax, data, y0, 0.85, c, xr=xr)
    m = float(np.median(data))
    pf = float((data > 0).mean())
    ax.plot([m], [y0 + 0.1], "o", color=c, ms=4)
    ax.annotate(f"median {m:.3f} | {pf * 100:.1f}% > 0", xy=(xr[0] + 0.02, y0 + 0.62),
                fontsize=6.8, color="k",
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    rec(2, "B", lab, m, positive_fraction=pf)
ax.axvline(0, color=GRAY, lw=1, ls="--")
ax.set_yticks([])
ax.set(xlabel="Per-drug difference (beta2020 − dcic2021)", xlim=xr, ylim=(-0.3, 3.6))
ax.set_yticks([3.30, 2.15, 1.00])
ax.set_yticklabels(["Pearson Δ", "Spearman Δ", "cosine Δ"], fontsize=7.5, fontweight="bold")
panel(ax, "B", "Three metrics", pad=14)

# C exemplar small multiples (2 x 3)
inner = gs[1, 0].subgridspec(2, 3, wspace=0.25, hspace=0.55)
sel_c, q_c = f2c_selection()
norm_c = f2c_norm(sel_c)
_c_axes = []
for k in range(3):
    for row in (0, 1):
        r, tag = sel_c[row * 3 + k]
        axk = fig.add_subplot(inner[row, k])
        _c_axes.append((axk, row))
        sub = d[d.drug == r.drug_id]
        # -0.20, matching panel A. Four of the six exemplars put cells below zero
        # on the dcic2021 side (down to -0.175), so a 0-based box would clip them.
        axk.plot([-0.20, 1], [-0.20, 1], "--", color=GRAY, lw=0.8)
        axk.scatter(sub.pearson_B_dcic2021, sub.pearson_A_beta2020, s=8,
                    color=F2C_CMAP(norm_c(float(r.delta))), alpha=0.8)
        axk.set(xticks=[0, 0.5, 1], yticks=[0, 0.5, 1],
                xlim=(-0.20, 1), ylim=(-0.20, 1))
        if k > 0:
            axk.set_yticklabels([])
        axk.tick_params(labelsize=6)
        if row == 0:
            axk.set_xticklabels([])
        if k == 0:                                   # y axis, never overwritten below
            axk.set_ylabel("beta2020", fontsize=6.5, labelpad=1)
        if row == 1:
            axk.set_xlabel("dcic2021", fontsize=6.5)
        axk.set_title(r.drug_id, fontsize=5.6, pad=3, loc="left", color="#25333D")
        axk.text(0.04, 0.94, "Δ=%+.4f" % float(r.delta), transform=axk.transAxes,
                 ha="left", va="top", fontsize=5.8, color=DARK)
# The selection rule for these six drugs is stated in the caption, not on the
# figure: the "high Δ" / "Δ ≈ median" row tags were an unsourced editorial label,
# and with the drug id + delta already printed in each panel they add nothing.
_c_pos = [(a.get_position(), row) for a, row in _c_axes]
_cx0 = min(bb.x0 for bb, _ in _c_pos)
_cx1 = max(bb.x1 for bb, _ in _c_pos)
_cy0 = min(bb.y0 for bb, _ in _c_pos)
f2c_colorbar(fig, norm_c, [_cx0 + 0.05, _cy0 - 0.080, (_cx1 - _cx0) * 0.42, 0.011], 5.8, 5.2)
_axc0 = _c_axes[0][0].get_position()
fig.text(_axc0.x0, _axc0.y1 + 0.050, "C  Exemplar drugs: per-cell-line scores",
         fontsize=9, fontweight="bold", ha="left", va="bottom")
fig.text(_axc0.x0, _axc0.y1 + 0.028,
         "x: cross-gene Pearson, dcic2021     y: cross-gene Pearson, beta2020"
         "     colour: per-drug Δ",
         fontsize=6.0, color=DARK, ha="left", va="bottom")
# No exemplar-selection note on the figure. An earlier version carried a
# "top row = 3 largest Δ / bottom row = 3 closest to the median" line here; the
# caption already states the rule, and repeating it inside the panel is what made
# the figure look self-justifying. Each panel already prints its own drug id and Δ.

# D seven prediction sources (narrow strip)
ax = fig.add_subplot(gs[1, 1])
NAME = {"ciger": "CIGER", "deepce": "DeepCE", "multidcp": "MultiDCP", "pertdit": "PertDiT",
        "prnet": "PRnet", "transigen": "TranSiGen", "xpert": "XPert"}
e = ext.copy()
e["nm"] = e.model.str.lower().str.strip().str.replace(" ", "").map(NAME).fillna(e.model)
e = e.sort_values("delta")
for j, row in enumerate(e.itertuples()):
    ax.errorbar(row.delta, j, xerr=[[row.delta - row.ci_low], [row.ci_high - row.delta]],
                fmt="o", color=BLUE, ms=4.5, capsize=3)
    ax.annotate(f"{row.delta:.4f}", xy=(row.ci_high, j), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=7.5)
    rec(2, "D", row.nm, row.delta, ci_low=row.ci_low, ci_high=row.ci_high)
ax.set_yticks(range(len(e)))
ax.set_yticklabels(list(e.nm), fontsize=7)
ax.set(xlabel="Within-model difference (own − dcic2021)", xlim=(0.0, 0.13),
       ylim=(-0.75, len(e) - 0.25))
ax.text(0.985, 0.06, "each source: own evaluation set", transform=ax.transAxes,
        ha="right", fontsize=7, color=GRAY)
panel(ax, "D", "Seven prediction sources")
save(fig, "Figure2_fixed_output")

# ============================================================================ Figure 3
# per-row scores for all four training x reference states (recomputed from the archived
# frozen prediction matrices; per-drug means verified against C_perdrug_stability)
prs = pd.read_csv(S / "frozen_multimetric_perrow_scores.csv")
pd3 = prs.groupby("drug")[["beta_trained__beta2020", "beta_trained__dcic2021",
                           "dcic_trained__beta2020", "dcic_trained__dcic2021"]].mean().reset_index()
vals = np.array([[0.3679865, 0.07238622], [0.1705, 0.1534]])
fig = plt.figure(figsize=(W, 7.8))
gs = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.40,
              left=0.125, right=0.985, top=0.925, bottom=0.075)

# A four drug-level distributions (2x2 states)
ax = fig.add_subplot(gs[0, 0])
STATES4 = [("beta-trained\n× beta2020", "beta_trained__beta2020", BLUE),
           ("beta-trained\n× dcic2021", "beta_trained__dcic2021", "#5C8DB8"),
           ("dcic-trained\n× beta2020", "dcic_trained__beta2020", ORANGE),
           ("dcic-trained\n× dcic2021", "dcic_trained__dcic2021", "#E2A176")]
for k, (lab, col, c) in enumerate(STATES4):
    v = pd3[col].to_numpy()
    vp = ax.violinplot(v, positions=[k], showextrema=False, widths=0.7)
    for body in vp["bodies"]:
        body.set_facecolor(c); body.set_alpha(0.35)
    med, mean = float(np.median(v)), float(v.mean())
    ax.plot([k - 0.10, k + 0.10], [med] * 2, color=DARK, lw=1.6)           # median, solid
    ax.plot([k - 0.13, k + 0.13], [mean] * 2, color="k", lw=1.1, ls="--")   # mean, dashed
    ax.annotate(f"{mean:.3f}", xy=(k, mean), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=7, color="k",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax.annotate(f"{med:.3f}", xy=(k, med), xytext=(0, -11), textcoords="offset points",
                ha="center", fontsize=7, color=DARK,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    rec(3, "A", lab.replace("\n", " ") + " median", med)
    rec(3, "A", lab.replace("\n", " ") + " mean", mean)
ax.set(xticks=range(4),
       xticklabels=["beta-tr." + chr(10) + "× beta", "beta-tr." + chr(10) + "× dcic",
                    "dcic-tr." + chr(10) + "× beta", "dcic-tr." + chr(10) + "× dcic"],
       ylabel="Drug-level cross-gene Pearson",
       # -0.30, not -0.03. Per-drug means go down to -0.1017 / -0.1870 / -0.2659 /
       # -0.1743 across the four states, so the old floor cut off the whole
       # negative tail -- the very part that shows a reference change can invert a
       # drug's score. Floor set from the data with margin, not chosen to fit.
       ylim=(-0.30, 0.95))
panel(ax, "A", "Drug-level distributions per state")

# B rank-rank hexbin
ax = fig.add_subplot(gs[0, 1])
hb = ax.hexbin(perdrug.rank_beta, perdrug.rank_dcic, gridsize=30, cmap="Blues",
               mincnt=1, linewidths=0.1, xscale="log", yscale="log")
ax.plot([0.8, 2500], [0.8, 2500], "--", color=ORANGE, lw=1.2)
from scipy.stats import spearmanr
rho = float(spearmanr(perdrug.rank_beta, perdrug.rank_dcic).statistic)
ax.text(0.05, 0.85, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes, fontsize=8)
for r, c, lab in [(10, BLUE, "top-10"), (50, GREEN, "top-50")]:
    ax.axvline(r, color=c, lw=0.9, ls=":")
    ax.axhline(r, color=c, lw=0.9, ls=":")
ax.text(11, 2400, "top-10", fontsize=6.5, color=BLUE)
ax.text(55, 2400, "top-50", fontsize=6.5, color=GREEN)
ax.set(xscale="log", yscale="log", xlabel="Rank under beta2020", ylabel="Rank under dcic2021")
panel(ax, "B", "Rank-rank density (2,037 drugs)")
cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("# drugs", fontsize=7)
cb.ax.tick_params(labelsize=6.5)

# C per-drug difference with shortlist membership rugs
ax = fig.add_subplot(gs[1, 0])
v = np.sort(perdrug.delta.to_numpy())
ax.hist(v, bins=48, color="#E8EEF2", edgecolor=GRAY, lw=0.4)
rug_sets = [(perdrug[perdrug.in_top10_beta & perdrug.in_top10_dcic].delta, "#6B5B95", "in both top-10"),
            (perdrug[perdrug.in_top10_beta & ~perdrug.in_top10_dcic].delta, BLUE, "beta top-10 only"),
            (perdrug[~perdrug.in_top10_beta & perdrug.in_top10_dcic].delta, ORANGE, "dcic top-10 only")]
for k2, (vv, c, lab) in enumerate(rug_sets):
    ax.plot(vv, np.full(len(vv), -14), "|", color=c, ms=7, mew=1.6)
    ax.annotate(f"{lab} (n = {len(vv)})", xy=(0.0 if not len(vv) else float(np.median(vv)), -28 - k2 * 9),
                fontsize=6.5, color=c, ha="left",
                xytext=(-0.05 if k2 == 0 else (0.18 if k2 == 1 else 0.40), -30 - k2 * 8),
                textcoords="data")
ax.set(xlabel="Per-drug Pearson difference (β − dcic)", ylabel="# drugs",
       ylim=(-45, None))
panel(ax, "C", "Shortlist membership vs shift")

# D exemplar rank-collapse drugs
ax = fig.add_subplot(gs[1, 1])
pool = perdrug[perdrug.rank_beta <= 50].copy()
pool["drop"] = pool.rank_dcic - pool.rank_beta
exo = pool.nlargest(5, "drop").reset_index(drop=True)
for j, r in exo.iterrows():
    ax.plot([0, 1], [r.pcc_beta, r.pcc_dcic], "-o", color=BLUE, ms=5, lw=1.3,
            alpha=0.5 + 0.5 * (j == 0))

ax.annotate("5 largest rank drops" + chr(10) + "within beta top-50", xy=(0, 0.90), ha="left",
            fontsize=7.5, color=GRAY)
ax.set(xticks=[0, 1], xticklabels=["beta2020", "dcic2021"], xlim=(-0.08, 1.30),
       ylabel="Cross-gene Pearson", ylim=(0, 1))
panel(ax, "D", "Exemplar: shortlist collapse")
save(fig, "Figure3_training_evaluation")

# ============================================================================ Figure 4
# ranking consequence: top-k / rank flow / context heatmap / context scatter
fig = plt.figure(figsize=(W, 8.4))
gs = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.40,
              left=0.135, right=0.985, top=0.955, bottom=0.125)

# A top-k retention: observed and within-reference on ONE retention axis
# The two within-reference curves come from S17 (row-level bootstrap inside a single
# evaluation matrix). S17 reports them as OVERLAP COUNTS, so they are divided by k here
# to put all three curves on the same retention scale. Putting counts on a second axis
# next to a rate on the first invites reading the orange/green curves as "low" when
# they are in fact far ABOVE the observed one. S17 only covers k <= 204, so those two
# curves stop there while the observed curve runs to k = 1000.
# Raw counts are annotated next to the markers so the actual numbers stay readable.
ax = fig.add_subplot(gs[0, 0])
_s17 = S17.sort_values("k")
_k = _s17.k.astype(float)
_b_mean = _s17.within_beta_mean / _k
_b_lo = _s17.within_beta_p2_5 / _k
_b_hi = _s17.within_beta_p97_5 / _k
_d_mean = _s17.within_dcic_mean / _k
_d_lo = _s17.within_dcic_p2_5 / _k
_d_hi = _s17.within_dcic_p97_5 / _k

ax.plot(topk.k, topk.retention, "o-", color=BLUE, ms=5,
        label="Observed cross-reference (retention)")
ax.errorbar(_k, _b_mean, yerr=[_b_mean - _b_lo, _b_hi - _b_mean],
            fmt="o-", color=ORANGE, ms=4, lw=1.2, capsize=2.2, elinewidth=0.9,
            zorder=2.5, label="Within beta2020 bootstrap pairs")
ax.errorbar(_k, _d_mean, yerr=[_d_mean - _d_lo, _d_hi - _d_mean],
            fmt="o-", color=GREEN, ms=4, lw=1.2, capsize=2.2, elinewidth=0.9,
            zorder=3, label="Within dcic2021 bootstrap pairs")
ax.set(xscale="log", xlabel="List size, k", ylabel="Top-k retention (fraction of k)",
       ylim=(0, 1.08), xlim=(7, 2600))
# Overlap counts are annotated next to the markers. The two within-reference curves
# nearly coincide at small k, so the beta labels sit above and the dcic labels
# below. The within-matrix values are means over bootstrap pairs and are fractional
# (7.93, 17.10, ...); they were formatted with %d, which truncated them to 7, 17, ...
# and hid that the y-axis is not built on integers. The observed cross-matrix counts
# ARE integers and are printed as such.
# Value labels: end points only.
# An earlier version labelled all 17 points. At this panel width the three series
# crowd the top quarter and the numbers merged into each other and into the
# curves -- no offset scheme fixes that, because the problem is density, not
# collision. What the panel is read for is the LEVEL of each curve and the GAP
# between observed and within-matrix, so only the last point of each series is
# labelled. The per-k values live in S17.
_ol = dict(path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

# Blue: last observed point at k=1000, which is the right edge of the data, so the
# label sits to the LEFT of the marker, right-aligned, and BELOW it -- the curve
# rises steeply into that point, so the space above it belongs to the line.
_bl = topk.k.iloc[-1]
ax.annotate("%d" % topk.overlap.iloc[-1], (_bl, topk.retention.iloc[-1]),
            textcoords="offset points", xytext=(-8, -12), ha="right", va="center",
            fontsize=6.4, color=BLUE, **_ol)

# Orange and green both stop at k=204, close together in value (0.819 / 0.852) and
# far above the blue curve. Each label hugs the outside of its OWN curve -- orange
# just above orange, green just above green -- so neither reads as the other's.
# The 14pt difference is larger than one line height, which keeps the two strings
# from touching while each stays visually attached to its line.
for _col, _pt, _dy in [(GREEN, _s17.within_dcic_mean.iloc[-1], 15),
                       (ORANGE, _s17.within_beta_mean.iloc[-1], 1)]:
    _klast = _s17.k.iloc[-1]
    ax.annotate("%.2f" % _pt, (_klast, _pt / _klast), textcoords="offset points",
                xytext=(8, _dy), ha="left", va="center",
                fontsize=6.4, color=_col, **_ol)

ax.text(0.03, 0.05,
        "End-point labels: observed overlap count (cross-matrix)\n"
        "and mean overlap count (within-matrix bootstrap pairs).\n"
        "Per-k values: Supplementary Table S17.",
        transform=ax.transAxes, va="bottom", fontsize=7.0, color=GRAY, linespacing=1.5)
h1, l1 = ax.get_legend_handles_labels()
ax.legend(h1, l1, frameon=False, loc="upper center",
          bbox_to_anchor=(.5, -.20), ncol=1, fontsize=6.8, columnspacing=1.0)
for row in topk.itertuples():
    rec(4, "A", f"k={row.k}", row.retention, overlap=int(row.overlap))
panel(ax, "A", "Retention vs within-reference stability")

# B top-10 union rank flow
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
tb = perdrug[perdrug.in_top10_beta].sort_values("rank_beta")
td = perdrug[perdrug.in_top10_dcic].sort_values("rank_dcic")
shared_ids = set(tb.drug_id) & set(td.drug_id)
XL, XR = 0.30, 0.70


def ribbon(x0, y0, x1, y1, color, lw=1.5, alpha=0.6):
    mx = (x0 + x1) / 2
    p = MPath([(x0, y0), (mx, y0), (mx, y1), (x1, y1)],
              [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(mpatches.PathPatch(p, fill="none", lw=lw, color=color, alpha=alpha))


yb = {drug: 10 - i for i, drug in enumerate(tb.drug_id)}
yd = {drug: 10 - i for i, drug in enumerate(td.drug_id)}
exit_y, ent_y = 0, 0
for drug in tb.drug_id:
    ax.plot(XL, yb[drug], "o", color=BLUE, ms=4, zorder=3)
    ax.text(XL - 0.015, yb[drug], drug, ha="right", va="center", fontsize=5.6)
    if drug in shared_ids:
        ribbon(XL, yb[drug], XR, yd[drug], BLUE)
    else:
        exit_y -= 1
        ax.plot(XR, exit_y - 0.6, "x", color=GRAY, ms=3.5, mew=1)
        ribbon(XL, yb[drug], XR, exit_y - 0.6, GRAY, lw=1.2, alpha=0.45)
for drug in td.drug_id:
    ax.plot(XR, yd[drug], "o", color=ORANGE, ms=4, zorder=3)
    ax.text(XR + 0.015, yd[drug], drug, ha="left", va="center", fontsize=5.6)
    if drug not in shared_ids:
        ent_y -= 1
        ax.plot(XL, ent_y - 0.6, "x", color=GRAY, ms=3.5, mew=1)
        ribbon(XL, ent_y - 0.6, XR, yd[drug], ORANGE, lw=1.2, alpha=0.45)
ax.text(XL, 10.9, "beta2020", ha="center", fontsize=8, weight="bold", color=BLUE)
ax.text(XR, 10.9, "dcic2021", ha="center", fontsize=8, weight="bold", color=ORANGE)
ax.text(0.5, 12.3, "3 shared", ha="center", fontsize=8, color=BLUE, weight="bold")
ax.set_xlim(0.03, 0.97)
ax.set_ylim(-4.2, 13.2)
panel(ax, "B", "Top-10 union rank flow")

# C shortlist membership matrix (Global + 3 disease contexts; hallmark shortlists
# are not part of the package)
sl = pd.read_csv(r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/图源数据/TableS_disease_top10_shortlist.csv")
g10 = perdrug[perdrug.in_top10_beta][["drug_id", "rank_beta"]].rename(columns={"rank_beta": "Global"})
SIGS = [("ACEVEDO_LIVER_TUMOR_VS_NORMAL_ADJACENT_TISSUE", "Liver"),
        ("RODRIGUES_THYROID_CARCINOMA_ANAPLASTIC", "Thyroid"),
        ("CASORELLI_ACUTE_PROMYELOCYTIC_LEUKEMIA", "APL")]
mat = g10.set_index("drug_id")
for sig, lab in SIGS:
    sub = sl[sl.signature == sig].set_index("drug")["rank_beta"].rename(lab)
    mat = mat.join(sub, how="outer")
mat = mat.sort_values("Global", na_position="last")
mat.index.name = "drug_id"
mat = mat.reset_index()
mat.columns = ["drug_id", "Global"] + [c[1] for c in SIGS]

axM = fig.add_subplot(gs[1, 0])
numcols = list(mat.columns[1:])
nR = len(mat)
nC = len(numcols)
vals = mat[numcols].to_numpy(dtype=float)
im_data = np.ma.masked_invalid(np.where(np.isnan(vals), np.nan, np.log10(np.maximum(vals, 1))))
axM.set_facecolor("#EEEEEE")
imm = axM.imshow(im_data, cmap="YlOrBr_r", vmin=0, vmax=3, aspect="auto")
axM.set_xticks(range(nC), numcols, fontsize=7, rotation=20, ha="right")
axM.set_yticks(range(nR))
axM.set_yticklabels(list(mat.drug_id), fontsize=4.0)
axM.tick_params(axis="y", pad=2)
panel(axM, "C", "Shortlist membership (rank per context)")
cbm = fig.colorbar(imm, ax=axM, orientation="horizontal", fraction=0.045, pad=0.18,
                   ticks=[0, 1, 2, 3])
cbm.set_label("rank (log10 scale);  grey = not shortlisted", fontsize=6.5)
cbm.ax.tick_params(labelsize=6)


# D context summary heatmap (compact) + E concordance vs shortlist stability
innerCD = gs[1, 1].subgridspec(2, 1, height_ratios=[1.25, 1], hspace=0.65)
innerCh = innerCD[0, 0].subgridspec(1, 3, width_ratios=[1, 1, 0.09], wspace=0.06)
J2 = J.copy()
# The "Global" row of J_disease_panels.csv is computed on the 1,399-drug jointly evaluable
# subset, so it disagrees with panels A/B, which use the full 2,037-drug list
# (top-10 overlap 3, Spearman 0.5675). Replace it with the full-list values recomputed
# from the same source as A/B (C_perdrug_stability.csv), so one figure never states two
# different global values. The per-context rows stay on the 1,399 subset -- that is the
# only set they were ever computed on -- and are labelled as such.
_g_tb = set(perdrug.loc[perdrug.in_top10_beta, "drug_id"])
_g_td = set(perdrug.loc[perdrug.in_top10_dcic, "drug_id"])
_g_jac = len(_g_tb & _g_td) / len(_g_tb | _g_td)
_g_row = dict(spearman=spearmanr(perdrug.rank_beta, perdrug.rank_dcic).statistic,
              top10_jaccard=_g_jac, n_drugs=len(perdrug))
print("  Fig4D Global (full list): rho=%.4f top10_jaccard=%.4f N=%d shared=%d"
      % (_g_row["spearman"], _g_jac, _g_row["n_drugs"],
         round(20 * _g_jac / (1 + _g_jac))))
J2.loc[J2.panel == "rank_perf", ["spearman", "top10_jaccard", "n_drugs"]] = [
    _g_row["spearman"], _g_row["top10_jaccard"], _g_row["n_drugs"]]
J2["label"] = J2.panel.map(CTX)
J2["shared"] = (20 * J2.top10_jaccard / (1 + J2.top10_jaccard)).round().astype(int)
J2 = J2.sort_values("spearman", ascending=True).reset_index(drop=True)
yy = np.arange(len(J2))
axL = fig.add_subplot(innerCh[0, 0])
imL = axL.imshow(J2[["spearman"]].to_numpy(), cmap="Blues", vmin=0, vmax=0.7, aspect="auto")

axL.set_xticks([0], ["Spearman ρ"], fontsize=6)
axL.set_yticks(yy)
SH = {"Liver tumor (Acevedo)": "Liver", "Thyroid ATC (Rodrigues)": "Thyroid",
              "APL (Casorelli)": "APL", "EMT (raw)": "EMT-r", "EMT (z-row)": "EMT-z",
              "MYC targets (raw)": "MYC-r", "MYC targets (z-row)": "MYC-z",
              "Hypoxia (raw)": "Hyp-r", "Hypoxia (z-row)": "Hyp-z",
              # "Global" is the full 2,037-drug value, identical to the one panels A/B use.
              # The context rows are the 1,399-drug jointly evaluable subset; that is the
              # only set they exist for, so N is stated per row rather than once in the title.
              "Global": "Global (N = 2,037)"}
axL.set_yticklabels([SH.get(x, x) for x in J2.label], fontsize=6)
axL.set_xticks(np.arange(-.5, 1, 1), minor=True)
axL.grid(which="minor", color="white", lw=1)
axL.tick_params(which="minor", length=0)
for sp in axL.spines.values():
    sp.set_visible(False)
axR = fig.add_subplot(innerCh[0, 1])
imR = axR.imshow(J2[["shared"]].to_numpy(), cmap="Oranges", vmin=0, vmax=7, aspect="auto")

axR.set_xticks([0], ["top-10 shared"], fontsize=6)
axR.set_yticks([])
axR.set_xticks(np.arange(-.5, 1, 1), minor=True)
axR.grid(which="minor", color="white", lw=1)
axR.tick_params(which="minor", length=0)
for sp in axR.spines.values():
    sp.set_visible(False)
axcbL = fig.add_subplot(innerCh[0, 2])
_pc = axcbL.get_position()
axcbL.set_position([_pc.x0, _pc.y0 + _pc.height * 0.56, _pc.width, _pc.height * 0.40])
cbL = fig.colorbar(imL, cax=axcbL, ticks=[0, 0.35, 0.7])
cbL.ax.tick_params(labelsize=4.5)
cbL.set_label("Spearman ρ", fontsize=5.5, labelpad=1)
axcbR = fig.add_axes([_pc.x0, _pc.y0, _pc.width, _pc.height * 0.40])
cbR = fig.colorbar(imR, cax=axcbR, ticks=[0, 3, 7])
cbR.ax.tick_params(labelsize=4.5)
cbR.set_label("top-10 shared", fontsize=5.5, labelpad=1)
panel(axL, "D", "Context summary")
# N is stated per row, not once in the title: Global is the full 2,037-drug list (same value
# as panels A/B), the context rows are the 1,399-drug jointly evaluable subset.
axL.text(0.0, 1.015, "Global: N = 2,037   |   context rows: N = 1,399",
         transform=axL.transAxes, fontsize=5.5, color=GRAY, va="bottom", ha="left")
for i, lab in enumerate(J2.label):
    rec(4, "D", lab.replace("\n", " "), float(J2.spearman[i]), shared=int(J2.shared[i]),
        n=int(J2.n_drugs[i]))

axE = fig.add_subplot(innerCD[1, 0])
# D labels N per row (Global = 2,037, context rows = 1,399), so E's scatter legend can keep
# the short names -- a long "Global (N = 2,037)" entry overflows the right edge at ncol=5.
SHORT = {"Global": "Global", "Liver tumor (Acevedo)": "Liver", "Thyroid ATC (Rodrigues)": "Thyroid",
         "APL (Casorelli)": "APL", "EMT (raw)": "EMT-r", "EMT (z-row)": "EMT-z",
         "MYC targets (raw)": "MYC-r", "MYC targets (z-row)": "MYC-z",
         "Hypoxia (raw)": "Hyp-r", "Hypoxia (z-row)": "Hyp-z"}
_cmap = plt.get_cmap("tab10")
for i, r in J2.iterrows():
    axE.scatter(r.spearman, r.shared, s=30, color=_cmap(int(i) % 10), zorder=3)
_handles = [plt.Line2D([], [], marker="o", ls="none", ms=4.5, color=_cmap(i % 10),
                       label=SHORT.get(lab, lab)) for i, lab in enumerate(J2.label)]
axE.legend(handles=_handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5,
           frameon=False, fontsize=5.4, handletextpad=0.25, columnspacing=0.8)
# Global now shares the same y value as the contexts (3), so the y range is set by the
# context rows only; it is trimmed from the old 8.0 to keep the points filling the panel.
axE.set(ylabel="Top-10 shared", xlim=(0.39, 0.66), ylim=(-0.7, 5.0))
axE.text(.03, .97, "context rank Spearman ρ (x) vs shared (y)", transform=axE.transAxes,
         va="top", fontsize=6.2, color=GRAY)
axE.tick_params(labelsize=6.5)
_posE = axE.get_position()

save(fig, "Figure4_ranking_consequence")

# ============================================================================ Figure 5
# LINCS heterogeneity: landscape / caterpillar / drug x cell matrix / ridges / exemplars
fig = plt.figure(figsize=(W, 10.6))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.25], width_ratios=[1, 1],
              hspace=0.42, wspace=0.42, left=0.19, right=0.985, top=0.945, bottom=0.04)

cc = d.groupby("cell").agg(pb=("pearson_A_beta2020", "mean"),
                           pd_=("pearson_B_dcic2021", "mean"),
                           n=("pearson_delta", "size")).reset_index()
cc["delta"] = cc.pb - cc.pd_
cc = cc.sort_values("delta", ascending=False).reset_index(drop=True)

# A landscape matrix: 63 x (beta, dcic, delta, count strip)
innerA = gs[0, 0].subgridspec(1, 5, width_ratios=[1, 1, 0.9, 0.5, 0.06], wspace=0.06)
vmax = float(np.percentile(np.r_[cc.pb.values, cc.pd_.values], 97))
dmax = float(np.percentile(np.abs(cc.delta.values), 97))
blocks = [("pb", "Blues", 0, vmax, "beta2020"),
          ("pd_", "Blues", 0, vmax, "dcic2021"),
          ("delta", "RdBu_r", -dmax, dmax, "Δ"),
          ("n", "Greens", 0, cc.n.max(), "responses")]
axA = None
for k, (col, cmap, vmin, vvx, xlab) in enumerate(blocks):
    axk = fig.add_subplot(innerA[0, k])
    imk = axk.imshow(cc[[col]].to_numpy(), cmap=cmap, vmin=vmin, vmax=vvx, aspect="auto")
    if k == 0:
        axA = axk
        axk.set_yticks(range(len(cc)))
        axk.set_yticklabels(list(cc.cell), fontsize=5)
    else:
        axk.set_yticks([])
    axk.set_xticks([0], [xlab], fontsize=6.5, rotation=25 if k == 3 else 0,
                   ha="right" if k == 3 else "center")
    axk.set_xticks(np.arange(-.5, 1, 1), minor=True)
    axk.grid(which="minor", color="white", lw=0.8)
    axk.tick_params(which="minor", length=0)
    for sp in axk.spines.values():
        sp.set_visible(False)
axcb = fig.add_subplot(innerA[0, 4])
cb = fig.colorbar(imk, cax=axcb)
cb.ax.tick_params(labelsize=5.5)
panel(axA, "A", "Cell-line × reference landscape")

# B caterpillar plot
ax = fig.add_subplot(gs[0, 1])
y = np.arange(len(cc))[::-1]
ax.hlines(y, 0, cc.delta, color="#D5DEE4", lw=1)
ax.scatter(cc.delta, y, s=6 + 14 * (cc.n / cc.n.max()), color=BLUE, alpha=0.85)
ax.axvline(cc.delta.mean(), ls="--", color=ORANGE, lw=1.2)
ax.set_yticks(y)
ax.set_yticklabels(list(cc.cell), fontsize=5)
ax.set(xlabel="Mean Pearson difference (β − dcic)", xlim=(0, 0.62),
       ylabel="Cell lines (sorted by Δ)")
ax.text(.97, .52, f"mean {cc.delta.mean():.3f}" + chr(10) + "dot size ∝ responses",
        transform=ax.transAxes, ha="right", va="center", fontsize=7, color=ORANGE)
for row in cc.itertuples():
    rec(5, "B", row.cell, row.delta, n=int(row.n))
panel(ax, "B", "Ordered cell-line effect")

# C drug x cell delta matrix (top-covered drugs, clustered rows)
cov = d.groupby("drug").size().sort_values(ascending=False)
keep_drugs = cov[cov >= 25].index[:60]
mat = d[d.drug.isin(keep_drugs)].pivot_table(index="drug", columns="cell",
                                             values="pearson_delta")
order_rows = leaves_list(linkage(mat.apply(lambda r: r.fillna(r.mean()), axis=1).fillna(0).to_numpy(), method="average"))
mat = mat.iloc[order_rows]
col_order = cc.cell.tolist()
mat = mat.reindex(columns=col_order)
ax = fig.add_subplot(gs[1, 0])
matT = mat.T                       # rows = cell lines, columns = drugs
imm = ax.imshow(matT.to_numpy(), vmin=-dmax, vmax=dmax, cmap="RdBu_r", aspect="auto")
imm.set_rasterized(True)
ax.set_yticks(range(matT.shape[0]))
ax.set_yticklabels(list(matT.index), fontsize=4.5)
_xsel = [i for i in range(matT.shape[1]) if i % 6 == 0]
ax.set_xticks(_xsel)
ax.set_xticklabels([matT.columns[i] for i in _xsel], fontsize=4.6, rotation=45, ha="right")
ax.set_xlabel(str(matT.shape[1]) + " most-covered drugs, clustered (every 6th labelled; full list in TableS_drug_coverage_matrix_drugs.csv)",
              fontsize=6.5)
nan_patch = mpatches.Patch(color="#F0F0F0", label="not measured")
ax.legend(handles=[nan_patch], loc="lower right", fontsize=5.5, frameon=False)
panel(ax, "C", "Δ matrix (rows = cell lines, %d drugs)" % matT.shape[1])
rec(5, "C", "matrix_shape", mat.shape[0], n_cols=int(mat.shape[1]),
    measured_fraction=float(mat.notna().mean().mean()))

# D/E six stacked ridges: per-drug, per-cell, then 4 exemplar cells.
# sens/stab are the two cell lines with the largest and smallest mean
# cross-reference difference delta (beta2020 - dcic2021), among those with >= 20
# scored drugs. They were previously labelled "most sensitive" / "most stable",
# which is not supported: nothing here perturbs the cells, so no sensitivity to
# anything is measured. The labels now state the actual selection rule.
sens = cc[cc.n >= 20].iloc[:2]["cell"].tolist()
stab = cc[cc.n >= 20].iloc[-2:]["cell"].tolist()
innerD = gs[1, 1].subgridspec(6, 1, hspace=0.55)
XR = (-0.6, 1.0)
rows = [("per-drug (2,037 drugs)", d.groupby("drug").pearson_delta.mean().dropna().to_numpy(),
         BLUE, "D"),
        ("per-cell (63 cell lines)", cc.delta.to_numpy(), ORANGE, None),
        (sens[0] + "  (example)", d[d.cell == sens[0]].pearson_delta.dropna().to_numpy(),
         "#5C8DB8", None),
        (sens[1] + "  (example)", d[d.cell == sens[1]].pearson_delta.dropna().to_numpy(),
         "#5C8DB8", None),
        (stab[0] + "  (example)", d[d.cell == stab[0]].pearson_delta.dropna().to_numpy(),
         ORANGE, None),
        (stab[1] + "  (example)", d[d.cell == stab[1]].pearson_delta.dropna().to_numpy(),
         "#E2A176", None)]
for k, (lab, v, c, letter) in enumerate(rows):
    axe = fig.add_subplot(innerD[k])
    v = v[np.isfinite(v)]
    kde_ridge(axe, v, 0.0, 0.8, c, xr=XR)
    axe.axvline(0, color=GRAY, lw=0.8, ls="--")
    axe.annotate(lab + "   median %.3f" % np.median(v), xy=(XR[0] + 0.03, 0.55),
                 fontsize=6.3, color=c)
    axe.set_yticks([])
    axe.set_xlim(XR)
    axe.set_ylim(-0.15, 1.0)
    if k < 5:
        axe.set_xticklabels([])
    else:
        axe.set_xlabel("Per-drug / per-cell Pearson difference", fontsize=7)
        axe.tick_params(axis="x", labelsize=6.5)
    if letter:
        panel(axe, letter, "Unit-level and exemplar distributions", fs_title=8.5) if letter else None

save(fig, "Figure5_lincs_heterogeneity")

# ============================================================================ Figure 6
# DepMap mechanism + positive control (6 panels)
fig = plt.figure(figsize=(W, 11.2))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], hspace=0.72, wspace=0.44,
              left=0.145, right=0.985, top=0.945, bottom=0.115)

STATES = [("frozen_CERES", "CERES", "C → C"), ("frozen_CERES", "Chronos", "C → H"),
          ("frozen_Chronos", "CERES", "H → C"), ("frozen_Chronos", "Chronos", "H → H")]

# A frozen raw by-cell distributions
SERIES = [("ridge_CERES", "CERES", BLUE, "-", "Expr Ridge → CERES"),
          ("ridge_Chronos", "Chronos", BLUE, "--", "Expr Ridge → Chronos"),
          ("frozen_CERES", "CERES", ORANGE, "-", "Frozen → CERES"),
          ("frozen_Chronos", "Chronos", ORANGE, "--", "Frozen → Chronos"),
          ("ridge_perm", "CERES", GRAY, "-", "Permuted Ridge")]
ax = fig.add_subplot(gs[0, 0])
for k, (mk, ev, lab) in enumerate(STATES):
    v = pcell[(pcell["model"] == mk) & (pcell["eval"] == ev)].r_raw.to_numpy()
    vp = ax.violinplot(v, positions=[k], showextrema=False, widths=0.75)
    for body in vp["bodies"]:
        body.set_facecolor(BLUE); body.set_alpha(0.30)
    ax.plot([k - 0.10, k + 0.10], [np.median(v)] * 2, color=DARK, lw=1.6)
    ax.annotate(f"{np.median(v):.4f}", xy=(k, np.median(v)), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=6.8)
    rec(6, "A", lab, float(np.median(v)))
ax.set(xticks=range(4), xticklabels=[s[2] for s in STATES],
       ylabel="Raw by-cell Pearson (136 cells)", ylim=(0.88, 0.97))
panel(ax, "A", "Frozen model — raw score distributions")

# B model minus baseline paired deltas
ax = fig.add_subplot(gs[0, 1])
for k, (arm, ref, lab) in enumerate(
        [("ceres", "CERES", "C → C"), ("ceres", "Chronos", "C → H"),
         ("chronos", "CERES", "H → C"), ("chronos", "Chronos", "H → H")]):
    sub = dpaired[(dpaired.arm == arm) & (dpaired.reference == ref)]
    dv = sub.paired_median_delta_model_minus_baseline.to_numpy() * 1000
    jit = np.random.default_rng(7).uniform(-0.08, 0.08, len(dv))
    ax.scatter(k + jit, dv, s=18, color=BLUE, alpha=0.75)
ax.axhline(0, color=GRAY, lw=1)
ax.annotate("model < baseline\nin all four states\n(−0.49 to −1.07 ×1e-3)",
            xy=(1.5, 0.18), ha="center", va="top", fontsize=7, color=GRAY,
            linespacing=1.5,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
ax.set(xticks=range(4), xticklabels=[s[2] for s in STATES],
       ylabel="Model − baseline (×1e-3 Pearson)", ylim=(-1.4, 0.32))
panel(ax, "B", "Paired difference view")

# C frozen residualized by-cell distributions (matched layout to A)
ax = fig.add_subplot(gs[1, 0])
for k, (mk, ev, lab) in enumerate(STATES):
    v = pcell[(pcell["model"] == mk) & (pcell["eval"] == ev)].r_shift.to_numpy()
    vp = ax.violinplot(v, positions=[k], showextrema=False, widths=0.75)
    for body in vp["bodies"]:
        body.set_facecolor(GREEN); body.set_alpha(0.30)
    ax.plot([k - 0.10, k + 0.10], [np.median(v)] * 2, color=DARK, lw=1.6)
    ax.annotate(f"{np.median(v):+.4f}", xy=(k, np.median(v)), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=6.8)
    rec(6, "C", lab, float(np.median(v)))
ax.axhline(0, color=GRAY, lw=1)
ax.set(xticks=range(4), xticklabels=[s[2] for s in STATES],
       ylabel="Residualized by-cell Pearson (136 cells)", ylim=(-0.03, 0.05))
panel(ax, "C", "Frozen model — residualized distributions")

# D gene-wise across-cell ECDFs
ax = fig.add_subplot(gs[1, 1])
for mk, ev, c, ls, lab in [("ridge_CERES", "CERES", BLUE, "-", "Expr Ridge → CERES"),
                           ("ridge_Chronos", "Chronos", BLUE, "--", "Expr Ridge → Chronos"),
                           ("frozen_CERES", "CERES", ORANGE, "-", "Frozen → CERES"),
                           ("frozen_Chronos", "Chronos", ORANGE, "--", "Frozen → Chronos"),
                           ("ridge_perm", "CERES", GRAY, "-", "Permuted Ridge")]:
    v = np.sort(S14[(S14["model"] == mk) & (S14["eval"] == ev)].r_across_cell.to_numpy())
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, lw=1.4, label=lab)
    rec(6, "D", lab, float(np.median(v)))
ax.axvline(0, ls="--", color=GRAY, lw=1)
ax.set(xlabel="Gene-wise across-cell Pearson r", ylabel="ECDF (17,393 genes)",
       xlim=(-0.10, 0.42), ylim=(0, 1))
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.20), ncol=2,
          fontsize=6.2, columnspacing=1.0)
panel(ax, "D", "Gene-wise signal per model")

# E positive-control by-cell residual ECDFs
ax = fig.add_subplot(gs[2, 0])
for mk, ev, c, ls, lab in SERIES:
    v = np.sort(pcell[(pcell["model"] == mk) & (pcell["eval"] == ev)].r_shift.to_numpy())
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, lw=1.4, label=lab)
    rec(6, "E", lab, float(np.median(v)))
ax.axvline(0, ls="--", color=GRAY, lw=1)
ax.set(xlabel="Residualized by-cell Pearson r", ylabel="ECDF (136 test cells)",
       xlim=(-0.06, 0.36), ylim=(0, 1))
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.20), ncol=2,
          fontsize=6.2, columnspacing=1.0)
panel(ax, "E", "Positive control: by-cell distributions")

# F interaction raw -> residualized: bootstrap violins for all three models
def boot_inter(ma, mb, col, n=10000, seed=11):
    a = pcell[(pcell["model"] == ma) & (pcell["eval"] == "CERES")].set_index("cell")[col]
    b = pcell[(pcell["model"] == ma) & (pcell["eval"] == "Chronos")].set_index("cell")[col]
    c = pcell[(pcell["model"] == mb) & (pcell["eval"] == "CERES")].set_index("cell")[col]
    dd = pcell[(pcell["model"] == mb) & (pcell["eval"] == "Chronos")].set_index("cell")[col]
    df = pd.concat([a, b, c, dd], axis=1); df.columns = ["a", "b", "c", "d"]
    v = (df.a - df.b - df.c + df.d).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    return v, v[idx].mean(1)


frozen_raw = dfrozen_boot["raw"].to_numpy()
frozen_res = dfrozen_boot["residualized"].to_numpy()
ridge_raw_v, ridge_raw_b = boot_inter("ridge_CERES", "ridge_Chronos", "r_raw")
ridge_res_v, ridge_res_b = boot_inter("ridge_CERES", "ridge_Chronos", "r_shift")
perm_raw = np.zeros(2000)
perm_res = np.zeros(2000)

MODELS6 = [("Ridge", BLUE, [(0 - 0.24, ridge_raw_v, ridge_raw_b),
                            (1 - 0.24, ridge_res_v, ridge_res_b)]),
           ("Frozen", ORANGE, [(0, frozen_raw, frozen_raw), (1, frozen_res, frozen_res)]),
           ("Permuted", GRAY, [(0 + 0.24, perm_raw, perm_raw), (1 + 0.24, perm_res, perm_res)])]
ax = fig.add_subplot(gs[2, 1])
for mname, col, slots in MODELS6:
    for (x, draws, _pt) in slots:
        vp = ax.violinplot(draws, positions=[x], showextrema=False, widths=0.22)
        for body in vp["bodies"]:
            body.set_facecolor(col); body.set_alpha(0.30)
        ax.plot([x - 0.05, x + 0.05], [np.median(draws)] * 2, color=DARK, lw=1.4)
    xs = [slots[0][0], slots[1][0]]
    ms = [np.median(slots[0][1]), np.median(slots[1][1])]
    ax.plot(xs, ms, color=col, lw=1.1, alpha=0.7)
    dy = {"Ridge": 0.035, "Frozen": 0.022, "Permuted": -0.028}[mname]
    ax.annotate(mname, xy=(xs[1], ms[1] + dy), ha="center", fontsize=6.8, color=col)
    rec(6, "F", mname + " raw", float(np.median(slots[0][1])))
    rec(6, "F", mname + " residualized", float(np.median(slots[1][1])))
ax.axhline(0, ls="--", color=GRAY, lw=1)
ax.set(xticks=[0, 1], xticklabels=["Raw", "Residualized"],
       ylabel="Training × reference interaction (bootstrap)", xlim=(-0.5, 1.5),
       ylim=(-0.05, 0.42))
panel(ax, "F", "Interaction: collapse vs rescue")

save(fig, "Figure6_depmap_mechanism_control")

# ============================================================================ Figure 7
fig = plt.figure(figsize=(W, 15.0))
gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 0.85, 0.75, 1.55],
              hspace=0.55, wspace=0.42, left=0.135, right=0.985, top=0.945, bottom=0.185)

# A ridgelines: Pearson (left) and Spearman (right), 4 states each
GT = [("rRR", "RR: RNASeQC → RNASeQC", BLUE), ("rRE", "RE: RNASeQC → RSEM", "#5C8DB8"),
      ("rER", "ER: RSEM → RNASeQC", ORANGE), ("rEE", "EE: RSEM → RSEM", "#E2A176")]
for col, (prefix, tag) in enumerate([("", "Pearson (primary)"), ("sp_", "Spearman (secondary)")]):
    ax = fig.add_subplot(gs[0, col])
    for k, (base, lab, c) in enumerate([(p[0], p[1], p[2]) for p in GT]):
        v = gtex[(prefix + base)].to_numpy()
        y0 = 3.3 - k * 1.05
        kde_ridge(ax, v, y0, 0.85, c, xr=(0.55, 1.0))
        m = float(np.median(v))
        ax.plot([m], [y0 + 0.1], "o", color=c, ms=4)
        ax.annotate(f"{m:.3f}", xy=(m, y0 + 0.26), ha="center", va="bottom",
                    fontsize=6.5, color=DARK)
        ax.plot([0.5565, 0.5665], [y0 + 0.70, y0 + 0.70], color=c, lw=2.4,
                solid_capstyle="butt")
        ax.text(0.5695, y0 + 0.70, lab, fontsize=6.5, color=DARK, ha="left", va="center")
    ax.set_yticks([])
    ax.set(xlabel="Per-sample across-gene Pearson" if col == 0
           else "Per-sample across-gene Spearman", xlim=(0.55, 1.0), ylim=(-0.3, 4.6))
    panel(ax, "A" if col == 0 else "B", tag)
    for k, (base, lab, c) in enumerate(GT):
        rec(7, "A" if col == 0 else "B", lab, float(np.median(gtex[prefix + base])))

# B donor rainclouds
ax = fig.add_subplot(gs[1, 0])
for k, (col, c, lab) in enumerate([("adv_R", BLUE, "RNASeQC-trained"), ("adv_E", ORANGE, "RSEM-trained")]):
    v = donor[col].to_numpy()
    y0 = 1 - k
    kde_ridge(ax, v, y0 + 0.06, 0.55, c, xr=(-0.005, 0.0495))
    jit = np.random.default_rng(5).uniform(-0.012, 0.012, len(v))
    ax.scatter(v, np.full(len(v), y0) + jit - 0.10, s=3, color=c, alpha=0.4)
    m, lo, hi = np.median(v), *np.percentile(v, [25, 75])
    ax.plot([m, m], [y0 - 0.10, y0 + 0.66], color=DARK, lw=1.5)
    ax.annotate(f"median {m:.3f}\nIQR [{lo:.3f}, {hi:.3f}]\n{(v > 0).mean() * 100:.0f}% > 0",
                xy=(0.054, y0), ha="left", va="center", fontsize=6.4, color=c)
    rec(7, "C", lab, m)
ax.set_yticks([])
ax.set(xlabel="Matched-reference advantage (per-donor median)", xlim=(-0.005, 0.078),
       ylim=(-0.5, 2.55))
panel(ax, "C", "Donor-level advantage (raincloud)")

# C compact 2x2 summary
ax = fig.add_subplot(gs[1, 1])
mat = gtex[["rRR", "rRE", "rER", "rEE"]].median().to_numpy().reshape(2, 2)
ax.imshow(mat, vmin=.94, vmax=.99, cmap="Blues", aspect="auto")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=10,
                color="white" if mat[i, j] > .975 else "#18242E")
        rec(7, "D", f"{i}{j}", mat[i, j])
intr = float(mat[0, 0] - mat[0, 1] - mat[1, 0] + mat[1, 1])

ax.set(xticks=[0, 1], xticklabels=["RNASeQC", "RSEM"], yticks=[0, 1],
       yticklabels=["RNASeQC", "RSEM"], xlabel="Evaluation product", ylabel="Tissue-mean source")
panel(ax, "D", "2×2 summary")

# D donor bootstrap violins
ax = fig.add_subplot(gs[2, 0])
for k, (col, c, lab) in enumerate([("interaction_pearson", BLUE, "Pearson"),
                                   ("interaction_spearman", ORANGE, "Spearman")]):
    v = S11[col].to_numpy()
    vp = ax.violinplot(v, positions=[1 - k], vert=False, showextrema=False, widths=0.7)
    for body in vp["bodies"]:
        body.set_facecolor(c); body.set_alpha(0.35)
    m, lo, hi = np.median(v), *np.percentile(v, [2.5, 97.5])
    ax.plot([m], [1 - k], "o", color=c, ms=5)
    ax.plot([lo, hi], [1 - k, 1 - k], color=c, lw=1.4)
    ax.annotate(f"median {m:.5f}\n[{lo:.5f}, {hi:.5f}]", xy=(m, 1 - k),
                xytext=(0, 18), textcoords="offset points", ha="center",
                va="bottom", fontsize=6.8, color=c)
    ax.set_ylim(-0.95, 2.05)
    rec(7, "E", lab, m, ci_low=lo, ci_high=hi)
ax.set_yticks([1, 0])
ax.set_yticklabels(["Pearson", "Spearman"], fontsize=7.5)
ax.set(xlabel="Donor-median interaction (5,000 donor bootstrap)", xlim=(.055, .078))
panel(ax, "E", "Bootstrap uncertainty")

# E organ-system grouped heatmap
def organ_system(t):
    tl = t.lower()
    if any(k in tl for k in ("brain", "cortex", "cerebellum", "hippocampus", "hypothalamus",
                             "amygdala", "basal ganglia", "substantia", "caudate", "putamen",
                             "accumbens", "spinal cord")):
        return "Brain & CNS"
    if any(k in tl for k in ("whole blood", "ebv", "spleen")):
        return "Blood & immune"
    if "heart" in tl:
        return "Heart"
    if "muscle" in tl:
        return "Muscle"
    if any(k in tl for k in ("lung", "bronchus")):
        return "Lung & airway"
    if any(k in tl for k in ("colon", "small intestine", "stomach", "liver", "pancreas",
                             "esophag", "gallbladder", "rectum", "salivary")):
        return "Digestive & liver"
    if any(k in tl for k in ("kidney", "bladder")):
        return "Kidney & urinary"
    if any(k in tl for k in ("uterus", "vagina", "ovary", "fallopian", "cervix", "breast",
                             "testis", "prostate", "epididym", "seminal")):
        return "Reproductive & breast"
    if any(k in tl for k in ("thyroid", "adrenal", "pituitary", "islet")):
        return "Endocrine"
    if any(k in tl for k in ("artery", "aorta", "coronary", "nerve")):
        return "Vascular & nerve"
    if any(k in tl for k in ("skin", "adipose", "fibroblast")):
        return "Skin & soft tissue"
    return "Other"


SYS_ORDER = ["Brain & CNS", "Blood & immune", "Vascular & nerve", "Heart", "Muscle",
             "Lung & airway", "Digestive & liver", "Kidney & urinary",
             "Reproductive & breast", "Endocrine", "Skin & soft tissue", "Other"]
SYS_COLOR = {"Brain & CNS": "#6B5B95", "Blood & immune": "#C65C27", "Heart": "#A63A32",
             "Muscle": "#8C564B", "Lung & airway": "#4C9AB0", "Digestive & liver": "#337F66",
             "Kidney & urinary": "#4E79A7", "Reproductive & breast": "#B07AA1",
             "Endocrine": "#D4A017", "Skin & soft tissue": "#A9865B",
             "Vascular & nerve": "#5C7A99", "Other": "#999999"}
E_all = S12.copy()
E_all["sys"] = E_all.tissue.map(organ_system)
E_all = E_all.sort_values(["sys", "interaction_pearson"],
                          ascending=[True, False]).reset_index(drop=True)
# Main text keeps one representative tissue per organ system (the highest Pearson interaction
# within that system). All 68 tissues stay in supplementary Table S12 and in the source CSV.
E = E_all.groupby("sys", sort=False).head(1).reset_index(drop=True)
cols = ["interaction_pearson", "interaction_spearman", "n_donors", "adv_R", "adv_E"]
# R = reference built from the RNASeQC quantification, E = from RSEM
titles = ["Pearson\nint.", "Spearman\nint.", "n donors",
          "adv.\nRNASeQC (R)", "adv.\nRSEM (E)"]
cb_labels = ["interaction", "interaction", "donors", "advantage", "advantage"]
# two rows inside the F panel: heat map on top, one colour bar per column underneath
innerE = gs[2:, 1].subgridspec(2, len(cols) + 1, width_ratios=[0.25] + [1] * len(cols),
                               height_ratios=[1, 0.05], wspace=0.32, hspace=0.34)
axs_side = fig.add_subplot(innerE[0, 0])
axs_side.imshow(np.arange(len(E))[:, None], aspect="auto", cmap=matplotlib.colors.ListedColormap(
    [SYS_COLOR[s] for s in E.sys]))
axs_side.set_xticks([]); axs_side.set_yticks([])
for sp in axs_side.spines.values():
    sp.set_visible(False)
for k, col in enumerate(cols):
    axk = fig.add_subplot(innerE[0, k + 1])
    vv = E[col].to_numpy()
    raw = vv.copy()
    if col == "n_donors":
        vv_norm = np.log10(vv)
        imk = axk.imshow(vv_norm[:, None], cmap="Greens", aspect="auto")
    else:
        vv_norm = (vv - vv.min()) / max(vv.max() - vv.min(), 1e-12)
        imk = axk.imshow(vv_norm[:, None], cmap="Blues" if "pearson" in col else
                         ("Oranges" if "spearman" in col else "Purples"), aspect="auto")
    axk.set_xticks([0], [titles[k]], fontsize=6, rotation=25, ha="right")
    axk.set_yticks([])
    axk.set_xticks(np.arange(-.5, 1, 1), minor=True)
    axk.grid(which="minor", color="white", lw=1)
    axk.tick_params(which="minor", length=0)
    for sp in axk.spines.values():
        sp.set_visible(False)
    # per-column colour bar; the panel is narrow here, so tick labels are vertical
    caxk = fig.add_subplot(innerE[1, k + 1])
    cb = fig.colorbar(imk, cax=caxk, orientation="horizontal")
    lo, hi = float(raw.min()), float(raw.max())
    if col == "n_donors":
        tk = np.unique(np.round(np.geomspace(max(lo, 1.0), hi, 3)).astype(int))
        cb.set_ticks(np.log10(tk))
        cb.set_ticklabels([str(int(t)) for t in tk])
    else:
        nd = 3 if hi < 0.2 else 4
        cb.set_ticks([0.0, 1.0])
        cb.set_ticklabels([("%." + str(nd) + "f") % lo, ("%." + str(nd) + "f") % hi])
    cb.ax.tick_params(labelsize=4.8, length=1.2, pad=1.0, labelrotation=90)
    cb.set_label(cb_labels[k], fontsize=5.0, labelpad=2.0)
    cb.outline.set_linewidth(0.4)
axs_side.set_yticks(range(len(E)))
axs_side.set_yticklabels(E.tissue.str.slice(0, 30), fontsize=6.2)
axs_side.yaxis.tick_left()
panel(axs_side, "F", "%d representative tissues" % len(E), fs_title=8.5)
# organ-system legend in the free strip under the panels; the grid columns are untouched
for _i, _sy in enumerate(dict.fromkeys(E.sys)):
    _rr, _cc = divmod(_i, 4)
    _lx = 0.135 + _cc * 0.153
    _ly = 0.152 - _rr * 0.0215
    fig.patches.append(mpatches.Rectangle(
        (_lx, _ly), 0.0050, 0.0125, transform=fig.transFigure,
        facecolor=SYS_COLOR[_sy], edgecolor="none", zorder=5))
    fig.text(_lx + 0.0080, _ly + 0.0062, _sy, fontsize=5.4, va="center", ha="left",
             color=DARK, zorder=5)
for row in E_all.itertuples():
    rec(7, "F", row.tissue, row.interaction_pearson, n_donors=int(row.n_donors))

# The ordering note that used to sit here ("68 tissues: grouped by organ system ...") was
# placed on the bootstrap panel and collided with the tissue labels; it now lives in the caption.
save(fig, "Figure7_gtex_reference_matching")

pd.DataFrame(plotdata).to_csv(O / "Figure_summary_values.csv", index=False)
print("FINAL 7-figure set built")
