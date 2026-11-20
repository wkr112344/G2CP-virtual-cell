"""Render every panel as its own standalone figure (no cropping).

Each panel is redrawn into a dedicated figure of an appropriate size, using the same
data, style helpers and drawing code as the composite build.
Output: gigascience_panels_rendered_20260930/  (PNG 600 dpi + PDF)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.path import Path as MPath
import matplotlib
import matplotlib.patheffects as pe
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

R = Path(__file__).resolve().parents[1]
S, O = R / "source_data", R / "supplementary"
OUT = Path(r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_panels_rendered_20260930")
OUT.mkdir(parents=True, exist_ok=True)
MM = 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})
BLUE, ORANGE, GRAY, GREEN = "#216A94", "#C65C27", "#687681", "#337F66"
LIGHT, DARK = "#9FC3D8", "#172A38"

# Every label drawn inside a panel is black with a white outline. On a white page
# the outline is invisible, but it keeps text readable where a ridge, a median
# rule, a colour bar or an axis line passes underneath, and stops hairline
# numerals from being swallowed by an edge.
import matplotlib.patheffects as _pe
OUTLINE = [_pe.withStroke(linewidth=2.2, foreground="white")]


def ink(size=7.5, bold=False, italic=False, color="black"):
    """Black text with a white outline, for labels that may overlap artwork."""
    return dict(color=color, fontsize=size, weight="bold" if bold else "normal",
                style="italic" if italic else "normal", path_effects=OUTLINE)


def panel(ax, l, title, fs_title=10, pad=10):
    ax.set_title(l + "  " + title, loc="left", fontweight="bold", pad=pad,
                 fontsize=fs_title, color="black", path_effects=OUTLINE)

# --- Fig 2C shared: exemplar selection, delta colour mapping, colour bar ---------
# Truncated Blues, so the light end stays clearly visible against a white background.
F2C_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "f2c_blue", plt.get_cmap("Blues")(np.linspace(0.40, 1.0, 256)))


def f2c_selection():
    """Six exemplar drugs: 3 with the largest per-drug delta, 3 closest to the median.

    Restricted to drugs with >= 20 cell lines (36 of the 2,037), so both the
    exemplar choice and the median the bottom row sits on are defined on that
    subset. Returns the subset medians so the panel can be labelled honestly.
    """
    ncells = d.groupby("drug").size().rename("n_cells")
    pdc = perdrug.set_index("drug_id").join(ncells).reset_index()
    pdc = pdc[pdc.n_cells >= 20]              # enough cell lines for a readable scatter
    strong = pdc.nlargest(3, "delta").reset_index(drop=True)
    stable = pdc.iloc[(pdc.delta - np.median(pdc.delta)).abs().argsort()[:3]].reset_index(drop=True)
    sel = ([(strong.iloc[k], "high") for k in range(3)] +
           [(stable.iloc[k], "mid") for k in range(3)])
    return sel, float(np.median(pdc.delta)), len(pdc), float(np.median(perdrug.delta))


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



def kde_ridge(ax, data, y0, height, color, xr=None, alpha=0.55):
    kde = gaussian_kde(data)
    if xr is None:
        xr = (float(np.min(data)), float(np.max(data)))
    xs = np.linspace(xr[0], xr[1], 300)
    dens = kde(xs); dens = dens / dens.max() * height
    ax.fill_between(xs, y0, y0 + dens, color=color, alpha=alpha, lw=0.8)
    ax.plot(xs, y0 + dens, color=color, lw=1)


# ------------------------------------------------------------------ data
d = pd.read_csv(S / "frozen_prediction_matrices/frozen_perrow_three_metrics.csv")
for m in ("pearson", "spearman", "cosine"):
    d[m + "_delta"] = d[m + "_A_beta2020"] - d[m + "_B_dcic2021"]
perdrug = pd.read_csv(S / "C_perdrug_stability.csv")
topk = pd.read_csv(O / "S3_topk_performance_ranking.csv")
ext = pd.read_csv(S / "B_sevenmodel.csv")
J = pd.read_csv(S / "J_disease_panels.csv")
gtex = pd.read_csv(S / "gtex_reference_sensitivity/per_sample_scores.csv")
gtex["adv_R"] = gtex.rRR - gtex.rRE
gtex["adv_E"] = gtex.rEE - gtex.rER
donor = gtex.groupby("donor")[["adv_R", "adv_E"]].median()
S11 = pd.read_csv(O / "S11_gtex_donor_bootstrap.csv")
S12 = pd.read_csv(O / "S12_gtex_tissue_sensitivity.csv")
D = S / "depmap_background_control"
dquad_raw = pd.read_csv(D / "task5_raw_2x2_quadrants.csv")
dquad_res = pd.read_csv(D / "task5_residualized_2x2_quadrants.csv")
dpaired = pd.read_csv(D / "task2_model_vs_trainmean_baseline_paired.csv")
S6 = pd.read_csv(O / "S6_depmap_interaction_bootstrap_summary.csv")
dgene = pd.read_csv(D / "task3_genewise_raw_scores.csv")
dfrozen_boot = pd.read_csv(D / "task6_interaction_bootstrap_distributions.csv")
pcell = pd.read_csv(S / "depmap_positive_control/results/per_cell_metrics.csv")
S14 = pd.read_csv(O / "S14_depmap_positive_control_genewise_across_cell.csv")
S15 = pd.read_csv(O / "S15_depmap_positive_control_interaction_bootstrap.csv")
prs = pd.read_csv(S / "frozen_multimetric_perrow_scores.csv")
pd3 = prs.groupby("drug")[["beta_trained__beta2020", "beta_trained__dcic2021",
                           "dcic_trained__beta2020", "dcic_trained__dcic2021"]].mean().reset_index()
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
SHORT = {"Global": "Global", "Liver tumor (Acevedo)": "Liver", "Thyroid ATC (Rodrigues)": "Thyroid",
         "APL (Casorelli)": "APL", "EMT (raw)": "EMT-r", "EMT (z-row)": "EMT-z",
         "MYC targets (raw)": "MYC-r", "MYC targets (z-row)": "MYC-z",
         "Hypoxia (raw)": "Hyp-r", "Hypoxia (z-row)": "Hyp-z"}

# ==================================================================== panels
def f2a(ax):
    div = make_axes_locatable(ax)
    ax_top = div.append_axes("top", size="18%", pad=0.02, sharex=ax)
    ax_right = div.append_axes("right", size="18%", pad=0.02, sharey=ax)
    ax.hexbin(perdrug.pcc_dcic, perdrug.pcc_beta, gridsize=38, cmap="Blues", mincnt=1,
              linewidths=0.1)
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1)
    ax.plot(perdrug.pcc_dcic, perdrug.pcc_beta, "o", ms=1.6, color="#0B3A5C", alpha=0.25)
    n_pos = int((perdrug.delta > 0).sum())
    ax.text(0.04, 0.955, "%d / %d drugs shift in the same direction" % (n_pos, len(perdrug)),
            transform=ax.transAxes, va="top", fontsize=8, color=DARK)
    ax.set(xlabel="Cross-gene Pearson, dcic2021", ylabel="Cross-gene Pearson, beta2020",
           xlim=(0, 1), ylim=(0, 1))
    ax_top.hist(perdrug.pcc_dcic, bins=40, color=BLUE, alpha=0.6); ax_top.axis("off")
    ax_right.hist(perdrug.pcc_beta, bins=40, orientation="horizontal", color=BLUE, alpha=0.6)
    ax_right.axis("off")
    panel(ax_top, "A", "Reference-vs-reference density (2,037 drugs)", pad=14)


def f2b(ax):
    agg = d.groupby("drug")[["pearson_delta", "spearman_delta", "cosine_delta"]].mean()
    xr = (-0.15, 0.65)
    for k, (lab, col, c) in enumerate([("Pearson Δ", "pearson_delta", BLUE),
                                       ("Spearman Δ", "spearman_delta", ORANGE),
                                       ("cosine Δ", "cosine_delta", GREEN)]):
        data = agg[col].dropna().to_numpy(); y0 = 2.4 - k * 1.15
        kde_ridge(ax, data, y0, 0.85, c, xr=xr)
        med = float(np.median(data)); pf = float((data > 0).mean())
        ax.plot([med], [y0 + 0.1], "o", color=c, ms=4)
        ax.annotate("median %.3f | %.0f%% > 0" % (med, pf * 100), xy=(xr[0] + 0.02, y0 + 0.62),
                    fontsize=7, color="k",
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.axvline(0, color=GRAY, lw=1, ls="--")
    ax.set(xlabel="Per-drug difference (beta2020 − dcic2021)", xlim=xr, ylim=(-0.3, 3.6))
    ax.set_yticks([3.30, 2.15, 1.00])
    ax.set_yticklabels(["Pearson Δ", "Spearman Δ", "cosine Δ"], fontsize=7.5, fontweight="bold")
    panel(ax, "B", "Three metrics, aligned distributions")


def f2c(fig):
    inner = GridSpec(2, 3, figure=fig, wspace=0.30, hspace=0.46, left=0.155, right=0.985,
                     top=0.775, bottom=0.225)
    sel_c, _med, _nq, _med_all = f2c_selection()
    norm_c = f2c_norm(sel_c)
    _c_axes = []
    for k in range(3):
        for row in (0, 1):
            r, tag = sel_c[row * 3 + k]
            axk = fig.add_subplot(inner[row, k])
            _c_axes.append((axk, row))
            sub = d[d.drug == r.drug_id]
            axk.plot([0, 1], [0, 1], "--", color=GRAY, lw=0.8)
            axk.scatter(sub.pearson_B_dcic2021, sub.pearson_A_beta2020, s=9,
                        color=F2C_CMAP(norm_c(float(r.delta))), alpha=0.8)
            axk.set(xticks=[0, 1], yticks=[0, 1], xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
            if row == 0:
                axk.set_xticklabels([])
            if k > 0:
                axk.set_yticklabels([])
            if k == 0:                               # y axis, never overwritten below
                axk.set_ylabel("beta2020", fontsize=7.5, labelpad=1)
            if row == 1:
                axk.set_xlabel("dcic2021", fontsize=7.5)
            axk.set_title(r.drug_id, fontsize=6.4, pad=3, loc="left", color="#25333D")
            axk.text(0.04, 0.94, "Δ=%+.4f" % float(r.delta), transform=axk.transAxes,
                     ha="left", va="top", fontsize=6.4, color=DARK)
    # The selection rule is stated on the figure, not as "high Δ" / "Δ ≈ median"
    # row tags: those two words were an unsourced editorial label, and the subset
    # restriction (>= 20 cell lines) changes which median the bottom row sits on.
    _c_pos = [(a.get_position(), row) for a, row in _c_axes]
    f2c_colorbar(fig, norm_c, [0.355, 0.085, 0.40, 0.022], 6.8, 6.2)
    fig.text(0.02, 0.965, "C  Exemplar drugs: per-cell-line scores", fontsize=10,
             fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.895, "x: cross-gene Pearson, dcic2021     y: cross-gene Pearson, beta2020"
                           "     colour: per-drug Δ",
             fontsize=7.2, color=DARK, ha="left", va="top")
    _cbb = _c_axes[-1][0].get_position()
    fig.text(_cbb.x0, _cbb.y0 - 0.045,
             "C: top row = 3 largest Δ; bottom row = 3 closest to the median\n"
             "of the %d drugs with ≥20 cell lines (median Δ = %.4f;\n"
             "over all 2,037 drugs it would be %.4f)." % (_nq, _med, _med_all),
             ha="left", va="top", fontsize=6.6, color=GRAY, linespacing=1.5)


def f2d(ax):
    NAME = {"ciger": "CIGER", "deepce": "DeepCE", "multidcp": "MultiDCP", "pertdit": "PertDiT",
            "prnet": "PRnet", "transigen": "TranSiGen", "xpert": "XPert"}
    e = ext.copy()
    e["nm"] = e.model.str.lower().str.strip().str.replace(" ", "").map(NAME).fillna(e.model)
    e = e.sort_values("delta")
    for j, row in enumerate(e.itertuples()):
        ax.errorbar(row.delta, j, xerr=[[row.delta - row.ci_low], [row.ci_high - row.delta]],
                    fmt="o", color=BLUE, ms=5, capsize=3)
        ax.annotate("%.4f" % row.delta, xy=(row.ci_high, j), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(len(e)))
    ax.set_yticklabels(list(e.nm), fontsize=7.5)
    ax.set(xlabel="Within-model difference (own − dcic2021)", xlim=(0.0, 0.135),
           ylim=(-0.8, len(e) - 0.2))
    ax.text(0.985, 0.06, "each source: own evaluation set", transform=ax.transAxes,
            ha="right", fontsize=7.5, color=GRAY)
    panel(ax, "D", "Seven prediction sources")


def f3a(ax):
    for k, (lab, col, c) in enumerate([("beta-trained" + chr(10) + "× beta2020",
                                        "beta_trained__beta2020", BLUE),
                                       ("beta-trained" + chr(10) + "× dcic2021",
                                        "beta_trained__dcic2021", LIGHT),
                                       ("dcic-trained" + chr(10) + "× beta2020",
                                        "dcic_trained__beta2020", ORANGE),
                                       ("dcic-trained" + chr(10) + "× dcic2021",
                                        "dcic_trained__dcic2021", "#E2A176")]):
        v = pd3[col].to_numpy()
        vp = ax.violinplot(v, positions=[k], showextrema=False, widths=0.7)
        for body in vp["bodies"]:
            body.set_facecolor(c); body.set_alpha(0.35)
        med, mean = float(np.median(v)), float(v.mean())
        ax.plot([k - 0.10, k + 0.10], [med] * 2, color=DARK, lw=1.6)          # median, solid
        ax.plot([k - 0.13, k + 0.13], [mean] * 2, color="k", lw=1.1, ls="--")  # mean, dashed
        ax.annotate("%.3f" % mean, xy=(k, mean), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=7.5, color="k",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        ax.annotate("%.3f" % med, xy=(k, med), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=7.5, color=DARK,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax.set(xticks=range(4), xticklabels=[s[0] for s in
           [("beta-tr." + chr(10) + "× beta",), ("beta-tr." + chr(10) + "× dcic",),
            ("dcic-tr." + chr(10) + "× beta",), ("dcic-tr." + chr(10) + "× dcic",)]],
           ylabel="Drug-level cross-gene Pearson", ylim=(-0.03, 0.95))
    panel(ax, "A", "Drug-level distributions per state")


def f3b(fig):
    from scipy.stats import spearmanr
    inner = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.045], wspace=0.05,
                     left=0.15, right=0.95, top=0.88, bottom=0.16)
    ax = fig.add_subplot(inner[0, 0]); axc = fig.add_subplot(inner[0, 1])
    hb = ax.hexbin(perdrug.rank_beta, perdrug.rank_dcic, gridsize=30, cmap="Blues",
                   mincnt=1, linewidths=0.1, xscale="log", yscale="log")
    ax.plot([0.8, 2500], [0.8, 2500], "--", color=ORANGE, lw=1.2)
    rho = float(spearmanr(perdrug.rank_beta, perdrug.rank_dcic).statistic)
    ax.text(0.05, 0.85, "Spearman \u03c1 = %.3f" % rho, transform=ax.transAxes, fontsize=8)
    for r, c in [(10, BLUE), (50, GREEN)]:
        ax.axvline(r, color=c, lw=0.9, ls=":")
        ax.axhline(r, color=c, lw=0.9, ls=":")
    ax.text(11, 2400, "top-10", fontsize=6.5, color=BLUE)
    ax.text(55, 2400, "top-50", fontsize=6.5, color=GREEN)
    ax.set(xscale="log", yscale="log", xlabel="Rank under beta2020", ylabel="Rank under dcic2021")
    panel(ax, "B", "Rank-rank density (2,037 drugs)")
    cb = fig.colorbar(hb, cax=axc)
    cb.set_label("# drugs", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)


def f3c(ax):
    v = np.sort(perdrug.delta.to_numpy())
    ax.hist(v, bins=48, color="#E8EEF2", edgecolor=GRAY, lw=0.4)
    rug_sets = [(perdrug[perdrug.in_top10_beta & perdrug.in_top10_dcic].delta, "#6B5B95",
                 "in both top-10"),
                (perdrug[perdrug.in_top10_beta & ~perdrug.in_top10_dcic].delta, BLUE,
                 "beta top-10 only"),
                (perdrug[~perdrug.in_top10_beta & perdrug.in_top10_dcic].delta, ORANGE,
                 "dcic top-10 only")]
    for k2, (vv, c, lab) in enumerate(rug_sets):
        ax.plot(vv, np.full(len(vv), -14), "|", color=c, ms=7, mew=1.6)
        ax.annotate("%s (n = %d)" % (lab, len(vv)),
                    xy=(0.0 if not len(vv) else float(np.median(vv)), -28 - k2 * 9),
                    fontsize=6.5, color=c, ha="left",
                    xytext=(-0.05 if k2 == 0 else (0.18 if k2 == 1 else 0.40), -30 - k2 * 8),
                    textcoords="data")
    ax.set(xlabel="Per-drug Pearson difference (\u03b2 \u2212 dcic)", ylabel="# drugs",
           ylim=(-45, None))
    panel(ax, "C", "Shortlist membership vs shift")


def f3d(ax):
    pool = perdrug[perdrug.rank_beta <= 50].copy()
    pool["drop"] = pool.rank_dcic - pool.rank_beta
    exo = pool.nlargest(5, "drop").reset_index(drop=True)
    for j, r in exo.iterrows():
        ax.plot([0, 1], [r.pcc_beta, r.pcc_dcic], "-o", color=BLUE, ms=5, lw=1.3,
                alpha=0.5 + 0.5 * (j == 0))
    ax.annotate("5 largest rank drops" + chr(10) + "within beta top-50", xy=(0, 0.95),
                fontsize=8, color=GRAY, va="top")
    ax.set(xticks=[0, 1], xticklabels=["beta2020", "dcic2021"], xlim=(-0.08, 1.35),
           ylabel="Cross-gene Pearson", ylim=(0, 1))
    panel(ax, "D", "Exemplar: shortlist collapse")


def f4a(ax):
    ax.plot(topk.k, topk.retention, "o-", color=BLUE, ms=5, label="retention (observed)")
    ax.plot(topk.k, np.clip(topk.chance_expected_retention, 1e-4, None), "--", color=GRAY,
            label="retention (random k/N)")
    ax.set(xscale="log", xlabel="List size, k", ylabel="Top-k retention", ylim=(0, 1))
    ax2 = ax.twinx()
    ax2.plot(topk.k, topk.overlap, "s-", color=ORANGE, ms=4, label="shared candidates")
    ax2.set_ylim(0, max(topk.overlap) * 1.25)
    ax2.set_ylabel("Shared candidates", color=ORANGE, fontsize=7.5)
    ax2.tick_params(axis="y", labelsize=7, colors=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax.text(0.03, 0.95, "k = 10:" + chr(10) + "3 of 10 shared", transform=ax.transAxes,
            va="top", fontsize=8, color=GRAY)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, -.24), ncol=3, fontsize=7.5)
    panel(ax, "A", "Top-k retention and shared candidates")


def f4b(ax):
    ax.axis("off")
    tb = perdrug[perdrug.in_top10_beta].sort_values("rank_beta")
    td = perdrug[perdrug.in_top10_dcic].sort_values("rank_dcic")
    shared = set(tb.drug_id) & set(td.drug_id)
    XL, XR = 0.30, 0.70
    yb = {dr: 10 - i for i, dr in enumerate(tb.drug_id)}
    yd = {dr: 10 - i for i, dr in enumerate(td.drug_id)}
    exit_y = ent_y = 0
    for dr in tb.drug_id:
        ax.plot(XL, yb[dr], "o", color=BLUE, ms=4, zorder=3)
        ax.text(XL - 0.015, yb[dr], dr, ha="right", va="center", fontsize=6)
        if dr in shared:
            mx = (XL + XR) / 2
            p = MPath([(XL, yb[dr]), (mx, yb[dr]), (mx, yd[dr]), (XR, yd[dr])],
                      [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
            ax.add_patch(mpatches.PathPatch(p, fill="none", lw=1.5, color=BLUE, alpha=0.6))
        else:
            exit_y -= 1
            ax.plot(XR, exit_y - 0.6, "x", color=GRAY, ms=3.5, mew=1)
            mx = (XL + XR) / 2
            p = MPath([(XL, yb[dr]), (mx, yb[dr]), (mx, exit_y - 0.6), (XR, exit_y - 0.6)],
                      [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
            ax.add_patch(mpatches.PathPatch(p, fill="none", lw=1.2, color=GRAY, alpha=0.45))
    for dr in td.drug_id:
        ax.plot(XR, yd[dr], "o", color=ORANGE, ms=4, zorder=3)
        ax.text(XR + 0.015, yd[dr], dr, ha="left", va="center", fontsize=6)
        if dr not in shared:
            ent_y -= 1
            ax.plot(XL, ent_y - 0.6, "x", color=GRAY, ms=3.5, mew=1)
            mx = (XL + XR) / 2
            p = MPath([(XL, ent_y - 0.6), (mx, ent_y - 0.6), (mx, yd[dr]), (XR, yd[dr])],
                      [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
            ax.add_patch(mpatches.PathPatch(p, fill="none", lw=1.2, color=ORANGE, alpha=0.45))
    ax.text(XL, 10.9, "beta2020", ha="center", fontsize=8.5, weight="bold", color=BLUE)
    ax.text(XR, 10.9, "dcic2021", ha="center", fontsize=8.5, weight="bold", color=ORANGE)
    ax.text(0.5, 12.4, "3 shared", ha="center", fontsize=8.5, color=BLUE, weight="bold")
    ax.set_xlim(0.03, 0.97); ax.set_ylim(-4.2, 13.2)
    panel(ax, "B", "Top-10 union rank flow")


def _membership():
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
    return mat


def f4c(fig):
    mat = _membership()
    inner = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.055], left=0.215, right=0.97,
                     top=0.90, bottom=0.10, hspace=0.42)
    ax = fig.add_subplot(inner[0, 0])
    axc = fig.add_subplot(inner[1, 0])
    num = list(mat.columns[1:])
    vals = mat[num].to_numpy(dtype=float)
    im_data = np.ma.masked_invalid(np.where(np.isnan(vals), np.nan, np.log10(np.maximum(vals, 1))))
    ax.set_facecolor("#EEEEEE")
    imm = ax.imshow(im_data, cmap="YlOrBr_r", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(num)), num, fontsize=8, rotation=18, ha="right")
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(list(mat.drug_id), fontsize=4.6)
    ax.tick_params(axis="y", pad=2)
    cb = fig.colorbar(imm, cax=axc, orientation="horizontal", ticks=[0, 1, 2, 3])
    cb.set_label("rank (log10 scale);  grey = not shortlisted", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    panel(ax, "C", "Shortlist membership (rank per context)")


def _ctx():
    J2 = J.copy()
    J2["label"] = J2.panel.map(CTX)
    J2["shared"] = (20 * J2.top10_jaccard / (1 + J2.top10_jaccard)).round().astype(int)
    return J2.sort_values("spearman", ascending=True).reset_index(drop=True)


def f4d(fig):
    J2 = _ctx()
    y = np.arange(len(J2))
    gsD = GridSpec(2, 1, figure=fig, height_ratios=[1.25, 1], hspace=0.55,
                   left=0.17, right=0.92, top=0.91, bottom=0.07)
    inner = gsD[0, 0].subgridspec(1, 3, width_ratios=[1, 1, 0.10], wspace=0.06)
    axL = fig.add_subplot(inner[0, 0]); axR = fig.add_subplot(inner[0, 1])
    axc = fig.add_subplot(inner[0, 2])
    p = axc.get_position()
    axc.set_position([p.x0, p.y0 + p.height * 0.56, p.width, p.height * 0.40])
    axc2 = fig.add_axes([p.x0, p.y0, p.width, p.height * 0.40])
    imL = axL.imshow(J2[["spearman"]].to_numpy(), cmap="Blues", vmin=0, vmax=0.7, aspect="auto")
    imR = axR.imshow(J2[["shared"]].to_numpy(), cmap="Oranges", vmin=0, vmax=7, aspect="auto")
    axL.set_xticks([0], ["Spearman \u03c1"], fontsize=8)
    axL.set_yticks(y); axL.set_yticklabels([SHORT.get(x, x) for x in J2.label], fontsize=8)
    axR.set_xticks([0], ["top-10 shared"], fontsize=8); axR.set_yticks([])
    for a in (axL, axR):
        a.set_xticks(np.arange(-.5, 1, 1), minor=True)
        a.grid(which="minor", color="white", lw=1)
        a.tick_params(which="minor", length=0)
        for sp in a.spines.values():
            sp.set_visible(False)
    cb1 = fig.colorbar(imL, cax=axc, ticks=[0, 0.35, 0.7]); cb1.ax.tick_params(labelsize=6.5)
    cb1.set_label("Spearman \u03c1", fontsize=7, labelpad=1)
    cb2 = fig.colorbar(imR, cax=axc2, ticks=[0, 3, 7]); cb2.ax.tick_params(labelsize=6.5)
    cb2.set_label("top-10 shared", fontsize=7, labelpad=1)
    panel(axL, "D", "Context summary")
    axE = fig.add_subplot(gsD[1, 0])
    cmap = plt.get_cmap("tab10")
    for i, r in J2.iterrows():
        axE.scatter(r.spearman, r.shared, s=42, color=cmap(int(i) % 10), zorder=3)
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=5, color=cmap(int(i) % 10),
                          label=SHORT.get(r.label, r.label)) for i, r in J2.iterrows()]
    axE.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=5,
               frameon=False, fontsize=7, handletextpad=0.25, columnspacing=0.8)
    axE.text(.03, .97, "context rank Spearman \u03c1 (x) vs top-10 shared (y)",
             transform=axE.transAxes, va="top", fontsize=8, color=GRAY)
    axE.set(ylabel="Top-10 shared", xlabel="Context rank Spearman \u03c1",
            xlim=(0.39, 0.655), ylim=(-0.7, 8.0))


def f5a(fig):
    cc = d.groupby("cell").agg(pb=("pearson_A_beta2020", "mean"),
                               pd_=("pearson_B_dcic2021", "mean"),
                               n=("pearson_delta", "size")).reset_index()
    cc["delta"] = cc.pb - cc.pd_
    cc = cc.sort_values("delta", ascending=False).reset_index(drop=True)
    inner = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 0.9, 0.5, 0.06], wspace=0.06,
                     left=0.17, right=0.88, top=0.90, bottom=0.16)
    vmax = float(np.percentile(np.r_[cc.pb.values, cc.pd_.values], 97))
    dmax = float(np.percentile(np.abs(cc.delta.values), 97))
    blocks = [("pb", "Blues", 0, vmax, "beta2020"), ("pd_", "Blues", 0, vmax, "dcic2021"),
              ("delta", "RdBu_r", -dmax, dmax, "Δ"), ("n", "Greens", 0, cc.n.max(), "responses")]
    first = None
    for k, (col, cmap, vmin, vvx, xlab) in enumerate(blocks):
        axk = fig.add_subplot(inner[0, k]); first = first or axk
        imk = axk.imshow(cc[[col]].to_numpy(), cmap=cmap, vmin=vmin, vmax=vvx, aspect="auto")
        if k == 0:
            axk.set_yticks(range(len(cc))); axk.set_yticklabels(list(cc.cell), fontsize=6)
        else:
            axk.set_yticks([])
        # All four block labels sat level. "responses" was rotated 25 degrees,
        # which read as a rendering fault rather than a deliberate stagger, and
        # nothing in the panel needs the extra width.
        axk.set_xticks([0], [xlab], fontsize=7.5, rotation=0, ha="center")
        axk.set_xticks(np.arange(-.5, 1, 1), minor=True)
        axk.grid(which="minor", color="white", lw=0.8)
        axk.tick_params(which="minor", length=0)
        for sp in axk.spines.values():
            sp.set_visible(False)
    axcb = fig.add_subplot(inner[0, 4])
    cb = fig.colorbar(imk, cax=axcb); cb.ax.tick_params(labelsize=6.5)
    cb.set_label("# responses", fontsize=7, labelpad=2)
    panel(first, "A", "Cell-line × reference landscape")


def _cc():
    cc = d.groupby("cell").agg(pb=("pearson_A_beta2020", "mean"),
                               pd_=("pearson_B_dcic2021", "mean"),
                               n=("pearson_delta", "size")).reset_index()
    cc["delta"] = cc.pb - cc.pd_
    return cc.sort_values("delta", ascending=False).reset_index(drop=True)


def f5b(ax):
    cc = _cc()
    y = np.arange(len(cc))[::-1]
    ax.hlines(y, 0, cc.delta, color="#D5DEE4", lw=1)
    ax.scatter(cc.delta, y, s=7 + 16 * (cc.n / cc.n.max()), color=BLUE, alpha=0.85)
    ax.axvline(cc.delta.mean(), ls="--", color=ORANGE, lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels(list(cc.cell), fontsize=6)
    ax.set(xlabel="Mean Pearson difference (β − dcic)", xlim=(0, 0.62),
           ylabel="Cell lines (sorted by Δ)")
    ax.text(.97, .52, "mean %.3f" % cc.delta.mean() + chr(10) + "dot size ∝ responses",
            transform=ax.transAxes, ha="right", va="center", **ink(7.5))
    panel(ax, "B", "Ordered cell-line effect")


def f5c(fig):
    cc = _cc()
    cov = d.groupby("drug").size().sort_values(ascending=False)
    keep = cov[cov >= 25].index[:60]
    mat = d[d.drug.isin(keep)].pivot_table(index="drug", columns="cell", values="pearson_delta")
    order = leaves_list(linkage(mat.apply(lambda r: r.fillna(r.mean()), axis=1).fillna(0).to_numpy(),
                                method="average"))
    mat = mat.iloc[order].reindex(columns=cc.cell.tolist())
    dmax = float(np.nanpercentile(np.abs(mat.to_numpy()), 97))
    matT = mat.T
    ax = fig.add_axes([0.24, 0.18, 0.62, 0.72])
    imm = ax.imshow(matT.to_numpy(), vmin=-dmax, vmax=dmax, cmap="RdBu_r", aspect="auto")
    imm.set_rasterized(True)
    ax.set_yticks(range(matT.shape[0])); ax.set_yticklabels(list(matT.index), fontsize=6)
    xsel = [i for i in range(matT.shape[1]) if i % 6 == 0]
    ax.set_xticks(xsel)
    ax.set_xticklabels([matT.columns[i] for i in xsel], fontsize=6, rotation=45, ha="right")
    ax.set_xlabel(str(matT.shape[1]) + " most-covered drugs, clustered (every 6th labelled)",
                  fontsize=7.5)
    ax.legend(handles=[mpatches.Patch(color="#F0F0F0", label="not measured")],
              loc="lower right", fontsize=6.5, frameon=False)
    panel(ax, "C", "Δ matrix (rows = cell lines)")


def f5d(fig):
    cc = _cc()
    ccx = cc[cc.n >= 20].reset_index(drop=True)
    # Four exemplar cell lines, taken as the two largest and two smallest mean
    # cross-reference differences among cell lines with >= 20 scored drugs. They
    # are illustrative, not extremes to be read as: earlier rounds labelled them
    # "most sensitive" through "most stable", then "largest Δ" through
    # "smallest Δ". Both are superlative claims about a spread the panel does not
    # test, so the rows now carry a neutral "example" tag and nothing else.
    sens = ccx.iloc[:2]["cell"].tolist(); stab = ccx.iloc[-2:]["cell"].tolist()
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
    inner = GridSpec(6, 1, figure=fig, hspace=0.55, left=0.10, right=0.97, top=0.91, bottom=0.09)
    for k, (lab, v, c, letter) in enumerate(rows):
        axe = fig.add_subplot(inner[k, 0])
        v = v[np.isfinite(v)]
        kde_ridge(axe, v, 0.0, 0.8, c, xr=(-0.6, 1.0))
        axe.axvline(0, color=GRAY, lw=0.8, ls="--")
        axe.annotate(lab + "   median %.3f" % np.median(v), xy=(-0.57, 0.55), **ink(7.5))
        axe.set_yticks([]); axe.set_xlim(-0.6, 1.0); axe.set_ylim(-0.15, 1.0)
        if k < 5:
            axe.set_xticklabels([])
        else:
            axe.set_xlabel("Per-drug / per-cell Pearson difference", fontsize=8)
        if letter:
            panel(axe, letter, "Unit-level and exemplar distributions", fs_title=9)


STATES = [("frozen_CERES", "CERES", "C → C"), ("frozen_CERES", "Chronos", "C → H"),
          ("frozen_Chronos", "CERES", "H → C"), ("frozen_Chronos", "Chronos", "H → H")]


def f6a(ax):
    for k, (mk, ev, lab) in enumerate(STATES):
        v = pcell[(pcell["model"] == mk) & (pcell["eval"] == ev)].r_raw.to_numpy()
        vp = ax.violinplot(v, positions=[k], showextrema=False, widths=0.75)
        for body in vp["bodies"]:
            body.set_facecolor(BLUE); body.set_alpha(0.30)
        ax.plot([k - 0.10, k + 0.10], [np.median(v)] * 2, color=DARK, lw=1.6)
        ax.annotate("%.4f" % np.median(v), xy=(k, np.median(v)), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set(xticks=range(4), xticklabels=[s[2] for s in STATES],
           ylabel="Raw by-cell Pearson (136 cells)", ylim=(0.88, 0.97))
    panel(ax, "A", "Frozen model — raw score distributions")


def f6b(ax):
    for k, (arm, ref, lab) in enumerate([("ceres", "CERES", "C → C"), ("ceres", "Chronos", "C → H"),
                                         ("chronos", "CERES", "H → C"), ("chronos", "Chronos", "H → H")]):
        sub = dpaired[(dpaired.arm == arm) & (dpaired.reference == ref)]
        dv = sub.paired_median_delta_model_minus_baseline.to_numpy() * 1000
        jit = np.random.default_rng(7).uniform(-0.08, 0.08, len(dv))
        ax.scatter(k + jit, dv, s=22, color=BLUE, alpha=0.75)
    ax.axhline(0, color=GRAY, lw=1)
    # Wording is the author's. Two earlier versions were rejected: "the gene-mean
    # baseline already explains raw scores" asserts a mechanism the panel cannot
    # support, and "model < baseline in all four states" inflates a gap that is
    # 0.0008 in Pearson units -- 0.09% of the ~0.93 scores, i.e. similar. The
    # caption carries the numbers; the panel states the qualitative reading.
    ax.annotate("The model and gene-mean baseline" + chr(10) +
                "obtain similar raw Pearson scores.",
                xy=(1.5, 0.34), ha="center", va="top", linespacing=1.5, **ink(7.5))
    ax.set(xticks=range(4), xticklabels=[s[2] for s in STATES],
           ylabel="Model − baseline (×1e-3 Pearson)", ylim=(-1.4, 0.46))
    panel(ax, "B", "Paired difference view")


def f6c(ax):
    for k, (mk, ev, lab) in enumerate(STATES):
        v = pcell[(pcell["model"] == mk) & (pcell["eval"] == ev)].r_shift.to_numpy()
        vp = ax.violinplot(v, positions=[k], showextrema=False, widths=0.75)
        for body in vp["bodies"]:
            body.set_facecolor(GREEN); body.set_alpha(0.30)
        ax.plot([k - 0.10, k + 0.10], [np.median(v)] * 2, color=DARK, lw=1.6)
        ax.annotate("%+.4f" % np.median(v), xy=(k, np.median(v)), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set(xticks=range(4), xticklabels=[s[2] for s in STATES],
           ylabel="Residualized by-cell Pearson (136 cells)", ylim=(-0.03, 0.05))
    panel(ax, "C", "Frozen model — residualized distributions")


SERIES = [("ridge_CERES", "CERES", BLUE, "-", "Expr Ridge → CERES"),
          ("ridge_Chronos", "Chronos", BLUE, "--", "Expr Ridge → Chronos"),
          ("frozen_CERES", "CERES", ORANGE, "-", "Frozen → CERES"),
          ("frozen_Chronos", "Chronos", ORANGE, "--", "Frozen → Chronos"),
          ("ridge_perm", "CERES", GRAY, "-", "Permuted Ridge")]


def f6d(ax):
    for mk, ev, c, ls, lab in SERIES:
        v = np.sort(S14[(S14["model"] == mk) & (S14["eval"] == ev)].r_across_cell.to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, lw=1.4, label=lab)
    ax.axvline(0, ls="--", color=GRAY, lw=1)
    ax.set(xlabel="Gene-wise across-cell Pearson r", ylabel="ECDF (17,393 genes)",
           xlim=(-0.10, 0.42), ylim=(0, 1))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2, fontsize=7)
    panel(ax, "D", "Gene-wise signal per model")


def f6e(ax):
    for mk, ev, c, ls, lab in SERIES:
        v = np.sort(pcell[(pcell["model"] == mk) & (pcell["eval"] == ev)].r_shift.to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, lw=1.4, label=lab)
    ax.axvline(0, ls="--", color=GRAY, lw=1)
    ax.set(xlabel="Residualized by-cell Pearson r", ylabel="ECDF (136 test cells)",
           xlim=(-0.06, 0.36), ylim=(0, 1))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2, fontsize=7)
    panel(ax, "E", "Positive control: by-cell distributions")


def f6f(ax):
    def boot_inter(ma, mb, col, n=10000, seed=11):
        a = pcell[(pcell["model"] == ma) & (pcell["eval"] == "CERES")].set_index("cell")[col]
        b = pcell[(pcell["model"] == ma) & (pcell["eval"] == "Chronos")].set_index("cell")[col]
        c = pcell[(pcell["model"] == mb) & (pcell["eval"] == "CERES")].set_index("cell")[col]
        dd = pcell[(pcell["model"] == mb) & (pcell["eval"] == "Chronos")].set_index("cell")[col]
        df = pd.concat([a, b, c, dd], axis=1); df.columns = ["a", "b", "c", "d"]
        v = (df.a - df.b - df.c + df.d).dropna().to_numpy()
        rng = np.random.default_rng(seed)
        return v, v[rng.integers(0, len(v), (n, len(v)))].mean(1)

    frozen_raw = dfrozen_boot["raw"].to_numpy(); frozen_res = dfrozen_boot["residualized"].to_numpy()
    _, ridge_raw_b = boot_inter("ridge_CERES", "ridge_Chronos", "r_raw")
    _, ridge_res_b = boot_inter("ridge_CERES", "ridge_Chronos", "r_shift")
    zeros = np.zeros(2000)
    MODELS6 = [("Ridge", BLUE, [(0 - 0.24, ridge_raw_b), (1 - 0.24, ridge_res_b)]),
               ("Frozen", ORANGE, [(0, frozen_raw), (1, frozen_res)]),
               ("Permuted", GRAY, [(0 + 0.24, zeros), (1 + 0.24, zeros)])]
    for mname, col, slots in MODELS6:
        for (x, draws) in slots:
            vp = ax.violinplot(draws, positions=[x], showextrema=False, widths=0.22)
            for body in vp["bodies"]:
                body.set_facecolor(col); body.set_alpha(0.30)
            ax.plot([x - 0.05, x + 0.05], [np.median(draws)] * 2, color=DARK, lw=1.4)
        xs = [slots[0][0], slots[1][0]]; ms = [np.median(slots[0][1]), np.median(slots[1][1])]
        ax.plot(xs, ms, color=col, lw=1.1, alpha=0.7)
        dy = {"Ridge": 0.035, "Frozen": 0.022, "Permuted": -0.028}[mname]
        ax.annotate(mname, xy=(xs[1], ms[1] + dy), ha="center", fontsize=7.5, color=col)
    ax.axhline(0, ls="--", color=GRAY, lw=1)
    ax.set(xticks=[0, 1], xticklabels=["Raw", "Residualized"],
           ylabel="Training × reference interaction (bootstrap)", xlim=(-0.5, 1.5),
           ylim=(-0.05, 0.42))
    panel(ax, "F", "Interaction: collapse vs rescue")


def f7a(ax):
    GT = [("rRR", "RR: RNASeQC → RNASeQC", BLUE),
          ("rRE", "RE: RNASeQC → RSEM", LIGHT),
          ("rER", "ER: RSEM → RNASeQC", ORANGE),
          ("rEE", "EE: RSEM → RSEM", "#E2A176")]
    for k, (base, lab, c) in enumerate(GT):
        v = gtex[base].to_numpy()
        y0 = 3.3 - k * 1.05
        kde_ridge(ax, v, y0, 0.85, c, xr=(0.55, 1.0))
        m = float(np.median(v))
        ax.plot([m], [y0 + 0.1], "o", color=c, ms=4)
        ax.plot([0.5565, 0.5665], [y0 + 0.70, y0 + 0.70], color=c, lw=2.6,
                solid_capstyle="butt")
        ax.text(0.5695, y0 + 0.70, lab, fontsize=7, color=DARK, ha="left", va="center")
        ax.annotate("%.3f" % m, xy=(m, y0 + 0.26), ha="center", va="bottom",
                    fontsize=7, color=DARK)
    ax.set_yticks([])
    ax.set(xlabel="Per-sample across-gene Pearson", xlim=(0.55, 1.0), ylim=(-0.3, 4.6))
    panel(ax, "A", "Per-sample distributions (Pearson)")


def f7b(ax):
    GT = [("sp_rRR", "RR: RNASeQC → RNASeQC", BLUE),
          ("sp_rRE", "RE: RNASeQC → RSEM", LIGHT),
          ("sp_rER", "ER: RSEM → RNASeQC", ORANGE),
          ("sp_rEE", "EE: RSEM → RSEM", "#E2A176")]
    for k, (base, lab, c) in enumerate(GT):
        v = gtex[base].to_numpy()
        y0 = 3.3 - k * 1.05
        kde_ridge(ax, v, y0, 0.85, c, xr=(0.55, 1.0))
        m = float(np.median(v))
        ax.plot([m], [y0 + 0.1], "o", color=c, ms=4)
        ax.plot([0.5565, 0.5665], [y0 + 0.70, y0 + 0.70], color=c, lw=2.6,
                solid_capstyle="butt")
        ax.text(0.5695, y0 + 0.70, lab, fontsize=7, color=DARK, ha="left", va="center")
        ax.annotate("%.3f" % m, xy=(m, y0 + 0.26), ha="center", va="bottom",
                    fontsize=7, color=DARK)
    ax.set_yticks([])
    ax.set(xlabel="Per-sample across-gene Spearman", xlim=(0.55, 1.0), ylim=(-0.3, 4.6))
    panel(ax, "B", "Per-sample distributions (Spearman)")


def f7c(ax):
    for k, (col, c, lab) in enumerate([("adv_R", BLUE, "RNASeQC-trained"),
                                       ("adv_E", ORANGE, "RSEM-trained")]):
        v = donor[col].to_numpy(); y0 = 1 - k
        kde_ridge(ax, v, y0 + 0.06, 0.55, c, xr=(-0.005, 0.0495))
        jit = np.random.default_rng(5).uniform(-0.012, 0.012, len(v))
        ax.scatter(v, np.full(len(v), y0) + jit - 0.10, s=3.5, color=c, alpha=0.4)
        med, lo, hi = np.median(v), *np.percentile(v, [25, 75])
        ax.plot([med, med], [y0 - 0.10, y0 + 0.66], color=DARK, lw=1.5)
        ax.annotate("median %.3f" % med + chr(10) + "IQR [%.3f, %.3f]" % (lo, hi) + chr(10) +
                    "%.0f%% > 0" % ((v > 0).mean() * 100), xy=(0.054, y0),
                    ha="left", va="center", fontsize=7, color=c)
    ax.set_yticks([])
    ax.set(xlabel="Matched-reference advantage (per-donor median)", xlim=(-0.005, 0.080),
           ylim=(-0.5, 2.55))
    panel(ax, "C", "Donor-level advantage (raincloud)")


def f7d(ax):
    mat = gtex[["rRR", "rRE", "rER", "rEE"]].median().to_numpy().reshape(2, 2)
    ax.imshow(mat, vmin=.94, vmax=.99, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "%.4f" % mat[i, j], ha="center", va="center", fontsize=11,
                    color="white" if mat[i, j] > .975 else "#18242E")
    intr = float(mat[0, 0] - mat[0, 1] - mat[1, 0] + mat[1, 1])
    ax.set(xticks=[0, 1], xticklabels=["RNASeQC", "RSEM"], yticks=[0, 1],
           yticklabels=["RNASeQC", "RSEM"], xlabel="Evaluation product",
           ylabel="Tissue-mean source")
    # The "interaction = 0.07283" line under the heat map was removed. That same
    # quantity is the subject of panel E, which shows its distribution across
    # donors; stating it here as a bare number gave the panel two jobs at once.
    # The four cell scores above are the reference and stay. intr is still
    # computed because it is what panel E's own numbers come from.
    panel(ax, "D", "2×2 summary")


def f7e(ax):
    for k, (col, c, lab) in enumerate([("interaction_pearson", BLUE, "Pearson"),
                                       ("interaction_spearman", ORANGE, "Spearman")]):
        v = S11[col].to_numpy()
        vp = ax.violinplot(v, positions=[1 - k], vert=False, showextrema=False, widths=0.7)
        for body in vp["bodies"]:
            body.set_facecolor(c); body.set_alpha(0.35)
        med, lo, hi = np.median(v), *np.percentile(v, [2.5, 97.5])
        ax.plot([med], [1 - k], "o", color=c, ms=5)
        ax.plot([lo, hi], [1 - k, 1 - k], color=c, lw=1.4)
        ax.annotate("median %.5f" % med + chr(10) + "[%.5f, %.5f]" % (lo, hi),
                    xy=(med, 1 - k), xytext=(0, 26), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=c)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Pearson", "Spearman"], fontsize=8)
    ax.set(xlabel="Donor-median interaction (5,000 donor bootstrap)", xlim=(.055, .078),
           ylim=(-0.85, 2.0))
    panel(ax, "E", "Bootstrap uncertainty")


def organ_system(t):
    tl = t.lower()
    # Kidney/bladder FIRST: "Kidney Cortex" contains "cortex", so the brain test
    # below used to claim it and the tissue landed in Brain & CNS, sorting among
    # neural tissues by a Pearson interaction that has nothing to do with the
    # brain. Specific organ terms must be tested before generic cortex.
    if any(k in tl for k in ("kidney", "bladder")):
        return "Kidney & urinary"
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


SYS_COLOR = {"Brain & CNS": "#6B5B95", "Blood & immune": "#C65C27", "Heart": "#A63A32",
             "Muscle": "#8C564B", "Lung & airway": "#4C9AB0", "Digestive & liver": "#337F66",
             "Kidney & urinary": "#4E79A7", "Reproductive & breast": "#B07AA1",
             "Endocrine": "#D4A017", "Skin & soft tissue": "#A9865B",
             "Vascular & nerve": "#5C7A99", "Other": "#999999"}


def f7f(fig, which="rep", letter="F", fs_tick=6.5, fs_ylab=7.0, title=None,
        top=0.90, bottom=0.265, legend=True):
    E_all = S12.copy(); E_all["sys"] = E_all.tissue.map(organ_system)
    E_all = E_all.sort_values(["sys", "interaction_pearson"],
                              ascending=[True, False]).reset_index(drop=True)
    # main text: one representative tissue per organ system (highest Pearson interaction
    # within that system); the full 68-tissue version is Supplementary Figure S2
    if which == "all":
        E = E_all
    else:
        E = E_all.groupby("sys", sort=False).head(1).reset_index(drop=True)
    cols = ["interaction_pearson", "interaction_spearman", "n_donors", "adv_R", "adv_E"]
    # R = reference built from the RNASeQC quantification, E = from RSEM
    titles = ["Pearson\ninteraction", "Spearman\ninteraction", "n donors",
              "adv.\nRNASeQC (R)", "adv.\nRSEM (E)"]
    cb_labels = ["interaction", "interaction", "donors", "advantage", "advantage"]
    inner = GridSpec(2, len(cols) + 1, figure=fig, width_ratios=[0.3] + [1] * len(cols),
                     height_ratios=[1, 0.05], wspace=0.30, hspace=0.22, left=0.30,
                     right=0.965, top=top, bottom=bottom)
    # the organ-system strip must span exactly the heat-map row, not the colour-bar row too
    side = fig.add_subplot(inner[0, 0])
    side.imshow(np.arange(len(E))[:, None], aspect="auto",
                cmap=matplotlib.colors.ListedColormap([SYS_COLOR[s] for s in E.sys]))
    side.set_xticks([]); side.set_yticks([])
    for sp in side.spines.values():
        sp.set_visible(False)
    for k, col in enumerate(cols):
        axk = fig.add_subplot(inner[0, k + 1])
        vv = E[col].to_numpy()
        raw = vv.copy()
        if col == "n_donors":
            vv_norm = np.log10(vv)
        else:
            vv_norm = (vv - vv.min()) / max(vv.max() - vv.min(), 1e-12)
        cmap = "Greens" if col == "n_donors" else ("Blues" if "pearson" in col else
                                                   ("Oranges" if "spearman" in col else "Purples"))
        imk = axk.imshow(vv_norm[:, None], cmap=cmap, aspect="auto")
        axk.set_xticks([0], [titles[k]], fontsize=fs_tick, rotation=25, ha="right")
        axk.set_yticks([])
        axk.set_xticks(np.arange(-.5, 1, 1), minor=True)
        axk.grid(which="minor", color="white", lw=1)
        axk.tick_params(which="minor", length=0)
        for sp in axk.spines.values():
            sp.set_visible(False)
        # per-column colour bar: without it the reader cannot tell what the shade means
        cax = fig.add_subplot(inner[1, k + 1])
        cb = fig.colorbar(imk, cax=cax, orientation="horizontal")
        lo, hi = float(raw.min()), float(raw.max())
        if col == "n_donors":
            # imshow holds log10(values), so the colour bar is already in log space
            tk = np.unique(np.round(np.geomspace(max(lo, 1.0), hi, 3)).astype(int))
            cb.set_ticks(np.log10(tk))
            cb.set_ticklabels([str(int(t)) for t in tk])
        else:
            # imshow holds min-max normalised values -> the bar spans 0..1, so ticks go
            # in normalised space and carry the raw values as labels
            nd = 3 if hi < 0.2 else 4
            cb.set_ticks([0.0, 1.0])
            cb.set_ticklabels([("%." + str(nd) + "f") % lo, ("%." + str(nd) + "f") % hi])
        cb.ax.tick_params(labelsize=fs_tick - 1.4, length=1.6, pad=1.2, labelrotation=90)
        cb.set_label(cb_labels[k], fontsize=fs_tick - 0.4, labelpad=2.6)
        cb.outline.set_linewidth(0.4)
    side.set_yticks(range(len(E)))
    side.set_yticklabels(E.tissue, fontsize=fs_ylab)
    side.yaxis.tick_left()
    panel(side, letter, title or ("%d representative tissues" % len(E)))
    # organ-system legend under the heat map: the colour strip on the left is otherwise
    # unexplained. The heat map itself is untouched; the canvas is simply taller.
    if legend:
        for i, _sy in enumerate(dict.fromkeys(E.sys)):
            rr, cc = divmod(i, 3)
            _lx = 0.30 + cc * 0.222
            _ly = bottom - 0.098 - rr * 0.0455
            fig.patches.append(mpatches.Rectangle(
                (_lx, _ly), 0.0062, 0.0165, transform=fig.transFigure,
                facecolor=SYS_COLOR[_sy], edgecolor="none", zorder=5))
            fig.text(_lx + 0.0098, _ly + 0.0082, _sy, fontsize=fs_ylab - 0.4,
                     va="center", ha="left", color=DARK, zorder=5)


# ==================================================================== export
RECTS = {
    "Figure5_B_caterpillar": [0.24, 0.06, 0.72, 0.90],
    "Figure4_A_topk": [0.18, 0.36, 0.70, 0.54],
    "Figure6_D_genewise_signal": [0.20, 0.36, 0.77, 0.54],
    "Figure6_E_positive_control": [0.20, 0.36, 0.77, 0.54],
    "Figure7_D_2x2_summary": [0.33, 0.24, 0.60, 0.62],
}


def export(name, draw, w_mm, h_mm, subplots=False):
    fig = plt.figure(figsize=(w_mm / MM, h_mm / MM))
    if subplots:
        draw(fig)
    else:
        ax = fig.add_axes(RECTS.get(name, [0.20, 0.20, 0.77, 0.70]))
        draw(ax)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        try:
            fig.savefig(OUT / (name + ext), **kw)
        except OSError as e:
            print("  ! locked, skipped", name + ext, e)
    plt.close(fig)
    print("  ", name)


JOBS = [
    ("Figure2_A_density_scatter", f2a, 120, 105, False),
    ("Figure2_B_three_metrics", f2b, 120, 85, False),
    ("Figure2_C_exemplar_drugs", f2c, 170, 110, True),
    ("Figure2_D_seven_sources", f2d, 120, 85, False),
    ("Figure3_A_drug_level_distributions", f3a, 120, 90, False),
    ("Figure3_B_rank_rank_density", f3b, 140, 110, True),
    ("Figure3_C_shortlist_membership", f3c, 120, 95, False),
    ("Figure3_D_shortlist_collapse", f3d, 120, 90, False),
    ("Figure4_A_topk", f4a, 130, 100, False),
    ("Figure4_B_rank_flow", f4b, 120, 100, False),
    ("Figure4_C_membership_matrix", f4c, 130, 130, True),
    ("Figure4_D_context_summary", f4d, 140, 180, True),
    ("Figure5_A_cell_line_landscape", f5a, 150, 200, True),
    ("Figure5_B_caterpillar", f5b, 135, 200, False),
    ("Figure5_C_delta_matrix", f5c, 150, 200, True),
    ("Figure5_D_unit_level_and_exemplars", f5d, 140, 190, True),
    ("Figure6_A_frozen_raw", f6a, 120, 95, False),
    ("Figure6_B_paired_difference", f6b, 120, 95, False),
    ("Figure6_C_frozen_residualized", f6c, 120, 95, False),
    ("Figure6_D_genewise_signal", f6d, 120, 110, False),
    ("Figure6_E_positive_control", f6e, 120, 110, False),
    ("Figure6_F_interaction", f6f, 130, 100, False),
    ("Figure7_A_pearson_ridgelines", f7a, 130, 105, False),
    ("Figure7_B_spearman_ridgelines", f7b, 130, 105, False),
    ("Figure7_C_donor_raincloud", f7c, 130, 95, False),
    ("Figure7_D_2x2_summary", f7d, 115, 100, False),
    ("Figure7_E_bootstrap", f7e, 130, 100, False),
    ("Figure7_F_tissue_matrix", f7f, 165, 258, True),
    # full 68-tissue version, moved out of the main text into Supplementary Figure S2
    ("Supplementary_Figure_S2_GTEx_tissue_matrix",
     lambda fig: f7f(fig, which="all", letter="S2", fs_tick=6.5, fs_ylab=5.8,
                     title="All 68 tissues, grouped by organ system", top=0.90, bottom=0.245),
     155, 258, True),
]

if __name__ == "__main__":
    for name, draw, w, h, sub in JOBS:
        export(name, draw, w, h, sub)
    print("rendered", len(JOBS), "panels ->", OUT)
