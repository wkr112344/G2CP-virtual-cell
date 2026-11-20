# DepMap positive control: expression-PCA -> multi-output Ridge
#   Purpose: on the same Figure 6 split (train 636 / test 136), test whether a predictor that genuinely
#   relies on cell-state features retains a positive signal under residualized evaluation, and contrast it with the frozen model.
#
# Metric definitions (key):
#   by-cell raw        : per test cell, across-gene Pearson(pred, truth)
#   by-cell shift      : subtract the per-gene training mean from both -> [mathematically identical to raw], a check only
#   by-cell zres       : (pred-mu)/sd vs (truth-mu)/sd with mu/sd from training cells only -> the real "gene-level background removal"
#   across-cell raw    : per gene, across test cells Pearson(pred, truth) (invariant to per-gene shift)
#   across-cell zres   : as above but with per-gene z first (equivalent to raw, being invariant to per-gene affine transforms); a check
#
# Objects:
#   mean_baseline (negative/basic control): prediction = per-gene training mean (a constant vector)
#   ridge_CERES / ridge_Chronos (positive control): X_TPM -> PCA(100) -> Ridge, with training targets CERES/Chronos respectively
#   ridge_perm (label-permutation negative control): same pipeline after shuffling the training Y row labels
#   frozen_G2CP (reference): the existing 5-seed averaged predictions from depmap_analysis_ready, one per arm
#
# Output: results/*.csv|json, figures/panel_ABC_positive_control.png
import os, sys, json, time
import numpy as np
from scipy.stats import rankdata   # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AL = r"C:/Users/wkr20/Desktop/depmap_second_corpus/work/DeepDEP_training_data/aligned_21Q2_908_3omics"
FR = r"C:/Users/wkr20/WorkBuddy/Claw/depmap_analysis_ready_20260929"
R = lambda *a: os.path.join(BASE, *a)
N_PC = 100
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
N_BOOT = 5000
SEED = 20260930
MIN_CELLS_PER_GENE = 30


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def load():
    X = np.load(os.path.join(AL, "X_TPM.npy")).astype(np.float32)
    Ycer = np.load(os.path.join(AL, "Y_CERES.npy")).astype(np.float32)
    Ychr = np.load(os.path.join(AL, "Y_Chronos.npy")).astype(np.float32)
    mask = np.load(os.path.join(AL, "common_observed_mask.npy"))
    split = json.load(open(os.path.join(AL, "split.json")))
    idx = json.load(open(os.path.join(FR, "index", "depmap_index.json")))
    return X, Ycer, Ychr, mask, split, idx


def corr_rows(P, T, M):
    """Per-row (=per-cell) across-column Pearson. A zero-variance prediction is recorded as 0 by definition (no linear information)."""
    out = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):
        m = M[i] & np.isfinite(T[i]) & np.isfinite(P[i])
        if m.sum() < 30:
            continue
        p = P[i][m].astype(np.float64); t = T[i][m].astype(np.float64)
        p = p - p.mean(); t = t - t.mean()
        if (p * p).sum() == 0 or (t * t).sum() == 0:
            out[i] = 0.0
            continue
        d = np.sqrt((p * p).sum() * (t * t).sum())
        if d > 0:
            out[i] = (p * t).sum() / d
    return out


def sanitize_mu_sd(mu, sd, mtr, Ytr):
    """NaN mu replaced by 0 and NaN/0 sd by 1 -- so shift/zres and raw use the same gene set."""
    mu = np.nan_to_num(np.asarray(mu, dtype=np.float64), nan=0.0)
    sd = np.asarray(sd, dtype=np.float64)
    bad = (~np.isfinite(sd)) | (sd <= 0)
    sd = np.where(bad, 1.0, sd)
    return mu, sd


def corr_cols(P, T, M):
    """Per-column (=per-gene) across-row Pearson: invariant to per-gene shift/scale. Returns (G,) + number of valid cells."""
    n, G = P.shape
    out = np.full(G, np.nan); ns = np.zeros(G, np.int32)
    for g in range(G):
        m = M[:, g] & np.isfinite(T[:, g]) & np.isfinite(P[:, g])
        k = int(m.sum()); ns[g] = k
        if k < MIN_CELLS_PER_GENE:
            continue
        p = P[m, g].astype(np.float64); t = T[m, g].astype(np.float64)
        p = p - p.mean(); t = t - t.mean()
        d = np.sqrt((p * p).sum() * (t * t).sum())
        if d > 0:
            out[g] = (p * t).sum() / d
    return out, ns


def boot_median_ci(vals, cells, nboot=N_BOOT, seed=SEED):
    """Resample by test cell (= donor unit) to give a CI for the median or the median-of-cell-medians."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=np.float64)
    ok = np.isfinite(v)
    by = {}
    for c, x, o in zip(cells, v, ok):
        if o:
            by.setdefault(c, []).append(x)
    keys = sorted(by)
    if not keys:
        return float("nan"), float("nan"), float("nan"), False, np.array([])
    arrs = [np.array(by[k]) for k in keys]
    point = float(np.median(np.concatenate(arrs)))
    out = np.empty(nboot)
    for b in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        out[b] = np.median(np.concatenate([arrs[i] for i in pick]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return point, float(lo), float(hi), bool(lo * hi > 0), out


def zres_stats(P, T, M, mu, sd):
    """Three calibers:
       raw   : as is
       shift : subtract the per-gene training mean from both (the original design; note a per-gene constant is not a global constant,
               so this genuinely changes the across-gene correlation -- it removes the gene-level background shared across genes)
       z     : additionally divide by the per-gene training standard deviation (an extra per-gene scale normalization)
       Returns (r_raw, r_shift, r_z, corr(r_shift, r_z))"""
    Ps = P - mu; Ts = T - mu
    Pz = Ps / sd; Tz = Ts / sd
    r_raw = corr_rows(P, T, M)
    r_shift = corr_rows(Ps, Ts, M)
    r_z = corr_rows(Pz, Tz, M)
    o = np.isfinite(r_shift) & np.isfinite(r_z)
    agree = float(np.corrcoef(r_shift[o], r_z[o])[0, 1]) if o.sum() > 2 else float("nan")
    return r_raw, r_shift, r_z, agree


def main():
    t0 = time.time()
    X, Ycer, Ychr, mask, split, idx = load()
    log("loaded: X%s Ycer%s mask%s" % (X.shape, Ycer.shape, mask.shape))

    tr = np.array(idx["train_rows_in_cell_order"], dtype=np.int64)
    te = np.array(idx["test_rows_in_cell_order"], dtype=np.int64)
    cells = [l.strip() for l in open(os.path.join(FR, "index", "cell_order.txt")) if l.strip()]
    te_cells = [cells[i] for i in te]
    assert [cells[i] for i in te] == [c for c in split["test_cells"]], "test-cell order does not match split.json"
    assert len(tr) == 636 and len(te) == 136
    log("train %d / test %d cells (matching the original Figure 6 split)" % (len(tr), len(te)))

    # --- features: training-set standardization + PCA(100) ---
    sc = StandardScaler().fit(X[tr])
    Ztr = sc.transform(X[tr]); Zte = sc.transform(X[te])
    pca = PCA(n_components=N_PC, random_state=0).fit(Ztr)
    Ktr = pca.transform(Ztr).astype(np.float64); Kte = pca.transform(Zte).astype(np.float64)
    log("PCA100 training-set explained variance %.3f" % pca.explained_variance_ratio_.sum())

    mtr = mask[tr]; mte = mask[te]
    preds, chosen = {}, {}

    def fit_ridge(Ytrain, tag):
        # alpha chosen by 5-fold CV on the training set only (the test set never participates)
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        best, best_s = ALPHAS[0], -np.inf
        ytr = np.nan_to_num(Ytrain)
        for a in ALPHAS:
            s = []
            for i_tr, i_va in kf.split(Ktr):
                m = Ridge(alpha=a).fit(Ktr[i_tr], ytr[i_tr])
                ph = m.predict(Ktr[i_va])
                tv = ytr[i_va]
                ok = np.isfinite(tv)
                ss_res = ((ph - tv)[ok] ** 2).sum()
                ss_tot = ((tv[ok] - tv[ok].mean()) ** 2).sum()
                s.append(1 - ss_res / max(ss_tot, 1e-12))
            sc_ = float(np.mean(s))
            log("    %s alpha=%g CV-R2=%.4f" % (tag, a, sc_))
            if sc_ > best_s:
                best_s, best = sc_, a
        chosen[tag] = {"alpha": best, "cv_r2": best_s}
        mdl = Ridge(alpha=best).fit(Ktr, ytr)
        return mdl.predict(Kte), best

    log("training Ridge: CERES arm")
    preds["ridge_CERES"], _ = fit_ridge(Ycer[tr], "ridge_CERES")
    log("training Ridge: Chronos arm")
    preds["ridge_Chronos"], _ = fit_ridge(Ychr[tr], "ridge_Chronos")

    log("training Ridge: label-permutation control (shuffled training Y row labels)")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tr))
    preds["ridge_perm"], _ = fit_ridge(Ycer[tr][perm], "ridge_perm")

    # frozen model (existing 5-seed average, test 136 x 17393)
    for arm, Ytag in [("ceres", "CERES"), ("chronos", "Chronos")]:
        acc = None
        for k in range(5):
            f = os.path.join(FR, "predictions_test", "seed%d_pred_%s_arm_test.npy" % (k, arm))
            a = np.load(f).astype(np.float32)
            acc = a if acc is None else acc + a
        preds["frozen_%s" % Ytag] = (acc / 5.0).astype(np.float64)

    # --- per-gene training mean/sd (train cells only) ---
    def prep_mu_sd(Y):
        mu = np.nanmean(np.where(mtr, Y[tr], np.nan), axis=0)
        sd = np.sqrt(np.nanmean(np.where(mtr, (Y[tr] - np.nan_to_num(mu, nan=0.0)) ** 2, np.nan), axis=0))
        return sanitize_mu_sd(mu, sd, mtr, Y[tr])

    MU_SD = {"CERES": prep_mu_sd(Ycer), "Chronos": prep_mu_sd(Ychr)}
    mu_cer, _ = MU_SD["CERES"]
    mu_chr, _ = MU_SD["Chronos"]
    log("mu/sd all finite after sanitize: %s" % all(np.isfinite(MU_SD[k][1]).all() for k in MU_SD))

    # constant baseline: prediction = per-gene training mean (the same vector for every cell)
    preds["mean_baseline_CERES"] = np.tile(mu_cer, (len(te), 1))
    preds["mean_baseline_Chronos"] = np.tile(mu_chr, (len(te), 1))

    res_cell, res_gene, summary = [], [], []
    for tag, P in preds.items():
        for Ytag, Y in [("CERES", Ycer), ("Chronos", Ychr)]:
            T = Y[te]
            mu, sd = MU_SD[Ytag]
            r_raw, r_shift, r_z, agree = zres_stats(P, T, mte, mu, sd)
            rg, ns = corr_cols(P, T, mte)
            # per-cell records
            for i, c in enumerate(te_cells):
                res_cell.append(dict(model=tag, eval=Ytag, cell=c, n_genes=int(mte[i].sum()),
                                     r_raw=r_raw[i], r_shift=r_shift[i], r_zres=r_z[i]))
            # per-gene records (raw only; across-cell is invariant to per-gene affine transforms)
            for g in range(len(rg)):
                if np.isfinite(rg[g]):
                    res_gene.append(dict(model=tag, eval=Ytag, gene_idx=g, n_cells=int(ns[g]),
                                         r_across_cell=rg[g]))
            summary.append(dict(model=tag, eval=Ytag,
                                n_pc=N_PC, alpha=chosen.get(tag, {}).get("alpha"),
                                cell_raw_median=float(np.nanmedian(r_raw)),
                                cell_shift_median=float(np.nanmedian(r_shift)),
                                cell_zres_median=float(np.nanmedian(r_z)),
                                shift_vs_z_corr=agree,
                                gene_across_cell_median=float(np.nanmedian(rg)),
                                gene_across_cell_frac_pos=float(np.nanmean(rg > 0)),
                                n_genes_across=int(np.isfinite(rg).sum())))
            log("%-22s eval=%-8s raw=%+.4f shift=%+.4f z=%+.4f across=%+.4f (shift~z r=%.3f)"
                % (tag, Ytag, np.nanmedian(r_raw), np.nanmedian(r_shift),
                   np.nanmedian(r_z), np.nanmedian(rg), agree))

    import csv
    with open(R("results", "per_cell_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res_cell[0].keys())); w.writeheader(); w.writerows(res_cell)
    with open(R("results", "per_gene_across_cell.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res_gene[0].keys())); w.writeheader(); w.writerows(res_gene)
    with open(R("results", "model_metric_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    # --- 2x2 (training target x evaluation reference), raw and zres calibers ---
    def cell2x2(M, Y, Ytag):
        T = Y[te]
        mu, sd = MU_SD[Ytag]
        r_raw, r_shift, r_z, _ = zres_stats(M, T, mte, mu, sd)
        return (float(np.nanmedian(r_raw)), float(np.nanmedian(r_shift)),
                float(np.nanmedian(r_z)))

    grid = {}
    pairs = {"Ridge": ("ridge_CERES", "ridge_Chronos"), "Frozen": ("frozen_CERES", "frozen_Chronos"),
             "Permuted": ("ridge_perm", "ridge_perm")}
    boot_out = {}
    for name, (a_cer, a_chr) in pairs.items():
        cc_raw, cc_s, cc_z = cell2x2(preds[a_cer], Ycer, "CERES")
        ch_raw, ch_s, ch_z = cell2x2(preds[a_cer], Ychr, "Chronos")
        hc_raw, hc_s, hc_z = cell2x2(preds[a_chr], Ycer, "CERES")
        hh_raw, hh_s, hh_z = cell2x2(preds[a_chr], Ychr, "Chronos")
        grid[name] = dict(CC_raw=cc_raw, CH_raw=ch_raw, HC_raw=hc_raw, HH_raw=hh_raw,
                          CC_shift=cc_s, CH_shift=ch_s, HC_shift=hc_s, HH_shift=hh_s,
                          CC_zres=cc_z, CH_zres=ch_z, HC_zres=hc_z, HH_zres=hh_z,
                          interaction_raw=(cc_raw - ch_raw) - (hc_raw - hh_raw),
                          interaction_shift=(cc_s - ch_s) - (hc_s - hh_s),
                          interaction_zres=(cc_z - ch_z) - (hc_z - hh_z))
        log("%s 2x2 raw: CC=%+.4f CH=%+.4f HC=%+.4f HH=%+.4f I=%+.4f | shift I=%+.4f | zres I=%+.4f"
            % (name, cc_raw, ch_raw, hc_raw, hh_raw, grid[name]["interaction_raw"],
               grid[name]["interaction_shift"], grid[name]["interaction_zres"]))

    # --- bootstrap: per-cell residualized metrics + across-cell metrics ---
    def cells_of(tag, ev, key):
        return ([r[key] for r in res_cell if r["model"] == tag and r["eval"] == ev],
                [r["cell"] for r in res_cell if r["model"] == tag and r["eval"] == ev])

    for key in ["r_shift", "r_zres"]:
        for tag, ev, nm in [("ridge_CERES", "CERES", "ridge_CERES"), ("ridge_perm", "CERES", "perm"),
                            ("mean_baseline_CERES", "CERES", "mean_baseline"),
                            ("frozen_CERES", "CERES", "frozen_CERES"),
                            ("frozen_Chronos", "Chronos", "frozen_Chronos"),
                            ("ridge_Chronos", "Chronos", "ridge_Chronos")]:
            v, cl = cells_of(tag, ev, key)
            boot_out["%s_%s" % (key, nm)] = boot_median_ci(v, cl)
    for tag in ["ridge_CERES", "frozen_CERES", "ridge_perm", "mean_baseline_CERES"]:
        v = [r["r_across_cell"] for r in res_gene if r["model"] == tag and r["eval"] == "CERES"]
        boot_out["across_cell_" + tag] = boot_median_ci(v, ["g%d" % i for i in range(len(v))])
    json.dump({"grid": grid, "chosen_alpha": chosen,
               "bootstrap": {k: {"point": v[0], "ci95": [v[1], v[2]],
                                 "excludes_zero": v[3]} for k, v in boot_out.items()}},
              open(R("results", "positive_control_summary.json"), "w"), indent=2)
    log("done, total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
