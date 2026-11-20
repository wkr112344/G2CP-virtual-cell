#!/usr/bin/env python3
"""Recreate manuscript Figure 7 from GTEx derived source data.

Inputs (default paths relative to package root):
  source_data/gtex_reference_sensitivity/per_sample_scores.csv
  source_data/gtex_reference_sensitivity/benchmark_summary.csv

Outputs:
  figures/Figure7_gtex_reference_matching.png
  figures/Figure7_gtex_reference_matching.pdf

The figure contains:
  A) 2x2 median Pearson benchmark matrix.
  B) Held-out sample-level matched-reference advantages
     (RR-RE and EE-ER) with the donor-bootstrap interaction CI.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_summary_value(summary: pd.DataFrame, term: str, col: str) -> float:
    row = summary.loc[summary["term"] == term]
    if row.empty:
        raise KeyError(f"Missing term {term!r} in benchmark_summary.csv")
    return float(row.iloc[0][col])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="source_data/gtex_reference_sensitivity/per_sample_scores.csv")
    p.add_argument("--summary", default="source_data/gtex_reference_sensitivity/benchmark_summary.csv")
    p.add_argument("--out-prefix", default="figures/Figure7_gtex_reference_matching")
    args = p.parse_args()

    scores = pd.read_csv(args.scores)
    summary = pd.read_csv(args.summary)

    required = {"RR", "RE", "ER", "EE"}
    if not required.issubset(scores.columns):
        raise ValueError(f"per_sample_scores.csv must contain {sorted(required)}")

    rr = get_summary_value(summary, "rRR", "pearson_median")
    re = get_summary_value(summary, "rRE", "pearson_median")
    er = get_summary_value(summary, "rER", "pearson_median")
    ee = get_summary_value(summary, "rEE", "pearson_median")
    interaction = get_summary_value(summary, "interaction_pearson", "pearson_median")
    ci_lo = get_summary_value(summary, "interaction_ci95_low_pearson", "pearson_median")
    ci_hi = get_summary_value(summary, "interaction_ci95_high_pearson", "pearson_median")

    mat = np.array([[rr, re], [er, ee]], dtype=float)
    adv_rna = (scores["RR"] - scores["RE"]).dropna().to_numpy()
    adv_rsem = (scores["EE"] - scores["ER"]).dropna().to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.02, 1.15]})

    im = ax1.imshow(mat, aspect="auto")
    ax1.set_xticks([0, 1], ["RNASeQC eval", "RSEM eval"])
    ax1.set_yticks([0, 1], ["RNASeQC-trained", "RSEM-trained"])
    ax1.set_title("A  Median Pearson across genes", loc="left", fontweight="bold")
    threshold = float(np.nanmean(mat))
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2.boxplot([adv_rna, adv_rsem], labels=["RNASeQC matched", "RSEM matched"], widths=0.15)
    ax2.axhline(0, linewidth=1)
    ax2.set_ylabel("Matched-reference advantage (Pearson r)")
    ax2.set_title("B  Held-out sample-level matching advantage", loc="left", fontweight="bold")
    ax2.text(
        0.02, 0.03,
        f"Interaction = {interaction:.5f}\n95% donor-bootstrap CI [{ci_lo:.5f}, {ci_hi:.5f}]",
        transform=ax2.transAxes,
        ha="left", va="bottom"
    )

    fig.tight_layout()
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(prefix) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(prefix) + ".pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
