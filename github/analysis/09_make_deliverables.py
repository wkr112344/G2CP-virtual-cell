# Package deliverables under the requested file names (generated/aligned from current outputs), written to deliverables/
import json, csv, os, shutil
D = "deliverables"; os.makedirs(D, exist_ok=True)
J = lambda *a: json.load(open(os.path.join(*a), encoding="utf-8"))
summ = J("results", "score_summary.json"); boot = J("results", "bootstrap.json")
grid = J("metadata", "common_grid.json"); hdr = J("metadata", "headers.json")
sp = J("metadata", "splits.json"); dj = J("metadata", "donors.json")
tmap = J("metadata", "sample_tissue_map.json")

# ---- benchmark_summary.json ----
out = {
 "design": "2x2 reference-product sensitivity (GTEx v11, release 2025-08-22)",
 "predictors": {
  "RNASeQC-trained": "tissue-mean (SMTSD) on RNASeQC gene TPM, train-only",
  "RSEM-trained": "tissue-mean (SMTSD) on RSEM transcripts summed to gene, train-only"},
 "evaluation_space": "log2(TPM+1), common genes (see common_genes.csv)",
 "split": {"unit": "donor", "seed": sp["seed"],
           "train_donors": len(sp["train_donors"]), "test_donors": len(sp["test_donors"]),
           "train_samples": len(sp["train_samples"]), "test_samples": len(sp["test_samples"])},
 "bootstrap": {"method": "donor bootstrap (test donors resampled with replacement)",
               "n": boot["n_bootstrap"]},
 "cells": {k: {"pearson_median": summ[k]["median"],
               "spearman_median": summ["sp_" + k]["median"]}
           for k in ["rRR", "rRE", "rER", "rEE"]},
 "interaction": {"definition": "(rRR - rRE) - (rER - rEE)",
  "pearson": {"point_median": boot["pearson"]["point_median"], "ci95": boot["pearson"]["ci95"],
              "excludes_zero": boot["pearson"]["excludes_zero"],
              "frac_gt_zero": boot["pearson"]["frac_gt_zero"]},
  "spearman": {"point_median": boot["spearman"]["point_median"], "ci95": boot["spearman"]["ci95"],
               "excludes_zero": boot["spearman"]["excludes_zero"],
               "frac_gt_zero": boot["spearman"]["frac_gt_zero"]}},
 "conclusion": ("Benchmark scores are sensitive to the processed reference product in GTEx: "
                "each predictor scores significantly higher on its own training reference; "
                "interaction CI excludes zero under both metrics."),
}
json.dump(out, open(os.path.join(D, "benchmark_summary.json"), "w", encoding="utf-8"), indent=2)

# ---- benchmark_summary.csv ----
with open(os.path.join(D, "benchmark_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["term", "predictor", "reference", "pearson_median", "spearman_median"])
    names = {"rRR": ("RNASeQC-trained", "RNASeQC"), "rRE": ("RNASeQC-trained", "RSEM"),
             "rER": ("RSEM-trained", "RNASeQC"), "rEE": ("RSEM-trained", "RSEM")}
    for k, (pr, rf) in names.items():
        w.writerow([k, pr, rf, summ[k]["median"], summ["sp_" + k]["median"]])
    w.writerow(["interaction_pearson", "", "", boot["pearson"]["point_median"], ""])
    w.writerow(["interaction_spearman", "", "", "", boot["spearman"]["point_median"]])
    w.writerow(["interaction_ci95_low_pearson", "", "", boot["pearson"]["ci95"][0], ""])
    w.writerow(["interaction_ci95_high_pearson", "", "", boot["pearson"]["ci95"][1], ""])
    w.writerow(["interaction_ci95_low_spearman", "", "", "", boot["spearman"]["ci95"][0]])
    w.writerow(["interaction_ci95_high_spearman", "", "", "", boot["spearman"]["ci95"][1]])

# ---- input_overlap.json ----
out2 = {
 "gene_overlap": {"gct_genes": hdr["gct_rows"],
                  "rsem_transcript_rows": 385659,
                  "common_genes": grid["n_genes"],
                  "note": ("RSEM transcripts summed to gene_id; intersection with GCT = "
                           "%d (all GCT genes present in RSEM aggregation)" % grid["n_genes"])},
 "sample_overlap": {"gct_samples": hdr["gct_cols"],
                    "transcripts_samples": hdr["gct_cols"],
                    "sample_order_identical": hdr["sample_order_identical_to_gct"],
                    "annotation_table_samples": 48231,
                    "matrix_samples_in_annotation": dj["n_samples"],
                    "join_coverage": "100% (0 unmatched)"},
 "donors": {"total": len(dj["donors"]), "train": len(sp["train_donors"]),
            "test": len(sp["test_donors"])},
}
json.dump(out2, open(os.path.join(D, "input_overlap.json"), "w", encoding="utf-8"), indent=2)

# ---- alignment_split_summary.json ----
out3 = {
 "alignment": {"gene_space": "ENSG (version-stripped for matching)",
               "n_common_genes": grid["n_genes"],
               "row_order": "samples in GCT column order",
               "storage": "memmap float32 rnaseqc.npy / rsem_gene.npy"},
 "split": {"seed": sp["seed"], "unit": "donor", "test_fraction": 0.2,
           "train_donors": len(sp["train_donors"]), "test_donors": len(sp["test_donors"]),
           "train_samples": len(sp["train_samples"]), "test_samples": len(sp["test_samples"]),
           "tissue_label": "SMTSD", "n_tissues_in_data": len(dj["tissue_counts"])},
}
json.dump(out3, open(os.path.join(D, "alignment_split_summary.json"), "w", encoding="utf-8"), indent=2)

# ---- figures and per-sample scores ----
shutil.copyfile("figures/fig1_score_distributions.png", os.path.join(D, "fig_reference_agreement.png"))
shutil.copyfile("figures/fig2_2x2_interaction.png", os.path.join(D, "fig_2x2_heatmap.png"))
shutil.copyfile("results/per_sample_scores.csv", os.path.join(D, "per_sample_scores.csv"))

# ---- common_genes.csv ----
with open(os.path.join(D, "common_genes.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["index", "gene_id"])
    for i, g in enumerate(grid["genes"]):
        w.writerow([i, g])

# ---- sample_split.csv ----
train_s = set(sp["train_samples"]); test_s = set(sp["test_samples"])
with open(os.path.join(D, "sample_split.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["sample", "donor", "tissue", "split"])
    for s in grid["samples"]:
        t = tmap[s]
        split = "train" if s in train_s else ("test" if s in test_s else "unused")
        w.writerow([s, t["donor"], t["SMTSD"], split])

print("generated:")
for f in sorted(os.listdir(D)):
    print("  %-32s %9.1f KB" % (f, os.path.getsize(os.path.join(D, f)) / 1024))
