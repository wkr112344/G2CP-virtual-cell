# 05 tissue-mean predictor (one per product, using train samples only)
# Definition (pre-registered): in log2(TPM+1) space, average each gene over train samples grouped by tissue_label;
# a test sample's prediction = the column for its tissue. Tissues missing from train -> fall back to the global train mean.
# Output data_processed/tissue_mean_{rnaseqc,rsem}.npz (tissues x genes, float32)
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, log, p

cfg = load_config()
lab = cfg["analysis"]["tissue_label"]
grid = json.load(open(p("metadata", "common_grid.json")))
samples, NG = grid["samples"], grid["n_genes"]
splits = json.load(open(p("metadata", "splits.json")))
tmap = json.load(open(p("metadata", "sample_tissue_map.json")))

tr_s = splits["train_samples"]
tr_idx = np.array([samples.index(s) for s in tr_s], dtype=np.int64)
tissues = sorted({tmap[s][lab] for s in tr_s})
ti = {t: k for k, t in enumerate(tissues)}
lab_of_sample = np.array([ti[tmap[s][lab]] for s in tr_s], dtype=np.int64)
NT = len(tissues)

for prod, fn in [("rnaseqc", "rnaseqc.npy"), ("rsem", "rsem_gene.npy")]:
    M = np.load(p("data_processed", fn), mmap_mode="r")
    t0 = time.time()
    means = np.zeros((NT, NG), dtype=np.float64)
    counts = np.zeros(NT, dtype=np.int64)
    CH = 512  # sample chunk size
    for lo in range(0, len(tr_s), CH):
        blk = np.log2(M[tr_idx[lo:lo+CH]].astype(np.float64) + 1.0)   # (c, NG)
        for k in range(NT):
            m = lab_of_sample[lo:lo+CH] == k
            if m.any():
                means[k] += blk[m].sum(0)
                counts[k] += int(m.sum())
        log("  %s: %d/%d train samples, %.0fs" % (prod, min(lo+CH, len(tr_s)), len(tr_s), time.time()-t0))
    means /= counts[:, None]
    # global-mean fallback (used when the tissue is missing from train)
    glob = means.sum(0) / counts.sum()
    np.savez_compressed(p("data_processed", "tissue_mean_%s.npz" % prod),
                        tissues=np.array(tissues), means=means.astype(np.float32),
                        fallback=glob.astype(np.float32), counts=counts,
                        tissue_label=lab)
    log("%s: %d tissues x %d genes, %.0fs -> tissue_mean_%s.npz" % (prod, NT, NG, time.time()-t0, prod))
