# 03 Align the two products (fast version)
#   pass1: raw gzip stream + bounded split over the gene-name column only, to fix the common gene space (fast)
#   pass2: read GCT in chunks with the pandas C engine (header=2), transpose each chunk into a memmap
#   RSEM:  map the npy (memmap) in row-order chunks onto the common gene columns
# Output memmap (n_samples x n_genes) float32: data_processed/{rnaseqc,rsem_gene}.npy
import os, sys, json, time, zlib
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, log, p, iter_numeric_csv_chunks

cfg = load_config()
hdr = json.load(open(p("metadata", "headers.json")))
samples = hdr["gct_sample_columns"]
S = len(samples)
EXP_GCT_ROWS = cfg["expected"]["gct_rows"]


def strip_v(g):
    return g.split(".")[0] if cfg["analysis"]["strip_gene_version"] else g


# --- RSEM gene matrix (memmap) ---
log("opening the RSEM gene matrix")
meta = json.load(open(p("data_processed", "RSEM_gene_tpm.meta.json")))
R = np.load(p("data_processed", "RSEM_gene_tpm.npy"), mmap_mode="r")
rsem_genes_raw = meta["gene_ids"]
rsem_samples = meta["samples"]
assert set(rsem_samples) == set(samples), "RSEM and GCT sample sets differ"
B_samplecols = np.array([rsem_samples.index(s) for s in samples], dtype=np.int64)
rsem_row = {strip_v(g): i for i, g in enumerate(rsem_genes_raw)}
log("RSEM: %d genes x %d samples (memmap %.2f GB)" % (R.shape[0], R.shape[1], R.shape[0]*R.shape[1]*4/1e9))

# --- pass1 already done by 02: reuse its gene list (= GCT intersected with RSEM, 74,628) ---
common = list(rsem_row.keys())   # the order kept by 02 (RSEM file order); the A/B memmaps just need matching column order
NG = len(common)
log("%d common genes (reusing 02 result, no rescan)" % NG)

A = np.lib.format.open_memmap(p("data_processed", "rnaseqc.npy"), mode="w+",
                              dtype=np.float32, shape=(S, NG))
Bm = np.lib.format.open_memmap(p("data_processed", "rsem_gene.npy"), mode="w+",
                               dtype=np.float32, shape=(S, NG))

# --- pass2: stream-read GCT with zlib+loadtxt, write A ---
log("pass2: streaming GCT -> rnaseqc.npy")
from common import iter_tsv_gz_blocks
gidx = {g: j for j, g in enumerate(common)}
t0 = time.time()
done = 0
for gid, vals in iter_tsv_gz_blocks(cfg["paths"]["gene_tpm_gz"],
                                    skip_lines=3, n_id_cols=2, id_index=0):
    gid = [strip_v(g) for g in gid]
    mask = np.fromiter((g in gidx for g in gid), bool, len(gid))
    if mask.any():
        cols = np.fromiter((gidx[g] for g in gid), np.int64, len(gid))[mask]
        A[:, cols] = vals[mask].T
    done += len(gid)
    log("  GCT %d/%d rows, %.0fs" % (done, EXP_GCT_ROWS, time.time()-t0))
A.flush()

# --- RSEM chunk mapping (collect columns in RSEM row order, write transposed chunks) ---
log("RSEM -> rsem_gene.npy (chunked transpose)")
order = [(i, gidx[g]) for g, i in ((g, rsem_row[g]) for g in common)]  # (rsem_row, out_col)
t0 = time.time()
K = 2000
for lo in range(0, len(order), K):
    blk = order[lo:lo+K]
    rrows = np.array([x[0] for x in blk], dtype=np.int64)
    ccols = np.array([x[1] for x in blk], dtype=np.int64)
    Bm[:, ccols] = np.asarray(R[rrows], dtype=np.float32)[:, B_samplecols].T
    log("  %d/%d, %.0fs" % (min(lo+K, len(order)), len(order), time.time()-t0))
Bm.flush()

json.dump({"genes": common, "samples": samples, "n_samples": S, "n_genes": NG},
          open(p("metadata", "common_grid.json"), "w"), indent=1)
log("PASS -> data_processed/{rnaseqc,rsem_gene}.npy (%d × %d) + metadata/common_grid.json" % (S, NG))
