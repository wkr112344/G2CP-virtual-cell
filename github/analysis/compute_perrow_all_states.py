"""Recompute per-row (11,275 cell x drug pairs) cross-gene Pearson for ALL FOUR
training x reference states, using the archived frozen prediction matrices and the
original reference caches. Replicates the design construction of
P_frozen_multimetric_cohorts.py exactly (assert-verified), then scores each row.

Output: gigascience_figwork2/GigaScience_GTEx_checked/source_data/
        frozen_multimetric_perrow_scores.csv
        columns: cell, drug,
                 beta_trained__beta2020, beta_trained__dcic2021,
                 dcic_trained__beta2020, dcic_trained__dcic2021
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = r"C:\Users\wkr20\WorkBuddy\2026-09-06-00-53-37"
OUT = r"C:\Users\wkr20\WorkBuddy\Claw\gigascience_v5\results"
D_REB = os.path.join(BASE, "data", "g2cp_cache_rebuilt")
D_B2 = os.path.join(BASE, "data", "g2cp_cache_beta_v2")
D_21 = os.path.join(BASE, "data", "g2cp_cache_2021")
GENE_MAP = os.path.join(BASE, "_scratch", "v31_mvpa", "93_gene_map_12328_to_978.json")
DEST = r"C:\Users\wkr20\WorkBuddy\Claw\gigascience_figwork2\GigaScience_GTEx_checked\source_data"
EPS = 1e-8


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def row_pearson(Pm, T):
    a = Pm - Pm.mean(1, keepdims=True)
    b = T - T.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + EPS)


# 1. design reconstruction (verbatim from P_frozen_multimetric_cohorts.py)
log("building the design set ...")
pmB = np.load(os.path.join(D_21, "pairs_meta.npz"), allow_pickle=True)
b_cell = np.array([str(x) for x in pmB["cell_name"]], dtype=object)
b_drug = np.array([str(x) for x in pmB["drug_id"]], dtype=object)

mR = np.load(os.path.join(D_REB, "meta.npz"), allow_pickle=True)
r_cl = np.array([str(x) for x in mR["cl_names"]], dtype=object)
r_dv = np.array([str(x) for x in mR["drug_vocab"]], dtype=object)
r_cn = np.array([r_cl[int(c)] for c in mR["cell"].astype(np.int64)], dtype=object)
r_dn = np.array([r_dv[int(k)] for k in mR["key"].astype(np.int64)], dtype=object)

Cstar = sorted(set(r_cn.tolist()) & set(b_cell.tolist()))
Dstar = sorted(set(r_dn.tolist()) & set(b_drug.tolist()))
perts = sorted(Dstar)
n_te = int(len(perts) * 0.9)
perm = np.random.RandomState(0).permutation(len(perts))
H = set(perts[i] for i in perm[n_te:])
Cs_set = set(Cstar)
r_pairs = set(zip(r_cn.tolist(), r_dn.tolist()))
b_pairs = set(zip(b_cell.tolist(), b_drug.tolist()))
Pstar = sorted(p for p in (r_pairs & b_pairs) if p[0] in Cs_set and p[1] in H)
nP = len(Pstar)
pidx = {p: i for i, p in enumerate(Pstar)}
log("design C*=%d D*=%d H=%d P*=%d" % (len(Cstar), len(Dstar), len(H), nP))
assert (len(Cstar), len(Dstar), len(H), nP) == (64, 20370, 2037, 11275), "design != archived caliber; aborting"

P_cells = np.array([p[0] for p in Pstar], dtype=object)
P_drugs = np.array([p[1] for p in Pstar], dtype=object)

# 2. 978 landmark axis
gm = json.load(io.open(GENE_MAP, encoding="utf-8"))
internal_col = np.asarray(gm["internal_col"], dtype=np.int64)
assert internal_col.size == 978

# 3. target matrices T_A / T_B -> 978
def pair_level_978(d, cell, drug, tag):
    Y = np.load(os.path.join(d, "y.npy"), mmap_mode="r")
    sel = [i for i in range(len(cell)) if (cell[i], drug[i]) in pidx]
    tgt = np.array([pidx[(cell[i], drug[i])] for i in sel], dtype=np.int64)
    T = np.zeros((nP, 978), dtype=np.float32)
    CH = 5000
    for s0 in range(0, len(sel), CH):
        blk = np.asarray(sel[s0:s0 + CH], dtype=np.int64)
        v = np.asarray(Y[blk], dtype=np.float32)
        T[tgt[s0:s0 + CH]] = v[:, internal_col]
        del v
    del Y
    log("  %s: %d rows matched -> T%s" % (tag, len(sel), T.shape))
    return T


pmB2 = np.load(os.path.join(D_B2, "pairs_meta.npz"), allow_pickle=True)
b2_cell = np.array([str(x) for x in pmB2["cell_name"]], dtype=object)
b2_drug = np.array([str(x) for x in pmB2["drug_id"]], dtype=object)

log("reading T_A = cache_beta_v2 (level5beta2020) ...")
T_A = pair_level_978(D_B2, b2_cell, b2_drug, "A")
log("reading T_B = cache_2021 (dcic2021) ...")
T_B = pair_level_978(D_21, b_cell, b_drug, "B")

# 4. frozen prediction matrices (already archived)
P_beta = np.load(os.path.join(OUT, "frozen_pred_matrix_beta_trained_11275x978.npy"))
P_dcic = np.load(os.path.join(OUT, "frozen_pred_matrix_dcic_trained_11275x978.npy"))
log("P_beta %s  P_dcic %s" % (P_beta.shape, P_dcic.shape))

# 5. per-row scores for the 2x2
log("scoring row by row ...")
out = pd.DataFrame({
    "cell": P_cells,
    "drug": P_drugs,
    "beta_trained__beta2020": row_pearson(P_beta, T_A),
    "beta_trained__dcic2021": row_pearson(P_beta, T_B),
    "dcic_trained__beta2020": row_pearson(P_dcic, T_A),
    "dcic_trained__dcic2021": row_pearson(P_dcic, T_B),
})

# 6. verify at the PER-DRUG level (the archived aggregates are per-drug means,
#    not row means - row mean and drug mean differ because cells/drug is unbalanced)
ref = pd.read_csv(os.path.join(OUT, "for_figures", "C_perdrug_stability.csv"))
g = out.groupby("drug")[["beta_trained__beta2020", "beta_trained__dcic2021",
                         "dcic_trained__beta2020", "dcic_trained__dcic2021"]].mean()
m = g.join(ref.set_index("drug_id")[["pcc_beta", "pcc_dcic"]], how="inner")
log("  drugs matched: %d / %d" % (len(m), len(ref)))
assert len(m) == 2037
for mine, refc in [("beta_trained__beta2020", "pcc_beta"),
                   ("beta_trained__dcic2021", "pcc_dcic")]:
    diff = (m[mine] - m[refc]).abs()
    log("  %-24s vs %-9s max|diff| = %.2e" % (mine, refc, diff.max()))
    assert diff.max() < 1e-3, "per-drug mismatch: " + mine
gm_dcic_b = g["dcic_trained__beta2020"].mean()
gm_dcic_d = g["dcic_trained__dcic2021"].mean()
log("  dcic_trained__beta2020 per-drug mean = %.6f (archived 0.1705)" % gm_dcic_b)
log("  dcic_trained__dcic2021 per-drug mean = %.6f (archived 0.1534)" % gm_dcic_d)
assert abs(gm_dcic_b - 0.1705) < 5e-4 and abs(gm_dcic_d - 0.1534) < 5e-4, "dcic-trained aggregation mismatch"

os.makedirs(DEST, exist_ok=True)
dest = os.path.join(DEST, "frozen_multimetric_perrow_scores.csv")
out.to_csv(dest, index=False)
log("wrote %s (%d rows)" % (dest, len(out)))
