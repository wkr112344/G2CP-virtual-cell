#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRnet_commonX_build.py -- build the training dataset for the PRnet "full set A" controlled 2x2 (memory-streaming version).

Design (core constraint: the two training arms share exactly the same input X; only the training target differs):
  - X (shared) = baseline/control expression (969 genes) + drug embedding (1024) + dose_z + time_z
  - Y_beta    = beta2020 treated expression (969 genes)      <- beta training arm target
  - Y_dcic    = dcic2021 SigCom CD signature (969 genes)     <- dcic training arm target
  - obs / drug / baseline / split / gene_axis are identical across arms; only the target is replaced.

Data sources (all already local; no download needed):
  - Desktop/Lincs_L1000.h5ad         (beta2020 PRnet training, 883269x978)
  - Desktop/cp_coeff_mat.gctx         (dcic2021 Level5 signatures, 718157x12327)
  - data/common_969_genes.tsv         (969 gene axis: dcic <-> beta symbol alignment)

Output:
  - data/prnet_commonX/PRnet_commonX_dualY.h5            (main training format, written in streaming chunks)
  - data/prnet_commonX/PRnet_commonX_betaY.h5ad          (X=Y_beta)  [skipped with --no-h5ad]
  - data/prnet_commonX/PRnet_commonX_dcicY.h5ad          (X=Y_dcic)  [skipped with --no-h5ad]
  - data/prnet_commonX/build_manifest.json                (locked preprocessing/statistics)

normalization locked: baseline / Y_beta / Y_dcic each z-scored per gene (full-population statistics),
  with the statistics written into the manifest; evaluation uses the same inverse transform / same-caliber correlations, keeping the arms comparable.

Memory strategy: all large arrays are streamed to/from h5 in row chunks, never loaded fully into RAM (handles the full ~3M samples).

drug embedding: uses rdkit Morgan fingerprints (radius=2, 1024-bit) as a genuine molecular representation,
  replacing the earlier SMILES n-gram hash fingerprint (which collides 99.9% of the time at full scale, collapsing drugs into a few
  buckets, degrading the model to per-gene mean prediction and giving a pseudo-null 2x2). Morgan fingerprints are cached per SMILES to avoid recomputation.
  (the training script concatenates dose_z/time_z onto the 1024 dims to form 1026-dim conditioning; the architecture is unchanged.)
  This script must run with a python that has rdkit (e.g. dpb311: C:/Users/wkr20/miniconda3/envs/dpb311/python.exe）。

Usage:
  python3 PRnet_commonX_build.py --limit 5000        # sanity subset
  python3 PRnet_commonX_build.py --no-h5ad           # full (aligned set), no h5ad
  python3 PRnet_commonX_build.py                     # full (aligned set), with h5ad
"""
import argparse, json, os, sys
import numpy as np
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

H5AD = r"C:/Users/wkr20/Desktop/Lincs_L1000.h5ad"
GCTX = r"C:/Users/wkr20/Desktop/cp_coeff_mat.gctx"
GENE_TSV = r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/common_969_genes.tsv"
S25 = r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/common_eval_pairs.csv"
OUT_DIR = r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/prnet_commonX"
N_DRUG = 1024

# ---------------------------------------------------------------- 969 gene axis
def load_gene_axis():
    beta_sym, dcic_sym = [], []
    with open(GENE_TSV) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            beta_sym.append(p[2])      # beta_original (symbol on the h5ad side)
            dcic_sym.append(p[1])      # gene_dcic    (symbol on the cp_coeff side)
    return beta_sym, dcic_sym

# ---------------------------------------------------------------- drug fingerprints (rdkit Morgan)
_MORGAN_CACHE = {}

def smiles_to_emb(smiles, dim=N_DRUG, radius=2):
    """rdkit Morgan fingerprints (1024-bit) as drug representation, replacing the n-gram hash (which collides 99.9% at full scale).
    Output is L2-normalized to unit vectors, matching the training script's conditioning scale. Cached per SMILES."""
    if smiles is None or (isinstance(smiles, str) and smiles.strip().lower() in ("", "nan", "none", "dmso")):
        return np.zeros(dim, dtype=np.float32)
    key = str(smiles)
    cached = _MORGAN_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        mol = Chem.MolFromSmiles(key)
        if mol is None:
            v = np.zeros(dim, dtype=np.float32)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=dim, useChirality=True)
            arr = np.zeros(dim, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            a = arr.astype(np.float32)
            nrm = float(np.sqrt((a * a).sum())) or 1.0
            v = (a / nrm).astype(np.float32)
    except Exception:
        v = np.zeros(dim, dtype=np.float32)
    _MORGAN_CACHE[key] = v
    return v

def num(x):
    x = x.decode() if isinstance(x, bytes) else str(x)
    try:
        return round(float(x.split()[0]), 3)
    except Exception:
        return None

# ---------------------------------------------------------------- main flow
def build(limit=None, no_h5ad=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    beta_sym, dcic_sym = load_gene_axis()
    N_GENE = len(beta_sym)
    assert N_GENE == 969, f"gene axis != 969: {N_GENE}"
    print(f"[gene] 969 axis loaded; beta[0]={beta_sym[0]} dcic[0]={dcic_sym[0]}")

    # ---- beta h5ad obs (per-row arrays, no big dict) ----
    with h5py.File(H5AD, "r") as f:
        def cat(name):
            g = f["obs/__categories/" + name]
            return [x.decode() if isinstance(x, bytes) else x for x in g[:]]
        cell_c = cat("cell_id"); pert_c = cat("pert_id"); smile_c = cat("canonical_smiles")
        cc = f["obs/cell_id"][:].astype(int)
        pc = f["obs/pert_id"][:].astype(int)
        dose = f["obs/pert_dose"][:]
        time = f["obs/pert_time"][:]
        ptype = f["obs/pert_type"][:].astype(int)          # 0=ctl_vehicle 1=trt_cp
        pci = f["obs/paired_control_index"][:].astype(np.int64)
        rsplit = f["obs/random_split_0"][:].astype(np.int64)
        scode = f["obs/canonical_smiles"][:].astype(int)
        var_idx = [x.decode() if isinstance(x, bytes) else x for x in f["var/_index"][:]]
    beta2col = {s: i for i, s in enumerate(var_idx)}
    beta_cols = np.array([beta2col[s] for s in beta_sym], dtype=int)
    print(f"[beta] obs loaded; trt_cp rows={int((ptype==1).sum())}; var genes={len(var_idx)}")

    # ---- collect treatment conditions (cell,pert,dose,time) -> beta rows ----
    trt = np.where(ptype == 1)[0]
    cond = {}
    for i in trt:
        k = (cell_c[cc[i]], pert_c[pc[i]], round(float(dose[i]), 3), round(float(time[i]), 3))
        cond.setdefault(k, []).append(i)
    n_cond_beta = len(cond)
    print(f"[beta] distinct (cell,pert,dose,time) treatment conditions: {n_cond_beta}")

    # ---- cp_coeff: key -> [cp col idx] ----
    with h5py.File(GCTX, "r") as f:
        col = f["0/META/COL"]
        gc = [x.decode() if isinstance(x, bytes) else x for x in col["cell_line"][:]]
        gp = [x.decode() if isinstance(x, bytes) else x for x in col["pert_id"][:]]
        gd = col["pert_dose"][:]; gt = col["pert_time"][:]
        dcic_var = [x.decode() if isinstance(x, bytes) else x for x in f["0/META/ROW/id"][:]]
    dcic2row = {s: i for i, s in enumerate(dcic_var)}
    dcic_rows = np.array([dcic2row[s] for s in dcic_sym], dtype=int)
    dcic_key2col = {}
    for j in range(len(gc)):
        k = (gc[j], gp[j], num(gd[j]), num(gt[j]))
        dcic_key2col.setdefault(k, []).append(j)
    print(f"[dcic] distinct keys={len(dcic_key2col)}; matrix genes={len(dcic_var)}")

    # ---- aligned subset (conditions present in both Y) ----
    aligned_keys = [k for k in cond if k in dcic_key2col]
    if limit is not None and limit < len(aligned_keys):
        aligned_keys = aligned_keys[:limit]
    key_index = {k: idx for idx, k in enumerate(aligned_keys)}
    n_cond = len(aligned_keys)
    print(f"[align] conditions with BOTH beta & dcic target: {n_cond}")

    # ---- sample-level lists (treatment row / its paired control row / owning condition idx) ----
    tx, ctl, kidx = [], [], []
    for k in aligned_keys:
        ki = key_index[k]
        for i in cond[k]:
            tx.append(i); ctl.append(pci[i]); kidx.append(ki)
    tx = np.array(tx, np.int64); ctl = np.array(ctl, np.int64); kidx = np.array(kidx, np.int32)
    N = len(tx)
    print(f"[assemble] final samples={N}")

    # ---- Y_dcic: scan the gctx in blocks, averaging CD signatures per condition (float32 accumulation) ----
    Yd_key = np.zeros((n_cond, N_GENE), np.float32)
    yd_cnt = np.zeros(n_cond, np.int32)
    col2kid = {}
    for k, cols in dcic_key2col.items():
        if k in key_index:
            for c in cols:
                col2kid[c] = key_index[k]
    need_set = set(col2kid.keys())
    with h5py.File(GCTX, "r") as f:
        M = f["0/DATA/0/matrix"]; nrow = M.shape[0]
        if len(need_set) < 0.5 * nrow:
            # subset: read only the needed rows (avoid scanning all 35 GB)
            rows = np.array(sorted(need_set), dtype=int)
            bd = M[rows, :]
            for idx, r in enumerate(rows):
                v = bd[idx, dcic_rows].astype(np.float32)
                kk = col2kid[r]; Yd_key[kk] += v; yd_cnt[kk] += 1
            print(f"[ydcic] direct-read {len(rows)} needed rows (subset path)")
        else:
            # full: one sequential block scan (35 GB of sequential reads)
            B = 5000
            for s in range(0, nrow, B):
                e = min(s + B, nrow); bd = M[s:e, :]
                for r in range(s, e):
                    if r in need_set:
                        v = bd[r - s, dcic_rows].astype(np.float32)
                        kk = col2kid[r]; Yd_key[kk] += v; yd_cnt[kk] += 1
    for ki in range(n_cond):
        if yd_cnt[ki] > 0:
            Yd_key[ki] /= yd_cnt[ki]
    print(f"[ydcic] accumulated CD signatures for {int((yd_cnt>0).sum())}/{n_cond} conditions")

    # ---- S25 set (cell, drug=pert_id) ----
    s25set = set()
    if os.path.exists(S25):
        import csv
        with open(S25) as fh:
            r = csv.reader(fh, delimiter="\t")
            next(r)
            for row in r:
                if len(row) >= 3:
                    s25set.add((row[1], row[2]))

    # ---- stream out dualY.h5 (raw) in chunks, accumulating per-gene statistics ----
    out_h5 = os.path.join(OUT_DIR, "PRnet_commonX_dualY.h5")
    BLK = 200000
    sumX = np.zeros(N_GENE, np.float64); sumX2 = np.zeros(N_GENE, np.float64)
    sumYb = np.zeros(N_GENE, np.float64); sumYb2 = np.zeros(N_GENE, np.float64)
    sumYd = np.zeros(N_GENE, np.float64); sumYd2 = np.zeros(N_GENE, np.float64)

    with h5py.File(H5AD, "r") as fh5, h5py.File(out_h5, "w") as fo:
        Xds = fh5["X"]
        fo.create_dataset("X_baseline", (N, N_GENE), np.float32)
        fo.create_dataset("Y_beta", (N, N_GENE), np.float32)
        fo.create_dataset("Y_dcic", (N, N_GENE), np.float32)
        fo.create_dataset("drug_emb", (N, N_DRUG), np.float32)
        fo.create_dataset("split", (N,), np.int8)
        fo.create_dataset("s25_eval", (N,), np.int8)
        fo.create_dataset("dose", (N,), np.float32)
        fo.create_dataset("time", (N,), np.float32)
        fo.create_dataset("cell", (N,), dtype="S24")
        fo.create_dataset("pert", (N,), dtype="S24")
        fo.create_dataset("gene_axis", data=np.array(beta_sym, dtype="S20"))

        for b0 in range(0, N, BLK):
            b1 = min(b0 + BLK, N)
            txb = tx[b0:b1]; ctlb = ctl[b0:b1]; kkb = kidx[b0:b1]
            uniq, inv_arr = np.unique(np.concatenate([txb, ctlb]), return_inverse=True)
            inv = {r: p for p, r in enumerate(uniq.tolist())}
            Xr = Xds[uniq, :][:, beta_cols].astype(np.float32)   # (len(uniq),969)
            Xb_block = Xr[[inv[r] for r in ctlb], :]             # control/baseline
            Yb_block = Xr[[inv[r] for r in txb], :]             # treatment
            smiles_block = [smile_c[scode[r]] for r in txb]
            drug_block = np.array([smiles_to_emb(s) for s in smiles_block], np.float32)
            Yd_block = Yd_key[kkb].astype(np.float32)
            cell_block = np.array([cell_c[cc[r]].encode()[:24] for r in txb], dtype="S24")
            pert_block = np.array([pert_c[pc[r]].encode()[:24] for r in txb], dtype="S24")
            dose_block = np.array([float(dose[r]) for r in txb], np.float32)
            time_block = np.array([float(time[r]) for r in txb], np.float32)
            s25_block = np.array([1 if (cell_c[cc[r]], pert_c[pc[r]]) in s25set else 0 for r in txb], np.int8)
            rsp_block = np.array([int(rsplit[r]) % 10 for r in txb], np.int8)
            split_block = np.where(rsp_block == 9, 2, np.where(rsp_block == 8, 1, 0)).astype(np.int8)
            split_block = np.where(s25_block == 1, 3, split_block).astype(np.int8)  # S25 -> eval holdout

            fo["X_baseline"][b0:b1] = Xb_block
            fo["Y_beta"][b0:b1] = Yb_block
            fo["Y_dcic"][b0:b1] = Yd_block
            fo["drug_emb"][b0:b1] = drug_block
            fo["split"][b0:b1] = split_block
            fo["s25_eval"][b0:b1] = s25_block
            fo["dose"][b0:b1] = dose_block
            fo["time"][b0:b1] = time_block
            fo["cell"][b0:b1] = cell_block
            fo["pert"][b0:b1] = pert_block

            sumX += Xb_block.sum(0); sumX2 += (Xb_block ** 2).sum(0)
            sumYb += Yb_block.sum(0); sumYb2 += (Yb_block ** 2).sum(0)
            sumYd += Yd_block.sum(0); sumYd2 += (Yd_block ** 2).sum(0)
            print(f"[write] block {b0}-{b1} done")

    # ---- locked z-score (per gene, chunked in place) ----
    muX, sdX = sumX / N, np.sqrt(np.maximum(sumX2 / N - (sumX / N) ** 2, 1e-12))
    muYb, sdYb = sumYb / N, np.sqrt(np.maximum(sumYb2 / N - (sumYb / N) ** 2, 1e-12))
    muYd, sdYd = sumYd / N, np.sqrt(np.maximum(sumYd2 / N - (sumYd / N) ** 2, 1e-12))
    with h5py.File(out_h5, "r+") as fo:
        for b0 in range(0, N, BLK):
            b1 = min(b0 + BLK, N)
            for name, mu, sd in (("X_baseline", muX, sdX), ("Y_beta", muYb, sdYb), ("Y_dcic", muYd, sdYd)):
                a = fo[name][b0:b1].astype(np.float64)
                fo[name][b0:b1] = ((a - mu) / sd).astype(np.float32)
    print("[norm] per-gene z-score applied (locked)")

    # ---- value-range check ----
    with h5py.File(out_h5, "r") as fo:
        Yb_r = (fo["Y_beta"][:].min(), fo["Y_beta"][:].max())
        Yd_r = (fo["Y_dcic"][:].min(), fo["Y_dcic"][:].max())
        sp = fo["split"][:]
    print(f"[check] Y_beta z range [{Yb_r[0]:.2f},{Yb_r[1]:.2f}]  Y_dcic z range [{Yd_r[0]:.2f},{Yd_r[1]:.2f}]")
    print(f"[check] train/val/test/eval(s25) = {(sp==0).sum()}/{(sp==1).sum()}/{(sp==2).sum()}/{(sp==3).sum()}")

    # ---- optional h5ad ----
    if not no_h5ad:
        try:
            import anndata
            with h5py.File(out_h5, "r") as fo:
                Xb_z = fo["X_baseline"][:]; Yb_z = fo["Y_beta"][:]; Yd_z = fo["Y_dcic"][:]
                drug = fo["drug_emb"][:]; split = fo["split"][:]; s25 = fo["s25_eval"][:]
                cell = [x.decode() for x in fo["cell"][:]]; pert = [x.decode() for x in fo["pert"][:]]
                dose = fo["dose"][:]; time = fo["time"][:]
            for tag, Y in (("betaY", Yb_z), ("dcicY", Yd_z)):
                ad = anndata.AnnData(Y)
                ad.obsm["drug_emb"] = drug
                ad.obs["cell"] = cell; ad.obs["pert"] = pert
                ad.obs["dose"] = dose; ad.obs["time"] = time
                ad.obs["split"] = split; ad.obs["s25_eval"] = s25
                ad.var["gene"] = beta_sym
                p = os.path.join(OUT_DIR, f"PRnet_commonX_{tag}.h5ad")
                ad.write_h5ad(p)
                print(f"[out] wrote {p}")
        except Exception as e:
            print(f"[warn] anndata h5ad write skipped: {e}")

    # ---- manifest ----
    manifest = dict(
        design="PRnet_commonX: shared X (baseline+drug+dose+time), swapped Y (beta vs dcic)",
        n_samples=int(N), n_conditions=int(n_cond),
        n_train=int((sp == 0).sum()), n_val=int((sp == 1).sum()),
        n_test=int((sp == 2).sum()), n_eval=int((sp == 3).sum()),
        gene_axis=beta_sym, n_genes=N_GENE,
        drug_emb=f"rdkit Morgan fingerprint ({N_DRUG}-bit, radius=2, L2-normalized) + dose_z + time_z (shared across arms; fixes n-gram 99.9% collision)",
        norm_baseline=dict(mu=muX.tolist(), sd=sdX.tolist()),
        norm_Y_beta=dict(mu=muYb.tolist(), sd=sdYb.tolist()),
        norm_Y_dcic=dict(mu=muYd.tolist(), sd=sdYd.tolist()),
        source_h5ad=H5AD, source_gctx=GCTX, source_gene_tsv=GENE_TSV, source_s25=S25,
        limit=limit,
    )
    with open(os.path.join(OUT_DIR, "build_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[done] manifest -> build_manifest.json | train/val/test/eval(s25) = "
          f"{manifest['n_train']}/{manifest['n_val']}/{manifest['n_test']}/{manifest['n_eval']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="limit the number of aligned conditions (sanity subset)")
    ap.add_argument("--no-h5ad", action="store_true", help="skip the h5ad write (saves space/time at full scale)")
    args = ap.parse_args()
    build(limit=args.limit, no_h5ad=args.no_h5ad)
