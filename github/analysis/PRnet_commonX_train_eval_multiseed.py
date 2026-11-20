#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRnet_commonX_train_eval_multiseed.py -- multi-seed LINCS PRnet study of the "full set A" controlled 2x2.

Same design constraints as the single-seed PRnet_commonX_train_eval.py (shared X; only the training target differs:
beta expression Y_beta vs dcic2021 CD signature Y_dcic). This script wraps the whole 2x2 pipeline in N seeds:

  for each seed k:
    1) set the random seed -> build both arms with the same architecture (PGM)
    2) train the beta arm (target=Y_beta) and the dcic arm (target=Y_dcic)
    3) predict on the S25 hold-out for each, computing pearson + spearman per sample (969-dim vector)
    4) four cells BB/BD/DB/DD + interaction = (BB-BD)-(DB-DD)   [one version each for sp & pe]

  cross-seed aggregation:
    - multiseed_2x2_summary.csv  (per-seed four cells + interaction)
    - multiseed_bootstrap.npz    (5000 hierarchical-bootstrap interaction replicates)
    - per_seed_interaction (n_seed,2)

Interaction significance uses a hierarchical bootstrap: each replicate resamples both seeds and samples,
giving a 95% CI for the interaction (matching the DepMap G/H caliber).

Note: must run with a python that has torch (dpb311: C:/Users/wkr20/miniconda3/envs/dpb311/python.exe）。
The data must first be produced by PRnet_commonX_build.py (Morgan version) as PRnet_commonX_dualY.h5.

Usage:
  python PRnet_commonX_train_eval_multiseed.py --seeds 5 --train-cap 200000
  python PRnet_commonX_train_eval_multiseed.py --seeds 1 --train-cap 40000 --limit-eval 5000   # quick sanity check
"""
import argparse, json, os, sys, math, random
import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

# ============================================================ PRnet model (PGM)
class PEncoder(nn.Module):
    def __init__(self, layer_sizes, z_dim, dr):
        super().__init__()
        self.FC = None
        if len(layer_sizes) > 1:
            self.FC = nn.Sequential()
            for i, (ins, outs) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    self.FC.add_module(f"L{i}", nn.Linear(ins, outs, bias=False))
                else:
                    self.FC.add_module(f"L{i}", nn.Linear(ins, outs))
                    self.FC.add_module(f"N{i}", nn.BatchNorm1d(outs))
                    self.FC.add_module(f"A{i}", nn.LeakyReLU(0.3))
                    self.FC.add_module(f"D{i}", nn.Dropout(p=dr))
        self.mean_encoder = nn.Linear(layer_sizes[-1], z_dim)
    def forward(self, x):
        if self.FC is not None:
            x = self.FC(x)
        return self.mean_encoder(x)

class PDecoder(nn.Module):
    def __init__(self, z_dim, layer_sizes, x_dim, dr):
        super().__init__()
        layer_sizes = [z_dim] + layer_sizes
        self.FirstL = nn.Sequential()
        self.FirstL.add_module("L0", nn.Linear(layer_sizes[0], layer_sizes[1], bias=False))
        self.FirstL.add_module("N0", nn.BatchNorm1d(layer_sizes[1]))
        self.FirstL.add_module("A0", nn.LeakyReLU(0.3))
        self.FirstL.add_module("D0", nn.Dropout(p=dr))
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (ins, outs) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i + 3 < len(layer_sizes):
                    self.HiddenL.add_module(f"L{i+1}", nn.Linear(ins, outs, bias=False))
                    self.HiddenL.add_module(f"N{i+1}", nn.BatchNorm1d(outs, affine=True))
                    self.HiddenL.add_module(f"A{i+1}", nn.LeakyReLU(0.3))
                    self.HiddenL.add_module(f"D{i+1}", nn.Dropout(p=dr))
        else:
            self.HiddenL = None
        self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.relu = nn.ReLU()
    def forward(self, z):
        d = self.FirstL(z)
        x = self.HiddenL(d) if self.HiddenL is not None else d
        r = self.recon_decoder(x)
        dim = r.size(1) // 2
        return torch.cat((self.relu(r[:, :dim]), r[:, dim:]), 1)

class PAdaptor(nn.Module):
    def __init__(self, layer_sizes, comb_dim, dr):
        super().__init__()
        self.FC = None
        if len(layer_sizes) > 1:
            self.FC = nn.Sequential()
            for i, (ins, outs) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    self.FC.add_module(f"L{i}", nn.Linear(ins, outs, bias=False))
                else:
                    self.FC.add_module(f"L{i}", nn.Linear(ins, outs))
                    self.FC.add_module(f"N{i}", nn.BatchNorm1d(outs))
                    self.FC.add_module(f"A{i}", nn.LeakyReLU(0.3))
                    self.FC.add_module(f"D{i}", nn.Dropout(p=dr))
        self.comb_encoder = nn.Linear(layer_sizes[-1], comb_dim)
    def forward(self, x):
        if self.FC is not None:
            x = self.FC(x)
        return self.comb_encoder(x)

class PGM(nn.Module):
    def __init__(self, x_dim, c_dim, n_dim, hidden, z_dim, adaptor, comb_adapt_dim, dr):
        super().__init__()
        enc = hidden.copy(); enc.insert(0, x_dim + c_dim)
        dec = hidden.copy(); dec.reverse(); dec.append(x_dim * 2)
        adp = adaptor.copy(); adp.insert(0, comb_adapt_dim)
        self.encoder = PEncoder(enc, z_dim, dr)
        self.decoder = PDecoder(z_dim + c_dim + n_dim, dec, x_dim, dr)
        self.adaptor = PAdaptor(adp, c_dim, dr)
    def forward(self, x, c, n):
        c = self.adaptor(c)
        z = self.encoder(torch.cat((x, c), 1))
        return self.decoder(torch.cat((z, c, n), 1))

# ============================================================ hyperparameters (locked to the single-seed version)
X_DIM = 969
HIDDEN = [128]
Z_DIM = 64
ADAPTOR = [128]
C_DIM = 64
DRUG_DIM = 1026      # 1024 (Morgan) + dose_z + time_z
DR_RATE = 0.05
LR = 1e-3
WD = 1e-8
BATCH = 512
N_EPOCHS = 150
SCHED_PATIENCE = 10
EARLY_PATIENCE = 20
CHUNK = 20000

def make_model(device, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    return PGM(X_DIM, C_DIM, 10, HIDDEN, Z_DIM, ADAPTOR, DRUG_DIM, DR_RATE).to(device)

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        if m.bias is not None: m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1, 0.02)
        if m.bias is not None: m.bias.data.zero_()

# ============================================================ streaming batch generator
def iter_batches(h5, idx, Yname, dose_z, time_z, batch, shuffle, device, with_y=True):
    sidx = np.sort(idx)
    for s in range(0, len(sidx), CHUNK):
        blk = sidx[s:s+CHUNK]
        Xb = h5["X_baseline"][blk].astype(np.float32)
        de = h5["drug_emb"][blk].astype(np.float32)
        dz = dose_z[blk][:, None]; tz = time_z[blk][:, None]
        cond = np.concatenate([de, dz, tz], 1).astype(np.float32)
        Y = h5[Yname][blk].astype(np.float32) if with_y else None
        perm = np.arange(len(blk))
        if shuffle:
            np.random.shuffle(perm)
        for b in range(0, len(blk), batch):
            pi = perm[b:b+batch]
            xb_t = torch.from_numpy(Xb[pi]).to(device)
            cond_t = torch.from_numpy(cond[pi]).to(device)
            if with_y:
                y_t = torch.from_numpy(Y[pi]).to(device)
                yield xb_t, cond_t, y_t
            else:
                yield xb_t, cond_t

# ============================================================ vectorized per-sample correlation (much faster than row loops)
def corr_rows(pred, true, method):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if method == "pearson":
        pm = pred - pred.mean(1, keepdims=True)
        tm = true - true.mean(1, keepdims=True)
    else:  # spearman: rank then pearson
        pm = np.apply_along_axis(rankdata, 1, pred)
        tm = np.apply_along_axis(rankdata, 1, true)
        pm = pm - pm.mean(1, keepdims=True)
        tm = tm - tm.mean(1, keepdims=True)
    num = (pm * tm).sum(1)
    den = np.sqrt((pm**2).sum(1) * (tm**2).sum(1))
    return np.where(den > 1e-12, num / den, np.nan)

# ============================================================ main flow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/prnet_commonX/PRnet_commonX_dualY.h5")
    ap.add_argument("--outdir", default=r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/prnet_commonX")
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds")
    ap.add_argument("--seed-base", type=int, default=2024)
    ap.add_argument("--train-cap", type=int, default=200000, help="cap on training-sample subsampling (0 = full)")
    ap.add_argument("--limit-eval", type=int, default=None, help="cap on the S25 eval subset (None = full)")
    ap.add_argument("--tag", default="multiseed")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} seeds={args.seeds} seed_base={args.seed_base} train_cap={args.train_cap}")

    with h5py.File(args.data, "r") as f:
        split = f["split"][:].astype(int)
        dose = f["dose"][:].astype(np.float64)
        time = f["time"][:].astype(np.float64)
        n = len(split)
    dm, ds = dose.mean(), dose.std() or 1.0
    tm_, ts = time.mean(), time.std() or 1.0
    dose_z = ((dose - dm) / ds).astype(np.float32)
    time_z = ((time - tm_) / ts).astype(np.float32)

    train_idx = np.where(split == 0)[0]
    # fixed train subset (consistent across seeds), isolating seed variance to weight init / shuffle order
    if args.train_cap and 0 < args.train_cap < len(train_idx):
        rng_t = np.random.default_rng(12345)
        train_idx = rng_t.choice(train_idx, args.train_cap, replace=False)
        print(f"[train] capped to {len(train_idx)} samples (fixed across seeds)")
    else:
        print(f"[train] full {len(train_idx)} samples")

    val_idx = np.where(split == 1)[0]
    if len(val_idx) > 60000:
        rng_v = np.random.default_rng(2024)
        val_idx = rng_v.choice(val_idx, 60000, replace=False)
    eval_idx = np.where(split == 3)[0]
    if args.limit_eval:
        eval_idx = eval_idx[:args.limit_eval]
    print(f"[split] n={n} train={len(train_idx)} val={len(val_idx)} eval(S25)={len(eval_idx)}")

    with h5py.File(args.data, "r") as f:
        Yb = f["Y_beta"][eval_idx]; Yd = f["Y_dcic"][eval_idx]
    M = len(eval_idx)

    criterion = nn.GaussianNLLLoss()

    def train_arm(Yname, save_path, seed):
        print(f"\n===== TRAIN ARM  target={Yname}  seed={seed} =====")
        model = make_model(device, seed); model.apply(weight_init)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=SCHED_PATIENCE, min_lr=1e-8)
        best_mse, patient, best_sd = np.inf, 0, None
        with h5py.File(args.data, "r") as h5:
            for ep in range(args.epochs):
                model.train(); tot = 0.0; cnt = 0
                for xb, cond, y in iter_batches(h5, train_idx, Yname, dose_z, time_z, BATCH, True, device, True):
                    opt.zero_grad()
                    out = model(xb, cond, torch.randn(xb.size(0), 10, device=device))
                    dim = out.size(1)//2
                    mean = out[:, :dim]; logvar = torch.nn.functional.softplus(out[:, dim:])
                    loss = criterion(mean, y, logvar)
                    loss.backward(); opt.step(); tot += loss.item()*xb.size(0); cnt += xb.size(0)
                model.eval(); vmse = 0.0; vc = 0
                with torch.no_grad():
                    for xb, cond, y in iter_batches(h5, val_idx, Yname, dose_z, time_z, BATCH, False, device, True):
                        out = model(xb, cond, torch.randn(xb.size(0), 10, device=device))
                        dim = out.size(1)//2; mean = out[:, :dim]
                        vmse += float(torch.nn.functional.mse_loss(mean, y).item())*xb.size(0); vc += xb.size(0)
                vmse /= max(vc, 1)
                sched.step(vmse)
                print(f"  epoch {ep:3d} train_loss={tot/cnt:.4f} val_mse={vmse:.4f} best={best_mse:.4f}")
                if vmse < best_mse:
                    best_mse = vmse; patient = 0; best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                elif patient <= EARLY_PATIENCE:
                    patient += 1
                else:
                    print("  early stop"); break
        if best_sd is not None:
            torch.save(best_sd, save_path); print(f"  saved -> {save_path}")
        return best_sd

    def predict_arm(sd, idx):
        model = make_model(device, 0); model.load_state_dict(sd); model.eval()
        preds = []
        with h5py.File(args.data, "r") as h5:
            with torch.no_grad():
                for xb, cond in iter_batches(h5, idx, "Y_beta", dose_z, time_z, BATCH*2, False, device, False):
                    out = model(xb, cond, torch.randn(xb.size(0), 10, device=device))
                    preds.append(out[:, :out.size(1)//2].detach().cpu().numpy())
        return np.concatenate(preds, 0)

    # ---- run the 2x2 per seed ----
    BB_sp, BD_sp, DB_sp, DD_sp = [], [], [], []
    BB_pe, BD_pe, DB_pe, DD_pe = [], [], [], []
    seed_interactions = []   # (sp, pe)
    for k in range(args.seeds):
        seed = args.seed_base + k
        tag = f"{args.tag}_seed{k}"
        sd_beta = train_arm("Y_beta", os.path.join(args.outdir, f"{tag}_arm_beta.pt"), seed)
        sd_dcic = train_arm("Y_dcic", os.path.join(args.outdir, f"{tag}_arm_dcic.pt"), seed)
        pred_beta = predict_arm(sd_beta, eval_idx)
        pred_dcic = predict_arm(sd_dcic, eval_idx)
        bb_sp = corr_rows(pred_beta, Yb, "spearman"); bd_sp = corr_rows(pred_beta, Yd, "spearman")
        db_sp = corr_rows(pred_dcic, Yb, "spearman"); dd_sp = corr_rows(pred_dcic, Yd, "spearman")
        bb_pe = corr_rows(pred_beta, Yb, "pearson");  bd_pe = corr_rows(pred_beta, Yd, "pearson")
        db_pe = corr_rows(pred_dcic, Yb, "pearson");  dd_pe = corr_rows(pred_dcic, Yd, "pearson")
        # free the large prediction matrices
        del pred_beta, pred_dcic
        BB_sp.append(bb_sp); BD_sp.append(bd_sp); DB_sp.append(db_sp); DD_sp.append(dd_sp)
        BB_pe.append(bb_pe); BD_pe.append(bd_pe); DB_pe.append(db_pe); DD_pe.append(dd_pe)
        int_sp = float(np.nanmean((bb_sp - bd_sp) - (db_sp - dd_sp)))
        int_pe = float(np.nanmean((bb_pe - bd_pe) - (db_pe - dd_pe)))
        seed_interactions.append((int_sp, int_pe))
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print(f"[seed {seed}] BB_sp={np.nanmean(bb_sp):+.4f} BD_sp={np.nanmean(bd_sp):+.4f} "
              f"DB_sp={np.nanmean(db_sp):+.4f} DD_sp={np.nanmean(dd_sp):+.4f} interaction_sp={int_sp:+.4f}")

    n_seed = args.seeds
    # ---- aggregation table ----
    rows = []
    for k in range(n_seed):
        rows.append(dict(
            seed=args.seed_base + k,
            BB_sp=np.nanmean(BB_sp[k]), BD_sp=np.nanmean(BD_sp[k]),
            DB_sp=np.nanmean(DB_sp[k]), DD_sp=np.nanmean(DD_sp[k]),
            interaction_sp=seed_interactions[k][0],
            BB_pe=np.nanmean(BB_pe[k]), BD_pe=np.nanmean(BD_pe[k]),
            DB_pe=np.nanmean(DB_pe[k]), DD_pe=np.nanmean(DD_pe[k]),
            interaction_pe=seed_interactions[k][1],
        ))
    import csv
    sum_csv = os.path.join(args.outdir, f"{args.tag}_2x2_summary.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- hierarchical bootstrap (double resampling of seed + sample) ----
    rng = np.random.default_rng(args.seed_base)
    NB = 5000
    rep_sp, rep_pe = np.empty(NB), np.empty(NB)
    for b in range(NB):
        chosen = rng.integers(0, n_seed, n_seed)   # resample seeds
        v_sp, v_pe = [], []
        for s in chosen:
            idx = rng.integers(0, M, M)             # resample samples (within seed)
            d_sp = (BB_sp[s][idx] - BD_sp[s][idx]) - (DB_sp[s][idx] - DD_sp[s][idx])
            d_pe = (BB_pe[s][idx] - BD_pe[s][idx]) - (DB_pe[s][idx] - DD_pe[s][idx])
            v_sp.append(d_sp); v_pe.append(d_pe)
        rep_sp[b] = np.nanmean(np.concatenate(v_sp))
        rep_pe[b] = np.nanmean(np.concatenate(v_pe))
    boot_npz = os.path.join(args.outdir, f"{args.tag}_bootstrap.npz")
    np.savez(boot_npz,
             hierarchical_interaction_spearman=rep_sp,
             hierarchical_interaction_pearson=rep_pe,
             per_seed_interaction=np.array(seed_interactions, dtype=float))

    med_sp = float(np.median(rep_sp)); ci_sp = (float(np.percentile(rep_sp, 2.5)), float(np.percentile(rep_sp, 97.5)))
    med_pe = float(np.median(rep_pe)); ci_pe = (float(np.percentile(rep_pe, 2.5)), float(np.percentile(rep_pe, 97.5)))
    print("\n================ MULTI-SEED 2x2 RESULT (spearman) ================")
    for r in rows:
        print(f"  seed {r['seed']}: BB={r['BB_sp']:+.4f} BD={r['BD_sp']:+.4f} DB={r['DB_sp']:+.4f} DD={r['DD_sp']:+.4f} int={r['interaction_sp']:+.4f}")
    print(f"  hierarchical bootstrap interaction_sp = {med_sp:+.4f}  95%CI [{ci_sp[0]:+.4f},{ci_sp[1]:+.4f}]  (n_rep={NB}, crosses_zero={ci_sp[0]*ci_sp[1]<0})")
    print(f"  hierarchical bootstrap interaction_pe = {med_pe:+.4f}  95%CI [{ci_pe[0]:+.4f},{ci_pe[1]:+.4f}]  (crosses_zero={ci_pe[0]*ci_pe[1]<0})")
    print(f"  -> {sum_csv}")
    print(f"  -> {boot_npz}")

if __name__ == "__main__":
    main()
