#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRnet_commonX_train_eval.py -- train the two PRnet arms of the "full set A" controlled 2x2,
and run dual-truth evaluation on the S25 held-out set to get the four cells BB/BD/DB/DD and the interaction.

Design constraints (locked identically to PRnet_commonX_build.py):
  - both arms share exactly the same input X = [X_baseline(969), drug_emb(1024), dose_z, time_z] (1026)
  - both arms share exactly the same network architecture / hyperparameters / optimizer / seed / split
  - the only variable = training target reference: beta expression (Y_beta) vs dcic2021 CD signature (Y_dcic)
  - evaluation uses scale-invariant correlations (pearson + spearman, per-sample 969-dim vector correlation), aggregated across samples
  - interaction = (BB - BD) - (DB - DD), with a bootstrap 95% CI

Data: data/prnet_commonX/PRnet_commonX_dualY.h5 (produced by the build script)
Split: split 0/1/2 = train/val/test (used by the training arms); split 3 = S25 evaluation hold-out (not used in training)

Memory strategy: all arrays are streamed from h5 in row chunks, never loaded fully into RAM (handles the full ~3M samples).
"""
import argparse, json, os, sys, math, random
import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

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

# ============================================================ hyperparameters (locked to train_lincs.py)
X_DIM = 969
HIDDEN = [128]
Z_DIM = 64
ADAPTOR = [128]
C_DIM = 64           # comb_dimension
DRUG_DIM = 1026      # 1024 (smiles n-gram) + dose_z + time_z
DR_RATE = 0.05
LR = 1e-3
WD = 1e-8
BATCH = 512
N_EPOCHS = 200
SEED = 2024
SCHED_PATIENCE = 10
EARLY_PATIENCE = 20
CHUNK = 20000        # streaming row-chunk size

def make_model(device):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    return PGM(X_DIM, C_DIM, 10, HIDDEN, Z_DIM, ADAPTOR, DRUG_DIM, DR_RATE).to(device)

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        if m.bias is not None: m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1, 0.02)
        if m.bias is not None: m.bias.data.zero_()

# ============================================================ streaming batch generator (reads from disk, not RAM)
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

def corr_pair(pred, true, method):
    out = np.empty(pred.shape[0], dtype=np.float64)
    fn = pearsonr if method == "pearson" else spearmanr
    for r in range(pred.shape[0]):
        a, b = pred[r], true[r]
        if method == "pearson":
            out[r] = np.nan if (np.std(a) < 1e-9 or np.std(b) < 1e-9) else pearsonr(a, b)[0]
        else:
            out[r] = np.nan if (len(np.unique(a)) < 2 or len(np.unique(b)) < 2) else spearmanr(a, b)[0]
    return out

# ============================================================ main flow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/prnet_commonX/PRnet_commonX_dualY.h5")
    ap.add_argument("--outdir", default=r"C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/prnet_commonX")
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--limit-eval", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    with h5py.File(args.data, "r") as f:
        split = f["split"][:].astype(int)
        dose = f["dose"][:].astype(np.float64)
        time = f["time"][:].astype(np.float64)
        n = len(split)
    dm, ds = dose.mean(), dose.std() or 1.0
    tm, ts = time.mean(), time.std() or 1.0
    dose_z = ((dose - dm) / ds).astype(np.float32)
    time_z = ((time - tm) / ts).astype(np.float32)

    train_idx = np.where(split == 0)[0]
    val_idx = np.where(split == 1)[0]
    if len(val_idx) > 60000:   # full-scale val is huge; use a subset for early-stopping monitoring to speed up
        rng = np.random.default_rng(SEED)
        val_idx = rng.choice(val_idx, 60000, replace=False)
    eval_idx = np.where(split == 3)[0]
    if args.limit_eval:
        eval_idx = eval_idx[:args.limit_eval]
    print(f"[split] n={n} train={len(train_idx)} val={len(val_idx)} eval(S25)={len(eval_idx)}")

    criterion = nn.GaussianNLLLoss()

    def train_arm(Yname, save_path):
        print(f"\n===== TRAIN ARM  target={Yname} =====")
        model = make_model(device); model.apply(weight_init)
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
        model = make_model(device); model.load_state_dict(sd); model.eval()
        preds = []
        with h5py.File(args.data, "r") as h5:
            with torch.no_grad():
                for xb, cond in iter_batches(h5, idx, "Y_beta", dose_z, time_z, BATCH*2, False, device, False):
                    out = model(xb, cond, torch.randn(xb.size(0), 10, device=device))
                    preds.append(out[:, :out.size(1)//2].detach().cpu().numpy())
        return np.concatenate(preds, 0)

    tag = (args.tag + "_") if args.tag else ""
    sd_beta = train_arm("Y_beta", os.path.join(args.outdir, f"{tag}arm_beta.pt"))
    sd_dcic = train_arm("Y_dcic", os.path.join(args.outdir, f"{tag}arm_dcic.pt"))

    with h5py.File(args.data, "r") as f:
        Yb = f["Y_beta"][eval_idx]; Yd = f["Y_dcic"][eval_idx]

    pred_beta = predict_arm(sd_beta, eval_idx)
    pred_dcic = predict_arm(sd_dcic, eval_idx)

    BB_sp, BB_pe = corr_pair(pred_beta, Yb, "spearman"), corr_pair(pred_beta, Yb, "pearson")
    BD_sp, BD_pe = corr_pair(pred_beta, Yd, "spearman"), corr_pair(pred_beta, Yd, "pearson")
    DB_sp, DB_pe = corr_pair(pred_dcic, Yb, "spearman"), corr_pair(pred_dcic, Yb, "pearson")
    DD_sp, DD_pe = corr_pair(pred_dcic, Yd, "spearman"), corr_pair(pred_dcic, Yd, "pearson")
    nanm = lambda a: float(np.nanmean(a))
    BB_s, BD_s, DB_s, DD_s = nanm(BB_sp), nanm(BD_sp), nanm(DB_sp), nanm(DD_sp)
    BB_p, BD_p, DB_p, DD_p = nanm(BB_pe), nanm(BD_pe), nanm(DB_pe), nanm(DD_pe)
    int_sp = (BB_s - BD_s) - (DB_s - DD_s)
    int_pe = (BB_p - BD_p) - (DB_p - DD_p)

    rng = np.random.default_rng(SEED); M = len(eval_idx)
    boot_sp, boot_pe = [], []
    for _ in range(1000):
        s = rng.integers(0, M, M)
        boot_sp.append((np.nanmean(BB_sp[s]) - np.nanmean(BD_sp[s])) - (np.nanmean(DB_sp[s]) - np.nanmean(DD_sp[s])))
        boot_pe.append((np.nanmean(BB_pe[s]) - np.nanmean(BD_pe[s])) - (np.nanmean(DB_pe[s]) - np.nanmean(DD_pe[s])))
    ci_sp = (np.percentile(boot_sp, 2.5), np.percentile(boot_sp, 97.5))
    ci_pe = (np.percentile(boot_pe, 2.5), np.percentile(boot_pe, 97.5))

    result = dict(
        design="PRnet_commonX controlled 2x2: shared X, swapped training target (beta vs dcic)",
        n_eval=int(M),
        cells_spearman=dict(BB=BB_s, BD=BD_s, DB=DB_s, DD=DD_s),
        cells_pearson=dict(BB=BB_p, BD=BD_p, DB=DB_p, DD=DD_p),
        interaction_spearman=float(int_sp), interaction_spearman_ci95=[float(ci_sp[0]), float(ci_sp[1])],
        interaction_pearson=float(int_pe), interaction_pearson_ci95=[float(ci_pe[0]), float(ci_pe[1])],
        hyperparams=dict(x_dim=X_DIM, hidden=HIDDEN, z_dim=Z_DIM, adaptor=ADAPTOR, comb_dim=C_DIM,
                         drug_dim=DRUG_DIM, dr=DR_RATE, lr=LR, wd=WD, batch=BATCH, epochs=args.epochs, seed=SEED),
    )
    out_json = os.path.join(args.outdir, f"{tag}commonX_2x2_result.json")
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    detail = np.column_stack([BB_sp, BD_sp, DB_sp, DD_sp, BB_pe, BD_pe, DB_pe, DD_pe])
    np.savetxt(os.path.join(args.outdir, f"{tag}commonX_2x2_detail.csv"), detail,
               delimiter=",", header="BB_sp,BD_sp,DB_sp,DD_sp,BB_pe,BD_pe,DB_pe,DD_pe")

    print("\n================ 2x2 RESULT (spearman, per-sample 969-dim vector corr) ================")
    print(f"  BB (beta-arm | beta-truth)  = {BB_s:+.4f}")
    print(f"  BD (beta-arm | dcic-truth)  = {BD_s:+.4f}")
    print(f"  DB (dcic-arm | beta-truth)  = {DB_s:+.4f}")
    print(f"  DD (dcic-arm | dcic-truth)  = {DD_s:+.4f}")
    print(f"  interaction = (BB-BD)-(DB-DD) = {int_sp:+.4f}  95%CI [{ci_sp[0]:+.4f},{ci_sp[1]:+.4f}]")
    print(f"  [pearson]   BB={BB_p:+.4f} BD={BD_p:+.4f} DB={DB_p:+.4f} DD={DD_p:+.4f} int={int_pe:+.4f} CI[{ci_pe[0]:+.4f},{ci_pe[1]:+.4f}]")
    print(f"  -> {out_json}")

if __name__ == "__main__":
    main()
