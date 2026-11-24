"""verify_ft2_20260911.py — v7 权重在公开 beta 语料上的「域内适配」第二轮：可配置损失与轮数。

动机：β 适配（`verify_ft_20260911.py`，loss = MSE + 1−逐样本PCC）把逐细胞系均值从 0.2773 抬到 0.3249、
癌系从 0.2687 抬到 0.3171（均超过预印本 0.359 / 0.296 的目标之一），但
  · 逐细胞系中位数 0.3435 < 0.362
  · 逐药物等权池化 PCC 0.3271 < v7 的 0.3414
本脚本试三种损失/轮数组合，看能不能把中位数与逐药物口径同时抬起来。

用法：
  python verify_ft2_20260911.py --loss both --epochs 32 --lr 2e-4 --out-pt assets/g2cp_v7_beta_ft2_both.pt
"""
import argparse, csv, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

BASE = r"C:\Users\wkr20\WorkBuddy\2026-09-06-00-53-37"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/g2cp_cache_beta_v2")
    ap.add_argument("--base", default="assets/g2cp_full_cpi_v7.pt")
    ap.add_argument("--loss", default="both", choices=["sample", "drug", "both"])
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out-pt", default="assets/g2cp_v7_beta_ft2.pt")
    ap.add_argument("--out-json", default="results/verify_ft2_20260911.json")
    a = ap.parse_args()
    P_ = lambda x: x if os.path.isabs(x) else os.path.join(BASE, x)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()

    meta = np.load(os.path.join(P_(a.cache), "meta.npz"), allow_pickle=True)
    key = meta["key"].astype(np.int64); cell = meta["cell"].astype(np.int64)
    kind = meta["kind"].astype(np.int64)
    y = np.load(os.path.join(P_(a.cache), "y.npy"), mmap_mode="r")
    fps = np.load(os.path.join(P_(a.cache), "drug_fps.npy"))
    N, G = y.shape
    ck = torch.load(P_(a.base), map_location="cpu", weights_only=False)
    sd = ck["net"]
    NGENE = sd["gene_emb.weight"].shape[0]
    NC = sd["cell_emb.weight"].shape[0] if "cell_emb.weight" in sd else sd["cell.weight"].shape[0]

    pert_uid = np.where(kind == 0, key, NGENE + key)
    rng = np.random.RandomState(0)
    perts = sorted(set(int(p) for p in pert_uid))
    perm = rng.permutation(len(perts))
    te_set = set(int(perts[i]) for i in perm[int(len(perts) * 0.9):])
    is_te = np.array([int(p) in te_set for p in pert_uid])
    te_idx = np.where(is_te)[0]; tr_all = np.where(~is_te)[0]
    r2 = np.random.RandomState(a.seed); p2 = r2.permutation(len(tr_all))
    n_val = int(len(tr_all) * a.val_frac)
    va_idx = tr_all[p2[:n_val]]; tr_idx = tr_all[p2[n_val:]]
    log("cache=%s base=%s loss=%s ep=%d lr=%g | train=%d val=%d test=%d"
        % (os.path.basename(P_(a.cache)), os.path.basename(P_(a.base)), a.loss, a.epochs, a.lr,
           len(tr_idx), len(va_idx), len(te_idx)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cp = nn.Linear(2048, 512)
            self.cell = nn.Embedding(NC, 32)
            self.head = nn.Sequential(nn.LayerNorm(544), nn.Linear(544, 1024), nn.GELU(), nn.Linear(1024, G))

        def forward(self, fp, ci):
            return self.head(torch.cat([F.normalize(self.cp(fp), dim=-1), self.cell(ci)], 1))

    net = Net(); own = net.state_dict()
    own["cp.weight"] = sd["cp_lin.weight"] if "cp_lin.weight" in sd else sd["cp.weight"]
    own["cp.bias"] = sd["cp_lin.bias"] if "cp_lin.bias" in sd else sd["cp.bias"]
    own["cell.weight"] = (sd["cell_emb.weight"] if "cell_emb.weight" in sd else sd["cell.weight"])[:NC]
    for k in ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias", "head.3.weight", "head.3.bias"]:
        own[k] = sd[k]
    net.load_state_dict(own); net = net.to(DEV)

    T = np.asarray(y[te_idx], dtype=np.float32)
    te_key = key[te_idx]; te_cell = cell[te_idx]
    lab = {}
    for r in csv.DictReader(open(os.path.join(BASE, "results", "stage60_e9_cell_difficulty.csv"), encoding="utf-8")):
        lab[r["cell"].strip()] = r["is_tumor"].strip()
    cl_names = [str(x) for x in meta["cl_names"]]

    def pcc_rows(P):
        x = P - P.mean(1, keepdims=True); z = T - T.mean(1, keepdims=True)
        return (x * z).sum(1) / (np.sqrt((x * x).sum(1) * (z * z).sum(1)) + 1e-8)

    def predict(idx):
        net.eval(); outs = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                r = idx[s:s + 1024]
                outs.append(net(torch.from_numpy(fps[key[r]]).to(DEV),
                                torch.from_numpy(cell[r]).to(DEV)).cpu().numpy().astype(np.float32))
        return np.concatenate(outs, 0)

    def evaluate():
        pc = pcc_rows(predict(te_idx))
        dmeans = np.array([pc[te_key == g].mean() for g in np.unique(te_key)])
        rows = [(cl_names[int(c)], int((te_cell == c).sum()), float(pc[te_cell == c].mean()))
                for c in np.unique(te_cell)]
        cv = np.array([v for c, n, v in rows if lab.get(c, "") == "1"])
        nv = np.array([v for c, n, v in rows if lab.get(c, "") == "0"])
        allv = np.array([v for c, n, v in rows])
        return dict(pcc_sample=float(pc.mean()), pcc_drug=float(dmeans.mean()),
                    percell_mean=float(allv.mean()), percell_median=float(np.median(allv)),
                    cancer_mean=float(cv.mean()), cancer_median=float(np.median(cv)),
                    noncancer_mean=float(nv.mean()), n_cells=len(rows), rows=rows)

    base = evaluate()
    log("BASE 逐样本=%.4f 逐药物=%.4f 逐系 mean=%.4f median=%.4f 癌=%.4f"
        % (base["pcc_sample"], base["pcc_drug"], base["percell_mean"], base["percell_median"], base["cancer_mean"]))

    Xd_tr = torch.from_numpy(np.asarray(fps[key[tr_idx]]))
    Xc_tr = torch.from_numpy(cell[tr_idx])
    Y_tr = torch.from_numpy(np.asarray(y[tr_idx], dtype=np.float32))
    kg_tr = torch.from_numpy(key[tr_idx].astype(np.int64))
    n_drug_tr = int(key[tr_idx].max()) + 1
    Xd_va = torch.from_numpy(np.asarray(fps[key[va_idx]])); Xc_va = torch.from_numpy(cell[va_idx])
    Y_va = torch.from_numpy(np.asarray(y[va_idx], dtype=np.float32))

    params = [net.cp.weight, net.cp.bias, net.cell.weight] + list(net.head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best = (-1, None, None)
    for ep in range(a.epochs):
        net.train(); perm2 = torch.randperm(len(Y_tr)); tot = 0.0; nb = 0
        for i in range(0, len(perm2), a.batch):
            b = perm2[i:i + a.batch]
            xd = Xd_tr[b].to(DEV); xc = Xc_tr[b].to(DEV); yb = Y_tr[b].to(DEV)
            kd = kg_tr[b].to(DEV)
            pr = net(xd, xc)
            aa = pr - pr.mean(1, keepdim=True); bb = yb - yb.mean(1, keepdim=True)
            pcv = (aa * bb).sum(1) / torch.sqrt((aa * aa).sum(1) * (bb * bb).sum(1) + 1e-8)
            loss = F.mse_loss(pr, yb)
            if a.loss == "sample":
                loss = loss + (1.0 - pcv.mean())
            elif a.loss == "drug":
                cnt = torch.bincount(kd, minlength=n_drug_tr).clamp(min=1).float()
                w = 1.0 / cnt[kd]; w = w / w.mean()
                loss = loss + (1.0 - (pcv * w).sum() / w.sum())
            else:
                cnt = torch.bincount(kd, minlength=n_drug_tr).clamp(min=1).float()
                w = 1.0 / cnt[kd]; w = w / w.mean()
                loss = loss + 0.5 * (1.0 - pcv.mean()) + 0.5 * (1.0 - (pcv * w).sum() / w.sum())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = predict(va_idx)
        va = pv - pv.mean(1, keepdims=True); vb = Y_va.numpy() - Y_va.numpy().mean(1, keepdims=True)
        vp = float(((va * vb).sum(1) / np.sqrt((va * va).sum(1) * (vb * vb).sum(1) + 1e-8)).mean())
        if vp > best[0]:
            best = (vp, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
        if ep % 4 == 0 or ep == a.epochs - 1:
            log("  ep%02d loss=%.4f val=%.4f (best %.4f@ep%s) %.0fs"
                % (ep, tot / max(nb, 1), vp, best[0], best[2], time.time() - t0))

    net.load_state_dict(best[1]); net = net.to(DEV)
    after = evaluate()
    log("AFTER 逐样本=%.4f 逐药物=%.4f 逐系 mean=%.4f median=%.4f 癌 mean=%.4f median=%.4f"
        % (after["pcc_sample"], after["pcc_drug"], after["percell_mean"], after["percell_median"],
           after["cancer_mean"], after["cancer_median"]))
    log("Δ vs BASE：逐样本 %+.4f 逐药物 %+.4f 逐系 mean %+.4f median %+.4f 癌 %+.4f"
        % (after["pcc_sample"] - base["pcc_sample"], after["pcc_drug"] - base["pcc_drug"],
           after["percell_mean"] - base["percell_mean"], after["percell_median"] - base["percell_median"],
           after["cancer_mean"] - base["cancer_mean"]))

    torch.save({"net": {k: v.cpu() for k, v in best[1].items()},
                "cl_names": ck.get("cl_names"), "drug_vocab": ck.get("drug_vocab")}, P_(a.out_pt))
    json.dump(dict(config=vars(a), baseline=base, after=after, best_val=best[0], best_epoch=best[2]),
              open(P_(a.out_json), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log("saved %s / %s | %.0fs" % (P_(a.out_pt), P_(a.out_json), time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
