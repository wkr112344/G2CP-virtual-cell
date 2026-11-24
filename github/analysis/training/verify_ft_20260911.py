"""verify_ft_20260911.py — v7 权重在【公开 beta 缓存】上做一次域内适配后，能不能全面超过预印本

动机（用户问："我要和原先预印本一样的，只能好不能差，能做到吗"）：
  预印本的所有数字（0.3047 / 0.442 / 0.480 vs 0.296 / 0.359）建在 143 系 / 305,297 样本语料上。
  公开 beta 缓存是它的子集（123 系 / 157,522 对），直接评测必然更低（已实测：逐细胞系 0.2753 vs 0.359）。
  但工作区自有证据（G2CP_阶段一_更正版结论.md）指出：**未做目标域适配的 checkpoint 会系统性低估**，
  同一个 v7 只训 head 就从 0.1545 → 0.2501（+0.0956，CI 不含 0）。

  所以本脚本检验：在公开 beta 缓存上按同一协议做一次全量微调（cp_lin + cell_emb + head），
  官方留出划分下的 PCC / 逐细胞系 PCC 能不能达到或超过预印本。

协议：官方 RandomState(0) 后 10% 药物留出 → test 16,065；其余 90% 作 train（其中再切 5% 当 val）
      损失 = MSE + (1 - per-sample PCC)；AdamW lr 3e-4 wd 1e-4；余弦退火；batch 256
      gene_emb 不参与该前向路径（冻结，保持与论文一致）
输出：results/verify_ft_20260911.json + assets/g2cp_v7_beta_ft.pt
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\Users\wkr20\WorkBuddy\2026-09-06-00-53-37"
CACHE = os.path.join(BASE, "data", "g2cp_cache_beta_all")
ASSETS = os.path.join(BASE, "assets")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, EPOCHS, LR, WD, BATCH = 0, 24, 3e-4, 1e-4, 256
torch.manual_seed(SEED); np.random.seed(SEED)


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    t0 = time.time()
    meta = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
    key = meta["key"].astype(np.int64); cell = meta["cell"].astype(np.int64)
    kind = meta["kind"].astype(np.int64)
    cl_names = [str(x) for x in meta["cl_names"]]
    y = np.load(os.path.join(CACHE, "y.npy"), mmap_mode="r")
    fps = np.load(os.path.join(CACHE, "drug_fps.npy"))
    N, G = y.shape
    ck = torch.load(os.path.join(ASSETS, "g2cp_full_cpi_v7.pt"), map_location="cpu", weights_only=False)
    sd = ck["net"]; NGENE = sd["gene_emb.weight"].shape[0]; NC = sd["cell_emb.weight"].shape[0]

    pert_uid = np.where(kind == 0, key, NGENE + key)
    rng = np.random.RandomState(0)
    perts = sorted(set(int(p) for p in pert_uid))
    perm = rng.permutation(len(perts))
    te_set = set(int(perts[i]) for i in perm[int(len(perts) * 0.9):])
    is_te = np.array([int(p) in te_set for p in pert_uid])
    te_idx = np.where(is_te)[0]; tr_all = np.where(~is_te)[0]
    r2 = np.random.RandomState(SEED); p2 = r2.permutation(len(tr_all))
    n_val = int(len(tr_all) * 0.05)
    va_idx = tr_all[p2[:n_val]]; tr_idx = tr_all[p2[n_val:]]
    log("train=%d val=%d test=%d (test pert=%d)" % (len(tr_idx), len(va_idx), len(te_idx), len(te_set)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cp = nn.Linear(2048, 512)
            self.cell = nn.Embedding(NC, 32)
            self.head = nn.Sequential(nn.LayerNorm(544), nn.Linear(544, 1024), nn.GELU(), nn.Linear(1024, G))

        def forward(self, fp, ci):
            return self.head(torch.cat([F.normalize(self.cp(fp), dim=-1), self.cell(ci)], 1))

    net = Net()
    own = net.state_dict()
    own["cp.weight"] = sd["cp_lin.weight"]; own["cp.bias"] = sd["cp_lin.bias"]
    own["cell.weight"] = sd["cell_emb.weight"]
    for k in ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias", "head.3.weight", "head.3.bias"]:
        own[k] = sd[k]
    net.load_state_dict(own)
    init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net = net.to(DEV)

    Xd_tr = torch.from_numpy(np.asarray(fps[key[tr_idx]]))
    Xc_tr = torch.from_numpy(cell[tr_idx])
    Y_tr = torch.from_numpy(np.asarray(y[tr_idx], dtype=np.float32))
    Xd_va = torch.from_numpy(np.asarray(fps[key[va_idx]]))
    Xc_va = torch.from_numpy(cell[va_idx])
    Y_va = torch.from_numpy(np.asarray(y[va_idx], dtype=np.float32))
    log("data loaded %.0fs" % (time.time() - t0))

    def pcc_rows(P, T):
        a = P - P.mean(1, keepdims=True); b = T - T.mean(1, keepdims=True)
        return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-8)

    def evaluate():
        net.eval(); outs = []
        with torch.no_grad():
            for s in range(0, len(te_idx), 1024):
                r = te_idx[s:s + 1024]
                outs.append(net(torch.from_numpy(fps[key[r]]).to(DEV),
                                torch.from_numpy(cell[r]).to(DEV)).cpu().numpy().astype(np.float32))
        P = np.concatenate(outs, 0)
        T = np.asarray(y[te_idx], dtype=np.float32)
        pc = pcc_rows(P, T); dirv = (np.sign(P) == np.sign(T))
        cellte = cell[te_idx]
        groups = [np.where(key[te_idx] == g)[0] for g in np.unique(key[te_idx])]
        um = np.array([np.nanmean(pc[g]) for g in groups])
        percell = np.array([pc[cellte == c].mean() for c in np.unique(cellte)])
        return dict(pcc_sample=float(pc.mean()), pcc_drug=float(um.mean()),
                    dir=float(dirv.mean()),
                    percell_mean=float(percell.mean()), percell_median=float(np.median(percell)),
                    percell_min=float(percell.min()), percell_max=float(percell.max()),
                    n_cells=int(len(np.unique(cellte))))

    base = evaluate()
    log("BASELINE(v7 原始权重) 逐样本=%.4f 逐药物=%.4f dir=%.2f%% | 逐细胞系 mean=%.4f median=%.4f"
        % (base["pcc_sample"], base["pcc_drug"], base["dir"] * 100, base["percell_mean"], base["percell_median"]))

    params = [net.cp.weight, net.cp.bias, net.cell.weight] + list(net.head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = (-1, None)
    for ep in range(EPOCHS):
        net.train(); perm2 = torch.randperm(len(Y_tr))
        tot = 0.0; nb = 0
        for i in range(0, len(perm2), BATCH):
            b = perm2[i:i + BATCH]
            xd = Xd_tr[b].to(DEV, non_blocking=True)
            xc = Xc_tr[b].to(DEV, non_blocking=True)
            yb = Y_tr[b].to(DEV, non_blocking=True)
            pr = net(xd, xc)
            # 损失 = MSE + (1 - per-sample PCC)，PCC 项需带梯度
            a = pr - pr.mean(1, keepdim=True); bb = yb - yb.mean(1, keepdim=True)
            pccg = ((a * bb).sum(1) / torch.sqrt((a * a).sum(1) * (bb * bb).sum(1) + 1e-8)).mean()
            loss = F.mse_loss(pr, yb) + (1.0 - pccg)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xd_va.to(DEV), Xc_va.to(DEV)).cpu().numpy()
        vp = float(pcc_rows(pv, Y_va.numpy()).mean())
        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        with torch.no_grad():
            net.load_state_dict(state)
        if vp > best[0]:
            best = (vp, {k: v.clone() for k, v in state.items()})
        if ep % 2 == 0 or ep == EPOCHS - 1:
            log("  ep%02d loss=%.4f val_pcc=%.4f (best %.4f) %.0fs" % (ep, tot / max(nb, 1), vp, best[0], time.time() - t0))

    net.load_state_dict(best[1]); net = net.to(DEV)
    after = evaluate()
    log("AFTER-FT(beta 域内适配) 逐样本=%.4f 逐药物=%.4f dir=%.2f%% | 逐细胞系 mean=%.4f median=%.4f min=%.4f max=%.4f"
        % (after["pcc_sample"], after["pcc_drug"], after["dir"] * 100,
           after["percell_mean"], after["percell_median"], after["percell_min"], after["percell_max"]))

    print()
    print("=" * 96)
    print("能否超过预印本？（预印本：新药 0.3047 / dir 60.8% / top5 73.8% / 逐细胞系均值 0.359）")
    print("=" * 96)
    for tag, o in [("v7 原始权重（未适配）", base), ("v7 beta 域内适配后", after)]:
        print("%-24s 逐样本=%.4f 逐药物=%.4f dir=%.2f%% 逐细胞系均值=%.4f"
              % (tag, o["pcc_sample"], o["pcc_drug"], o["dir"] * 100, o["percell_mean"]))
    print("-" * 96)
    print("Δ 逐细胞系均值 = %+.4f ；Δ 逐药物 PCC = %+.4f"
          % (after["percell_mean"] - base["percell_mean"], after["pcc_drug"] - base["pcc_drug"]))
    print("适配后 vs 预印本 0.359（逐细胞系均值）= %+.4f" % (after["percell_mean"] - 0.359))
    print("适配后 vs 预印本 0.3047（新药 PCC，逐药物等权）= %+.4f" % (after["pcc_drug"] - 0.3047))

    torch.save({"net": {k: v.cpu() for k, v in best[1].items()}, "cl_names": ck["cl_names"],
                "drug_vocab": ck["drug_vocab"]}, os.path.join(ASSETS, "g2cp_v7_beta_ft.pt"))
    json.dump({"baseline": base, "after_ft": after,
               "preprint_ref": {"novel_drug_pcc": 0.3047, "dir": 0.608, "top5": 0.738, "percell_mean": 0.359},
               "val_pcc_best": float(best[0]), "epochs": EPOCHS,
               "delta_percell": after["percell_mean"] - base["percell_mean"],
               "delta_pcc_drug": after["pcc_drug"] - base["pcc_drug"]},
              open(os.path.join(BASE, "results", "verify_ft_20260911.json"), "w"), indent=1)
    log("saved results/verify_ft_20260911.json + assets/g2cp_v7_beta_ft.pt | total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
