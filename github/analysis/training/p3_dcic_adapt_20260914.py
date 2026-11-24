"""p3_dcic_adapt_20260914.py — P3：在 dcic LINCS-sigs-2021 缓存上做**同口径**域内适配

目的（量化「ΔPCC = +0.2480 里有多少来自适配状态失配」）：
  跨 level-5 发布版本的落差此前只在 β 主源一侧做过域内适配，dcic 2021 侧从未适配，
  因此 +0.2480 中「模型未在该源适配」这一成分一直未被定量。本脚本把该臂补齐，
  使落差在「两侧都适配」与「两侧都未适配」两种状态下各有一个读数。

设计（与 verify_ft_20260911.py 的协议逐字一致，只把 CACHE 从 beta_all 换成 2021）：
  同一基座权重 assets/g2cp_full_cpi_v7.pt；同一可训练面（cp_lin + cell_emb + 表型头）；
  同一切分口径（官方 RandomState(0) 后 10% 扰动留出；训练段内再切 5% 作验证）；
  同一损失 MSE + (1 − 逐样本 PCC)；同一 AdamW lr 3e-4 wd 1e-4 bs 256 余弦退火 24 轮。
  ⇒ 与 β 臂（verify_ft_20260911.json：逐样本 0.3368 → 0.3443）**成对可比**。

跑法：
  python p3_dcic_adapt_20260914.py [--epochs N] [--threads T]

依赖：data/g2cp_cache_2021/（预处理缓存，未随归档发布）、assets/g2cp_full_cpi_v7.pt（基座权重）
输出：results/p3_dcic_adapt_20260914.json、assets/g2cp_v7_dcic_ft.pt
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = _HERE if os.path.isdir(os.path.join(_HERE, "results")) else os.path.dirname(_HERE)
CACHE = os.path.join(BASE, "data", "g2cp_cache_2021")
ASSETS = os.path.join(BASE, "assets")
OUT_JSON = os.path.join(BASE, "results", "p3_dcic_adapt_20260914.json")
OUT_PT = os.path.join(ASSETS, "g2cp_v7_dcic_ft.pt")
BASE_CKPT = os.path.join(ASSETS, "g2cp_full_cpi_v7.pt")

SEED, LR, WD, BATCH = 0, 3e-4, 1e-4, 256


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def pcc_rows(P, T):
    a = P - P.mean(1, keepdims=True)
    b = T - T.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tag", default="dcic")
    a = ap.parse_args()
    EPOCHS = a.epochs
    torch.set_num_threads(a.threads)

    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)

    meta = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
    key = meta["key"].astype(np.int64)
    cell = meta["cell"].astype(np.int64)
    kind = meta["kind"].astype(np.int64)
    y = np.load(os.path.join(CACHE, "y.npy"), mmap_mode="r")
    fps = np.load(os.path.join(CACHE, "drug_fps.npy"))
    N, G = y.shape
    log("source=%s  N=%d  G=%d  threads=%d  torch=%s" % (a.tag, N, G, a.threads, torch.__version__))

    ck = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    sd = ck["net"]
    NGENE = sd["gene_emb.weight"].shape[0]
    NC = sd["cell_emb.weight"].shape[0]
    log("base ckpt=%s  NGENE=%d  NC=%d" % (os.path.basename(BASE_CKPT), NGENE, NC))

    # ---- 切分：官方 RandomState(0) 后 10% 扰动留出（与 verify_ft 逐字一致）
    pert_uid = np.where(kind == 0, key, NGENE + key)
    rng = np.random.RandomState(0)
    perts = sorted(set(int(p) for p in pert_uid))
    perm = rng.permutation(len(perts))
    te_set = set(int(perts[i]) for i in perm[int(len(perts) * 0.9):])
    is_te = np.array([int(p) in te_set for p in pert_uid])
    te_idx = np.where(is_te)[0]
    tr_all = np.where(~is_te)[0]
    r2 = np.random.RandomState(SEED); p2 = r2.permutation(len(tr_all))
    n_val = int(len(tr_all) * 0.05)
    va_idx = tr_all[p2[:n_val]]; tr_idx = tr_all[p2[n_val:]]
    log("train=%d val=%d test=%d (test pert=%d, n_cell_in_test=%d)"
        % (len(tr_idx), len(va_idx), len(te_idx), len(te_set), len(np.unique(cell[te_idx]))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cp = nn.Linear(2048, 512)
            self.cell = nn.Embedding(NC, 32)
            self.head = nn.Sequential(nn.LayerNorm(544), nn.Linear(544, 1024), nn.GELU(),
                                      nn.Linear(1024, G))

        def forward(self, fp, ci):
            return self.head(torch.cat([F.normalize(self.cp(fp), dim=-1), self.cell(ci)], 1))

    net = Net()
    own = net.state_dict()
    own["cp.weight"] = sd["cp_lin.weight"]; own["cp.bias"] = sd["cp_lin.bias"]
    own["cell.weight"] = sd["cell_emb.weight"]
    for k in ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias",
              "head.3.weight", "head.3.bias"]:
        own[k] = sd[k]
    net.load_state_dict(own)
    net = net.to("cpu")

    Xd_tr = torch.from_numpy(np.asarray(fps[key[tr_idx]]))
    Xc_tr = torch.from_numpy(cell[tr_idx])
    Y_tr = torch.from_numpy(np.asarray(y[tr_idx], dtype=np.float32))
    Xd_va = torch.from_numpy(np.asarray(fps[key[va_idx]]))
    Xc_va = torch.from_numpy(cell[va_idx])
    Y_va = torch.from_numpy(np.asarray(y[va_idx], dtype=np.float32))
    log("data loaded %.0fs" % (time.time() - t0))

    def evaluate():
        net.eval(); outs = []
        with torch.no_grad():
            for s in range(0, len(te_idx), 1024):
                r = te_idx[s:s + 1024]
                outs.append(net(torch.from_numpy(fps[key[r]]),
                                torch.from_numpy(cell[r])).numpy().astype(np.float32))
        P = np.concatenate(outs, 0)
        T = np.asarray(y[te_idx], dtype=np.float32)
        pc = pcc_rows(P, T); dirv = (np.sign(P) == np.sign(T))
        cellte = cell[te_idx]
        groups = [np.where(key[te_idx] == g)[0] for g in np.unique(key[te_idx])]
        um = np.array([np.nanmean(pc[g]) for g in groups])
        percell = np.array([pc[cellte == c].mean() for c in np.unique(cellte)])
        return dict(pcc_sample=float(pc.mean()), pcc_drug=float(um.mean()),
                    dir=float(dirv.mean()),
                    percell_mean=float(percell.mean()),
                    percell_median=float(np.median(percell)),
                    percell_min=float(percell.min()), percell_max=float(percell.max()),
                    n_cells=int(len(np.unique(cellte))))

    base = evaluate()
    log("BASELINE(v7 raw on %s) 逐样本=%.4f 逐药物=%.4f dir=%.2f%% | 逐细胞系 mean=%.4f median=%.4f min=%.4f max=%.4f n=%d"
        % (a.tag, base["pcc_sample"], base["pcc_drug"], base["dir"] * 100,
           base["percell_mean"], base["percell_median"], base["percell_min"],
           base["percell_max"], base["n_cells"]))

    params = [net.cp.weight, net.cp.bias, net.cell.weight] + list(net.head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = (-1, None)
    curve = []
    for ep in range(EPOCHS):
        net.train(); perm2 = torch.randperm(len(Y_tr))
        tot = 0.0; nb = 0
        for i in range(0, len(perm2), BATCH):
            b = perm2[i:i + BATCH]
            xd = Xd_tr[b]; xc = Xc_tr[b]; yb = Y_tr[b]
            pr = net(xd, xc)
            aa = pr - pr.mean(1, keepdim=True); bb = yb - yb.mean(1, keepdim=True)
            pccg = ((aa * bb).sum(1) / torch.sqrt((aa * aa).sum(1) * (bb * bb).sum(1) + 1e-8)).mean()
            loss = F.mse_loss(pr, yb) + (1.0 - pccg)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xd_va, Xc_va).numpy()
        vp = float(pcc_rows(pv, Y_va.numpy()).mean())
        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if vp > best[0]:
            best = (vp, {k: v.clone() for k, v in state.items()})
        curve.append(vp)
        log("  ep%02d loss=%.4f val_pcc=%.4f (best %.4f) %.0fs"
            % (ep, tot / max(nb, 1), vp, best[0], time.time() - t0))

    net.load_state_dict(best[1])
    after = evaluate()
    log("AFTER-FT(%s 域内适配) 逐样本=%.4f 逐药物=%.4f dir=%.2f%% | 逐细胞系 mean=%.4f median=%.4f min=%.4f max=%.4f"
        % (a.tag, after["pcc_sample"], after["pcc_drug"], after["dir"] * 100,
           after["percell_mean"], after["percell_median"], after["percell_min"], after["percell_max"]))

    print()
    print("=" * 96)
    print("P3 · 同口径域内适配：beta_all 臂 vs %s 臂" % a.tag)
    print("=" * 96)
    print("%-34s %-10s %-10s %-10s %-10s" % ("", "逐样本", "逐药物", "dir", "逐细胞系均值"))
    print("%-34s %-10.4f %-10.4f %-10.2f %-10.4f"
          % ("%s  v7 raw（未适配）" % a.tag, base["pcc_sample"], base["pcc_drug"],
             base["dir"] * 100, base["percell_mean"]))
    print("%-34s %-10.4f %-10.4f %-10.2f %-10.4f"
          % ("%s  域内适配后" % a.tag, after["pcc_sample"], after["pcc_drug"],
             after["dir"] * 100, after["percell_mean"]))
    print("-" * 96)
    print("Δ 逐样本 = %+.4f ；Δ 逐药物 = %+.4f ；Δ 逐细胞系均值 = %+.4f"
          % (after["pcc_sample"] - base["pcc_sample"],
             after["pcc_drug"] - base["pcc_drug"],
             after["percell_mean"] - base["percell_mean"]))
    print("β 臂参照（verify_ft_20260911.json，beta_all 缓存）：逐样本 0.3368 → 0.3443（Δ +0.0075）；逐细胞系均值 0.2753 → 0.3234（Δ +0.0481）")

    json.dump({"tag": a.tag, "cache": CACHE.replace(BASE + os.sep, ""),
               "base_ckpt": os.path.basename(BASE_CKPT),
               "epochs": EPOCHS, "lr": LR, "wd": WD, "batch": BATCH,
               "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
               "n_test": int(len(te_idx)), "n_test_pert": int(len(te_set)),
               "n_cell_in_test": int(len(np.unique(cell[te_idx]))),
               "baseline": base, "after_ft": after, "val_curve": curve,
               "val_pcc_best": float(best[0]),
               "delta_pcc_sample": after["pcc_sample"] - base["pcc_sample"],
               "delta_pcc_drug": after["pcc_drug"] - base["pcc_drug"],
               "delta_percell_mean": after["percell_mean"] - base["percell_mean"],
               "beta_arm_ref": {"delta_pcc_sample": 0.0075, "delta_percell_mean": 0.0481,
                                "src": "results/verify_ft_20260911.json"},
               "elapsed_s": time.time() - t0},
              open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    torch.save({"net": {k: v.cpu() for k, v in best[1].items()},
                "cl_names": ck["cl_names"], "drug_vocab": ck["drug_vocab"]}, OUT_PT)
    log("saved %s + %s | total %.0fs" % (OUT_JSON, OUT_PT, time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
