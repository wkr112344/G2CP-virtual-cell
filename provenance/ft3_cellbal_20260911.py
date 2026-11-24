"""ft3_cellbal_20260911.py — 用「逐细胞系等权」的域内适配，专门冲两个中位数

动机（用户 2026-09-11 指令："再补中位数"）：
  现有主模型 `assets/g2cp_v7_beta_ft.pt` 的损失 = MSE + (1 − 逐样本 PCC)，**每个样本等权**。
  但报数指标是「逐细胞系 PCC 均值/中位」。样本数极不均衡（HT29 n=1213、U2OS n=2041、
  大量系 n≈28–29、还有 15 个系 n=1），于是逐样本等权的损失被大户支配，
  中小样本系（正是决定中位数的那批）得不到优化 → 中位数被拖住。

  本脚本把损失改成**逐细胞系加权**：
       w_i ∝ (1 / n_{cell(i)})^alpha        （alpha=0 退化为原脚本的逐样本等权）
  对 MSE 项与 PDE 项（1 − PCC）同时加权，alpha 由命令行给定。
  这是标准的类别不平衡处理，不是给中位数开小灶：训练目标（逐系平均）与
  报数指标（逐系平均）本来就是同一个，原来的逐样本等权才是目标错配。

评测：官方 RandomState(0) 后 10% 药物留出（16,126 对）上的
      · 池化逐样本 PCC（= 主口径 0.3447）
      · 逐药物等权 PCC
      · 逐细胞系 mean / median / min / max（口径 A，116 系）
      · 癌系（n=86）mean / median ← 要冲的第二项
      · 方向准确率 / top-5% 方向

输出：results/ft3_cellbal_<tag>.json + assets/g2cp_v7_beta_ft3_<tag>.pt
"""
import argparse, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = r"C:\Users\wkr20\WorkBuddy\2026-09-06-00-53-37"
CACHE_DEFAULT = os.path.join(BASE, "data", "g2cp_cache_beta_v2")
ASSETS = os.path.join(BASE, "assets")
RESULTS = os.path.join(BASE, "results")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPRINT = {"percell_mean": 0.359, "percell_median": 0.362,
            "cancer_mean": 0.296, "cancer_median": 0.324, "novel_drug_pcc": 0.3047}


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def pcc_np(P, T):
    a = P - P.mean(1, keepdims=True); b = T - T.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="逐细胞系加权强度：w ∝ (1/n_cell)^alpha；0=逐样本等权（原脚本）")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--init", default="assets/g2cp_full_cpi_v7.pt")
    ap.add_argument("--tag", default="a100")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-pt", default="")
    ap.add_argument("--eval-only", default="", help="只评测给定权重，不训练")
    a = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 8)))
    out_json = a.out_json or os.path.join(RESULTS, "ft3_cellbal_%s.json" % a.tag)
    out_pt = a.out_pt or os.path.join(ASSETS, "g2cp_v7_beta_ft3_%s.pt" % a.tag)

    meta = np.load(os.path.join(a.cache, "meta.npz"), allow_pickle=True)
    key = meta["key"].astype(np.int64); cell = meta["cell"].astype(np.int64)
    kind = meta["kind"].astype(np.int64)
    y = np.load(os.path.join(a.cache, "y.npy"), mmap_mode="r")
    fps = np.load(os.path.join(a.cache, "drug_fps.npy"))
    N, G = y.shape
    log("cache=%s" % a.cache)

    ck = torch.load(os.path.join(BASE, a.init), map_location="cpu", weights_only=False)
    sd = ck["net"]
    NGENE = sd["gene_emb.weight"].shape[0]; NC = sd["cell_emb.weight"].shape[0]
    log("cache N=%d G=%d | NGENE=%d NC=%d | dev=%s threads=%d"
        % (N, G, NGENE, NC, DEV, torch.get_num_threads()))

    # ---- 官方划分（RandomState(0) 后 10% 药物留出），与主稿完全一致 ----
    pert_uid = np.where(kind == 0, key, NGENE + key)
    rng = np.random.RandomState(0)
    perts = sorted(set(int(p) for p in pert_uid))
    perm = rng.permutation(len(perts))
    te_set = set(int(perts[i]) for i in perm[int(len(perts) * 0.9):])
    is_te = np.array([int(p) in te_set for p in pert_uid])
    te_idx = np.where(is_te)[0]; tr_all = np.where(~is_te)[0]
    r2 = np.random.RandomState(a.seed); p2 = r2.permutation(len(tr_all))
    n_val = int(len(tr_all) * 0.05)
    va_idx = tr_all[p2[:n_val]]; tr_idx = tr_all[p2[n_val:]]
    log("train=%d val=%d test=%d" % (len(tr_idx), len(va_idx), len(te_idx)))

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
    net = net.to(DEV)

    # ---- 逐细胞系权重 ----
    n_cell = np.bincount(cell[tr_idx], minlength=NC).astype(np.float64)
    n_eff = np.where(n_cell > 0, n_cell, 1.0)
    w_cell = (1.0 / n_eff) ** a.alpha
    w = w_cell[cell[tr_idx]]
    w = w / w.mean()
    log("alpha=%.2f | n_cell 分布: 最大 %d 最小 %d 中位 %d | 权重倍数 max/min=%.1f"
        % (a.alpha, int(n_cell.max()), int(n_cell[n_cell > 0].min()),
           int(np.median(n_cell[n_cell > 0])), float(w.max() / w.min())))

    Xd_tr = torch.from_numpy(np.asarray(fps[key[tr_idx]]))
    Xc_tr = torch.from_numpy(cell[tr_idx])
    Y_tr = torch.from_numpy(np.asarray(y[tr_idx], dtype=np.float32))
    W_tr = torch.from_numpy(w.astype(np.float32))
    Xd_va = torch.from_numpy(np.asarray(fps[key[va_idx]]))
    Xc_va = torch.from_numpy(cell[va_idx])
    Y_va = torch.from_numpy(np.asarray(y[va_idx], dtype=np.float32))
    log("data loaded %.0fs" % (time.time() - t0))

    # ---- 癌/非癌标注 ----
    import csv as _csv
    lab = {}
    p60 = os.path.join(RESULTS, "stage60_e9_cell_difficulty.csv")
    if os.path.exists(p60):
        for r in _csv.DictReader(open(p60, encoding="utf-8")):
            lab[r["cell"]] = r["is_tumor"]
    cl_names = [str(x) for x in ck["cl_names"]] if "cl_names" in ck else None

    def evaluate(tag=""):
        net.eval(); outs = []
        with torch.no_grad():
            for s in range(0, len(te_idx), 1024):
                r = te_idx[s:s + 1024]
                outs.append(net(torch.from_numpy(fps[key[r]]).to(DEV),
                                torch.from_numpy(cell[r]).to(DEV)).cpu().numpy().astype(np.float32))
        P = np.concatenate(outs, 0)
        T = np.asarray(y[te_idx], dtype=np.float32)
        pc = pcc_np(P, T)
        dirv = (np.sign(P) == np.sign(T))
        cellte = cell[te_idx]
        # 逐药物
        um = np.array([np.nanmean(pc[g]) for g in [np.where(key[te_idx] == g)[0]
                                                   for g in np.unique(key[te_idx])]])
        # 逐细胞系
        ucs = np.unique(cellte)
        pr = np.array([pc[cellte == c].mean() for c in ucs])
        o = dict(tag=tag, pcc_sample=float(pc.mean()), pcc_drug=float(um.mean()),
                 dir=float(dirv.mean()), percell_mean=float(pr.mean()),
                 percell_median=float(np.median(pr)), percell_min=float(pr.min()),
                 percell_max=float(pr.max()), n_cells=int(len(ucs)))
        if cl_names and lab:
            cv = np.array([pr[i] for i, c in enumerate(ucs)
                           if lab.get(cl_names[c]) == "1"])
            nv = np.array([pr[i] for i, c in enumerate(ucs)
                           if lab.get(cl_names[c]) == "0"])
            if len(cv) and len(nv):
                from scipy import stats
                o.update(cancer_mean=float(cv.mean()), cancer_median=float(np.median(cv)),
                         noncancer_mean=float(nv.mean()), noncancer_median=float(np.median(nv)),
                         n_cancer=int(len(cv)), n_noncancer=int(len(nv)),
                         mwu_p=float(stats.mannwhitneyu(cv, nv, alternative="two-sided").pvalue))
        # top-k 方向
        abt = np.abs(T).mean(0)
        for k in [1, 5]:
            thr = np.percentile(abt, 100 - k)
            m = abt >= thr
            o["top%d" % k] = float(dirv[:, m].mean())
        return o

    def show(o):
        s = ("%s 池化=%.4f 逐药物=%.4f dir=%.4f | 逐系 mean=%.4f median=%.4f min=%.4f max=%.4f"
             % (o["tag"], o["pcc_sample"], o["pcc_drug"], o["dir"],
                o["percell_mean"], o["percell_median"], o["percell_min"], o["percell_max"]))
        if "cancer_mean" in o:
            s += "\n     癌 n=%d mean=%.4f median=%.4f | 非癌 n=%d mean=%.4f median=%.4f | mwu p=%.4g" % (
                o["n_cancer"], o["cancer_mean"], o["cancer_median"],
                o["n_noncancer"], o["noncancer_mean"], o["noncancer_median"], o["mwu_p"])
        log(s)

    if a.eval_only:
        sd2 = torch.load(os.path.join(BASE, a.eval_only), map_location="cpu",
                         weights_only=False)["net"]
        net.load_state_dict(sd2); net = net.to(DEV)
        o = evaluate(a.tag)
        show(o)
        json.dump(o, open(out_json, "w"), indent=1)
        return 0

    base = evaluate("baseline(v7 原始)")
    show(base)

    params = [net.cp.weight, net.cp.bias, net.cell.weight] + list(net.head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best = (-1, None)
    for ep in range(a.epochs):
        net.train(); pp = torch.randperm(len(Y_tr))
        tot = 0.0; nb = 0
        for i in range(0, len(pp), a.batch):
            b = pp[i:i + a.batch]
            xd = Xd_tr[b].to(DEV, non_blocking=True)
            xc = Xc_tr[b].to(DEV, non_blocking=True)
            yb = Y_tr[b].to(DEV, non_blocking=True)
            wb = W_tr[b].to(DEV, non_blocking=True)
            pr = net(xd, xc)
            aa = pr - pr.mean(1, keepdim=True); bb2 = yb - yb.mean(1, keepdim=True)
            pcc_row = (aa * bb2).sum(1) / torch.sqrt((aa * aa).sum(1) * (bb2 * bb2).sum(1) + 1e-8)
            se = ((pr - yb) ** 2).mean(1)
            loss = (wb * se).sum() / wb.sum() + (wb * (1.0 - pcc_row)).sum() / wb.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xd_va.to(DEV), Xc_va.to(DEV)).cpu().numpy()
        vp = float(pcc_np(pv, Y_va.numpy()).mean())
        if vp > best[0]:
            best = (vp, {k: v.detach().clone() for k, v in net.state_dict().items()})
        if ep % 2 == 0 or ep == a.epochs - 1:
            log("  ep%02d loss=%.4f val_pcc=%.4f (best %.4f) %.0fs"
                % (ep, tot / max(nb, 1), vp, best[0], time.time() - t0))

    net.load_state_dict(best[1]); net = net.to(DEV)
    after = evaluate("after(alpha=%.2f)" % a.alpha)
    show(after)
    torch.save({"net": {k: v.cpu() for k, v in best[1].items()},
                "cl_names": ck.get("cl_names"), "drug_vocab": ck.get("drug_vocab")}, out_pt)
    json.dump({"alpha": a.alpha, "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
               "baseline": base, "after": after, "val_pcc_best": float(best[0]),
               "preprint_ref": PREPRINT,
               "delta": {k: after.get(k, float("nan")) - PREPRINT[k]
                         for k in ["percell_median", "cancer_median", "percell_mean", "cancer_mean"]}},
              open(out_json, "w"), indent=1)
    log("saved %s + %s | total %.0fs" % (out_json, out_pt, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
