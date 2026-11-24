"""step55: 【论文②增补实验 E-数据源敏感性】受控对照设计。

目的：把"数据源效应"与"覆盖度效应"分离。
控制变量：同一批细胞系 C*、同一批药物 D*、同一套 held-out 药物划分（RandomState(0) 后 10%，
           perts 取自 C*×D* 的共同药物集，两源共用同一个 H）。
只变数据源：
    A = GEO GSE92742 Level5 MODZ（cache_rebuilt，signature 级 → 聚合成对级）
    B = dcic 2021 cd-coefficient（cache_2021，对级）
统计：per-drug 的 PCC 差 -> 配对 bootstrap 给 ΔPCC 的 95% CI（审稿人要的严谨度）。
对照臂（非受控，仅参考）：B 在其全量 125 系 / 32,039 药上的 0.1539。
输出：results/stage55_source_sensitivity.json
"""
import os, sys, json, time, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# 项目根：脚本既可直接放在项目根，也可放在 code/ 子目录下（本归档的布局）
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = _HERE if os.path.isdir(os.path.join(_HERE, "results")) else os.path.dirname(_HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(BASE, "assets", "g2cp_full_cpi_v7.pt"),
                help="checkpoint 路径（原始 v7 键名或域内适配键名均可）")
ap.add_argument("--out", default="stage55_source_sensitivity.json",
                help="输出 json 文件名（存到 results/ 下）")
args = ap.parse_args()
DRB = os.path.join(BASE, "data", "g2cp_cache_rebuilt")
D21 = os.path.join(BASE, "data", "g2cp_cache_2021")
RES = os.path.join(BASE, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOT = 2000
TOPK = [1, 5, 10, 20, 50]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


t0 = time.time()
log("ckpt = %s" % args.ckpt)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
sd = ck["net"]
# 兼容两套键名：原始 v7 = cp_lin.weight / cell_emb.weight；域内适配 = cp.weight / cell.weight
K_CP_W = "cp.weight" if "cp.weight" in sd else "cp_lin.weight"
K_CP_B = "cp.bias" if "cp.bias" in sd else "cp_lin.bias"
K_CELL = "cell.weight" if "cell.weight" in sd else "cell_emb.weight"
G = len(ck["hvg"]) if "hvg" in ck else sd["head.3.weight"].shape[0]
NGENE = sd["gene_emb.weight"].shape[0] if "gene_emb.weight" in sd else None
NCELL = sd[K_CELL].shape[0]
log("keys: %s / %s | G=%d NCELL=%d" % (K_CP_W, K_CELL, G, NCELL))
fps = np.load(os.path.join(D21, "drug_fps.npy"))
# 药物指纹用 v7 drug_vocab 索引；cell 用 v7 cl_names 索引
v162 = [str(x) for x in ck["cl_names"]]
drugv = [str(x) for x in ck["drug_vocab"]]
cpos = {c: i for i, c in enumerate(v162)}
dpos = {d: i for i, d in enumerate(drugv)}

# ---------- B: 2021 ----------
pm = np.load(os.path.join(D21, "pairs_meta.npz"), allow_pickle=True)
b_cell = np.array([str(x) for x in pm["cell_name"]], dtype=object)
b_drug = np.array([str(x) for x in pm["drug_id"]], dtype=object)
log("2021 pairs=%d cells=%d drugs=%d" %
    (len(b_cell), len(set(b_cell)), len(set(b_drug))))

# ---------- A: GEO rebuilt ----------
mb = np.load(os.path.join(DRB, "meta.npz"), allow_pickle=True)
a_key = mb["key"].astype(np.int64)
a_cell = mb["cell"].astype(np.int64)
a_cl = np.array([str(x) for x in mb["cl_names"]], dtype=object)
a_dv = np.array([str(x) for x in mb["drug_vocab"]], dtype=object)
log("GEO  rows=%d cl_names=%d drug_vocab=%d" % (len(a_key), len(a_cl), len(a_dv)))

# ---------- 共同集 ----------
a_cellname = a_cell
a_cn = np.array([a_cl[int(c)] for c in a_cell], dtype=object)
a_dn = np.array([a_dv[int(k)] for k in a_key], dtype=object)
cellsA = set(a_cn); cellsB = set(b_cell)
drugsA = set(a_dn); drugsB = set(b_drug)
Cstar = sorted(cellsA & cellsB)
Dstar = sorted(drugsA & drugsB)
log("共同细胞系 C*=%d (GEO %d / 2021 %d) | 共同药物 D*=%d (GEO %d / 2021 %d)"
    % (len(Cstar), len(cellsA), len(cellsB), len(Dstar), len(drugsA), len(drugsB)))

# held-out 药物：两源共用，perts = sorted(D*)
perts = sorted(Dstar)
n_te = int(len(perts) * 0.9)
perm = np.random.RandomState(0).permutation(len(perts))
H = set(perts[i] for i in perm[n_te:])
log("held-out 药物 = %d / %d" % (len(H), len(perts)))

# ---------- 目标 pair 集 P*：两源都有、cell∈C*、drug∈H ----------
b_pairs = set(zip(b_cell.tolist(), b_drug.tolist()))
a_pairs = set(zip(a_cn.tolist(), a_dn.tolist()))
Pstar = sorted((p for p in (a_pairs & b_pairs)
                if p[0] in set(Cstar) and p[1] in H))
log("受控评测 pair 集 P* = %d" % len(Pstar))
pidx = {p: i for i, p in enumerate(Pstar)}
nP = len(Pstar)
assert nP > 500, "P* 太小，无法评测"

# ---------- 聚合 A（GEO，signature 级 -> 对级）----------
Yb = np.load(os.path.join(DRB, "y.npy"), mmap_mode="r")
TA = np.zeros((nP, G), dtype=np.float32)
cntA = np.zeros(nP, dtype=np.float32)
CH = 20000
hit = 0
for s in range(0, len(a_key), CH):
    e = min(s + CH, len(a_key))
    X = np.asarray(Yb[s:e], dtype=np.float32)
    for j in range(e - s):
        i = pidx.get((a_cn[s + j], a_dn[s + j]), -1)
        if i >= 0:
            TA[i] += X[j]
            cntA[i] += 1.0
            hit += 1
    del X
    if (s // CH) % 3 == 0:
        log("  GEO 扫描 %d/%d (%.0fs)" % (e, len(a_key), time.time() - t0))
TA /= np.maximum(cntA, 1)[:, None]
log("GEO 聚合完成：命中 %d 行 -> %d 对 (%.0fs)" % (hit, nP, time.time() - t0))
del Yb

# ---------- B（2021，已是对级）----------
Y21 = np.load(os.path.join(D21, "y.npy"), mmap_mode="r")
selB = [i for i in range(len(b_cell))
        if (b_cell[i], b_drug[i]) in pidx]
order = np.argsort(selB)
TB = np.zeros((nP, G), dtype=np.float32)
Bs = np.asarray(Y21[selB], dtype=np.float32)   # 已按 pair 顺序
for k, i in enumerate(selB):
    TB[pidx[(b_cell[i], b_drug[i])]] = Bs[k]
del Y21, Bs
log("2021 取对完成：%d 对" % len(selB))

# ---------- 模型预测 ----------
cell_idx = np.array([cpos[p[0]] for p in Pstar], dtype=np.int64)
drug_idx = np.array([dpos[p[1]] for p in Pstar], dtype=np.int64)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cp = nn.Linear(2048, 512)
        self.cell = nn.Embedding(NCELL, 32)
        self.head = nn.Sequential(nn.LayerNorm(544), nn.Linear(544, 1024), nn.GELU(),
                                  nn.Linear(1024, G))
        own = self.state_dict()
        own["cp.weight"] = sd[K_CP_W]
        own["cp.bias"] = sd[K_CP_B]
        own["cell.weight"] = sd[K_CELL]
        for k in ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias",
                  "head.3.weight", "head.3.bias"]:
            own[k] = sd[k]
        self.load_state_dict(own)

    def forward(self, fp, c):
        z = F.normalize(self.cp(fp), dim=-1)
        return self.head(torch.cat([z, self.cell(c)], 1))


net = Net().to(DEVICE).eval()
with torch.no_grad():
    preds = []
    for s in range(0, nP, 2048):
        preds.append(net(torch.from_numpy(fps[drug_idx[s:s + 2048]]).to(DEVICE),
                         torch.from_numpy(cell_idx[s:s + 2048]).to(DEVICE))
                     .cpu().numpy().astype(np.float32))
P = np.concatenate(preds, 0)
log("预测完成 %s" % (P.shape,))


def metrics(P, T):
    res = {k: [] for k in ["pcc", "dir"] + ["d%d" % k for k in TOPK]}
    for s in range(0, len(P), 2048):
        p = P[s:s + 2048]; t = T[s:s + 2048]
        a = p - p.mean(1, keepdims=True); b = t - t.mean(1, keepdims=True)
        res["pcc"].append((a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-8))
        ok = (np.sign(p) == np.sign(t))
        res["dir"].append(ok.mean(1))
        for k in TOPK:
            m_ = max(1, int(round(G * k / 100.0)))
            idx = np.argpartition(-np.abs(t), m_ - 1, axis=1)[:, :m_]
            res["d%d" % k].append(np.take_along_axis(ok, idx, 1).mean(1))
    return {k: np.concatenate(v, 0) for k, v in res.items()}


MA = metrics(P, TA)
MB = metrics(P, TB)
udrug = np.unique(drug_idx)
groups = [np.where(drug_idx == g)[0] for g in udrug]


def boot(v):
    um = np.array([np.nanmean(v[g]) for g in groups])
    return um


def ci(um, n=N_BOOT):
    brng = np.random.default_rng(12345)
    b = np.empty(n)
    for i in range(n):
        b[i] = np.nanmean(um[brng.integers(0, len(um), len(um))])
    return float(np.nanmean(um)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


out = {"design": {"Cstar_cells": len(Cstar), "Dstar_drugs": len(Dstar),
                  "heldout_drugs": len(H), "eval_pairs": int(nP),
                  "ckpt": os.path.basename(args.ckpt),
                  "split": "RandomState(0) 后10%药物（两源共用同一 H）"}}
for tag, M in [("A_GEO_MODZ", MA), ("B_2021_coeff", MB)]:
    o = {}
    for nm in ["pcc", "dir", "d1", "d5", "d10", "d20", "d50"]:
        o[nm] = ci(boot(M[nm]))
    out[tag] = o
    log("%-14s PCC=%+.4f [%.4f,%.4f]  dir=%.2f%%  top5=%.2f%%"
        % (tag, o["pcc"][0], o["pcc"][1], o["pcc"][2], o["dir"][0] * 100, o["d5"][0] * 100))

# 配对 bootstrap：per-drug 差
umA = boot(MA["pcc"]); umB = boot(MB["pcc"])
diff = umA - umB
dm, dl, dh = ci(diff)
umA_dir = boot(MA["dir"]); umB_dir = boot(MB["dir"])
ddm, ddl, ddh = ci(umA_dir - umB_dir)
umA5 = boot(MA["d5"]); umB5 = boot(MB["d5"])
d5m, d5l, d5h = ci(umA5 - umB5)
out["paired_diff_A_minus_B"] = {
    "pcc": [dm, dl, dh], "dir": [ddm, ddl, ddh], "d5": [d5m, d5l, d5h],
    "n_drugs": int(len(diff))}
log("配对差 (A−B): ΔPCC=%+.4f [%+.4f,%+.4f]  Δdir=%+.2fpt  Δtop5=%+.2fpt"
    % (dm, dl, dh, ddm * 100, d5m * 100))

print()
print("=" * 100)
print("增补实验 E · 数据源敏感性（受控：同细胞系 %d / 同药物 %d / 同划分 / 模型 %s）"
      % (len(Cstar), len(Dstar), os.path.basename(args.ckpt)))
print("=" * 100)
print("  评测 (药×系) 对 = %d,  held-out 药物 = %d" % (nP, len(H)))
print("  %-16s %-24s %-12s %-12s" % ("数据源", "PCC [95%CI]", "direction", "top-5%"))
for tag, label in [("A_GEO_MODZ", "A GEO GSE92742 MODZ"),
                   ("B_2021_coeff", "B dcic2021 cd-coefficient")]:
    o = out[tag]
    print("  %-16s %+.4f [%.4f,%.4f]   %.2f%%       %.2f%%"
          % (label, o["pcc"][0], o["pcc"][1], o["pcc"][2],
             o["dir"][0] * 100, o["d5"][0] * 100))
print()
print("  配对差 A−B（按药物聚类 bootstrap, n=%d 药物）:" % len(diff))
print("    ΔPCC  = %+.4f  [%+.4f, %+.4f]   %s"
      % (dm, dl, dh, "显著" if dl * dh > 0 else "不显著"))
print("    Δdir  = %+.2f pt [%+.2f, %+.2f]" % (ddm * 100, ddl * 100, ddh * 100))
print("    Δtop5 = %+.2f pt [%+.2f, %+.2f]" % (d5m * 100, d5l * 100, d5h * 100))
print()
print("  参考（非受控，覆盖度不同）：")
print("    A 全量 70 系 / 261,397       PCC=0.2825  dir=59.65%  top5=73.32%")
print("    B 全量 125 系 / 156,931      PCC=0.1539  dir=55.26%  top5=61.95%")
print("    论文 Table1（143 系）        PCC=0.3047  dir=60.80%  top5=73.80%")
out_path = args.out if os.path.isabs(args.out) else os.path.join(RES, args.out)
json.dump(out, open(out_path, "w"), indent=1)
log("saved %s | %.0fs" % (out_path, time.time() - t0))
