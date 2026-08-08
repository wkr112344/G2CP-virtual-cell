# -*- coding: utf-8 -*-
"""_eval_dir_acc.py —— 方向准确率评测: 预测上调/下调 vs 真实方向的一致性。
按基因真实效应强度分层: 强效应(|y|大)基因的方向准确率才是真实使用场景。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from train_g2cp_contrast import G2CPNet, CACHE_DIR
from unipret.compound_encoder import ECFP4_BITS

CKPT = os.path.join(BASE, "g2cp_full_cpi_v7.pt")
DEVICE = torch.device("cuda")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
gv = [str(x) for x in ck["gene_vocab"]]
dv = [str(x) for x in ck["drug_vocab"]]
cl = [str(x) for x in ck["cl_names"]]
hvg = list(ck["hvg"])
n_out = len(hvg)
emb = ck["net"]["head.0.weight"].shape[0] - 32
headw = ck["net"]["head.1.weight"].shape[0]

net = G2CPNet(len(gv), ECFP4_BITS, emb, len(cl), n_out, headw)
net.load_state_dict(ck["net"], strict=False)
with torch.no_grad():
    net.cp_lin.main.weight.data.copy_(ck["net"]["cp_lin.weight"])
    net.cp_lin.main.bias.data.copy_(ck["net"]["cp_lin.bias"])
    net.cp_lin.res[2].weight.data.zero_(); net.cp_lin.res[2].bias.data.zero_()
net.to(DEVICE).eval()

CACHE = os.path.join(BASE, "data", "g2cp_cache_fullgene")
m = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
kind, key, cell = m["kind"], m["key"], m["cell"]
y = np.load(os.path.join(CACHE, "y.npy"), mmap_mode="r")
fps = np.load(os.path.join(CACHE, "drug_fps.npy"))
cl_names = [str(x) for x in m["cl_names"]]
gene_cols = json.load(open(os.path.join(CACHE, "gene_cols.json")))
gene_mask = np.zeros(y.shape[1], dtype=np.float32)
gene_mask[gene_cols] = 1.0

# 同 seed 扰动级留出
n = len(kind)
pert_uid = np.where(kind == 0, key, len(gv) + key)
perts = sorted(set(int(p) for p in pert_uid))
rng = np.random.RandomState(0)
perm = rng.permutation(len(perts))
te_pert = set(int(perts[i]) for i in perm[: int(len(perts) * 0.1)])
te_idx = [i for i in range(n) if int(pert_uid[i]) in te_pert]
print(f"留出样本 {len(te_idx)}", flush=True)

def top_genes(out_t, t_t, layer=None, name=""):
    mask = t_t != 0
    pr, tr = out_t[mask], t_t[mask]
    if len(pr) < 50:
        return
    # 分层: 全部 / |t| 前25% / 前10% / 前5%
    mag = np.abs(tr)
    for label, k in [("全部", 1.0), ("强效应前25%", 0.25), ("强效应前10%", 0.10), ("强效应前5%", 0.05)]:
        nk = max(50, int(len(tr) * k))
        keep = np.argsort(-mag)[:nk]
        p2, t2 = pr[keep], tr[keep]
        acc = float((np.sign(p2) == np.sign(t2)).mean())
        pcc = float(np.corrcoef(p2, t2)[0, 1])
        print(f"  [{name}] {label}: 方向准确率 {acc*100:.1f}% | PCC {pcc:.3f} (n={len(t2)})", flush=True)

dir_acc_all = []
layer_bins = {"全部": [], "强效应前25%": [], "强效应前10%": [], "强效应前5%": []}
for i in te_idx:
    k, c = int(key[i]), int(cell[i])
    if kind[i] == 1:
        z = F.normalize(net.cp_lin(torch.from_numpy(fps[k]).float().unsqueeze(0).to(DEVICE)), dim=1)
    else:
        z = net.gene_emb(torch.tensor([k]).to(DEVICE))
    ce = net.cell_emb(torch.tensor([c]).to(DEVICE))
    with torch.no_grad():
        out = net.head(torch.cat([z, ce], dim=1))[0].cpu().numpy()
    t = np.asarray(y[i], dtype=np.float32)
    if kind[i] == 0:
        mask = gene_mask > 0
        p, r = out[mask], t[mask]
    else:
        mask = t != 0
        p, r = out[mask], t[mask]
    if len(p) > 50 and r.std() > 0:
        correct = (np.sign(p) == np.sign(r)).astype(float)
        dir_acc_all.append(correct.mean())
        mag = np.abs(r)
        n = len(r)
        for label, kk in [("强效应前25%", 0.25), ("强效应前10%", 0.10), ("强效应前5%", 0.05)]:
            keep = np.argsort(-mag)[: max(50, int(n * kk))]
            layer_bins[label].append(correct[keep].mean())
    if len(dir_acc_all) >= 2000:
        break

print(f"\n### 留出扰动方向准确率 (n={len(dir_acc_all)} 样本)")
print(f"总体方向准确率(全基因): {np.mean(dir_acc_all)*100:.1f}% (随机=50%)")
for label in ["强效应前25%", "强效应前10%", "强效应前5%"]:
    print(f"  {label}: {np.mean(layer_bins[label])*100:.1f}%")
