# -*- coding: utf-8 -*-
"""eval_full.py —— 全基因模型 g2cp_full_v3 的扰动级留出评测。

主缓存 305,297 样本: 按"扰动"(药物ID/基因ID) 90/10 留出。
留出的扰动在训练阶段从未出现过(主训练 + 微调的防遗忘混合都只用训练扰动)。
- 药物样本(12,328 列): 全基因 PCC
- 基因扰动样本(978 列真值): 只在 gene_cols 列算 PCC
- 按细胞系分组: 5系 / ASC / 其他
"""
import sys, os, json, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "data", "g2cp_cache_fullgene")
from train_g2cp_contrast import G2CPNet
from unipret.compound_encoder import ECFP4_BITS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "g2cp_full_v3.pt"

def pcc_mean(pp, tt):
    cs = []
    for a, b in zip(pp, tt):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        cs.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(cs)) if cs else 0.0

m = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
kind, key, cell = m["kind"], m["key"], m["cell"]
gene_vocab = [str(x) for x in m["gene_vocab"]]
drug_vocab = [str(x) for x in m["drug_vocab"]]
cl_names = [str(x) for x in m["cl_names"]]
y = np.load(os.path.join(CACHE, "y.npy"), mmap_mode="r")
fps = np.load(os.path.join(CACHE, "drug_fps.npy"))
n_out = y.shape[1]
n = len(kind)
gene_cols = json.load(open(os.path.join(CACHE, "gene_cols.json")))
gene_mask = np.zeros(n_out, dtype=np.float32)
gene_mask[gene_cols] = 1.0

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
gv = [str(x) for x in ck["gene_vocab"]]
dv = [str(x) for x in ck["drug_vocab"]]
cl = [str(x) for x in ck["cl_names"]]
hvg = list(ck["hvg"])
# 主缓存词表在模型词表中的索引
gv_idx = {g: i for i, g in enumerate(gv)}
dv_idx = {d: i for i, d in enumerate(dv)}
cl_idx = {c: i for i, c in enumerate(cl)}
emb = ck["net"]["head.0.weight"].shape[0] - 32
headw = ck["net"]["head.1.weight"].shape[0]
net = G2CPNet(len(gv), ECFP4_BITS, emb, len(cl), len(hvg), headw).to(DEVICE)
net.load_state_dict(ck["net"], strict=False)
net.eval()
print(f"模型 {CKPT}: 基因 {len(gv)}, 药 {len(dv)}, 系 {len(cl)}, 输出 {len(hvg)}", flush=True)

# 扰动级留出(与训练同 seed 同比例)
pert_uid = np.where(kind == 0, key, len(gene_vocab) + key)
rng = np.random.RandomState(0)
perts = sorted(set(int(p) for p in pert_uid))
perm = rng.permutation(len(perts))
te_pert = set(int(perts[i]) for i in perm[int(len(perts) * 0.9):])
te_idx = [i for i in range(n) if int(pert_uid[i]) in te_pert]
print(f"留出扰动 {len(te_pert)}/{len(perts)}, 留出样本 {len(te_idx)}", flush=True)

# 模型词表映射失败的样本剔除
ok = []
for i in te_idx:
    if kind[i] == 0:
        g = gene_vocab[int(key[i])]
        if g in gv_idx:
            ok.append((i, 0, gv_idx[g], cl_idx.get(cl_names[int(cell[i])], -1)))
    else:
        d = drug_vocab[int(key[i])]
        if d in dv_idx:
            ok.append((i, 1, dv_idx[d], cl_idx.get(cl_names[int(cell[i])], -1)))
print(f"可评测样本: {len(ok)}/{len(te_idx)}", flush=True)

pr_all, tr_all = [], []
pr_drug, tr_drug = [], []
pr_gene, tr_gene = [], []
per_cell = defaultdict(lambda: [[], []])
with torch.no_grad():
    for i, kd, ki, ci in ok:
        if ci < 0:
            continue
        if kd == 0:
            z = F.normalize(net.gene_emb(torch.tensor([ki], device=DEVICE).long()), dim=1)
        else:
            z = F.normalize(net.cp_lin(torch.from_numpy(fps[int(key[i])]).float().unsqueeze(0).to(DEVICE)), dim=1)
        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
        t = np.asarray(y[i], dtype=np.float32)
        if kd == 0:
            # 基因扰动只评 978 列
            gsel = gene_mask > 0
            pr_gene.append(out[gsel]); tr_gene.append(t[gsel])
        else:
            pr_drug.append(out); tr_drug.append(t)
            pr_all.append(out); tr_all.append(t)
        cn = cl_names[int(cell[i])]
        per_cell[cn][0].append(out); per_cell[cn][1].append(t)

print("\n" + "=" * 60)
print(f"### 扰动级留出 PCC (全基因模型 {os.path.basename(CKPT)})")
print(f"药物扰动 (n={len(pr_drug)}):  PCC = {pcc_mean(pr_drug, tr_drug):+.4f}")
print(f"基因扰动 (n={len(pr_gene)}):  PCC = {pcc_mean(pr_gene, tr_gene):+.4f} (978列)")
print("\n### 按细胞系(药物样本)")
for cn in ["HT29", "A375", "A549", "MCF7", "PC3", "ASC", "VCAP", "HA1E", "NPC", "NEU", "MDAMB231", "PHH"]:
    if cn in per_cell and len(per_cell[cn][0]) >= 20:
        pc = pcc_mean(per_cell[cn][0], per_cell[cn][1])
        print(f"  {cn:<10} n={len(per_cell[cn][0]):>5}  PCC = {pc:+.4f}")
print("\n### 对照: 训练见过的扰动(记忆/插值能力)")
tr_pert_set = set(int(p) for p in perts) - te_pert
tr_seen = [i for i in range(n) if int(pert_uid[i]) in tr_pert_set]
rng2 = np.random.RandomState(1)
tr_seen_s = rng2.choice(tr_seen, size=min(8000, len(tr_seen)), replace=False)
pr_s, tr_s = [], []
for i in tr_seen_s:
    if kind[i] == 0:
        g = gene_vocab[int(key[i])]
        if g not in gv_idx:
            continue
        ki = gv_idx[g]
        with torch.no_grad():
            z = F.normalize(net.gene_emb(torch.tensor([ki], device=DEVICE).long()), dim=1)
    else:
        d = drug_vocab[int(key[i])]
        if d not in dv_idx:
            continue
        ki = dv_idx[d]
        with torch.no_grad():
            z = F.normalize(net.cp_lin(torch.from_numpy(fps[int(key[i])]).float().unsqueeze(0).to(DEVICE)), dim=1)
    ci = cl_idx.get(cl_names[int(cell[i])], -1)
    if ci < 0:
        continue
    with torch.no_grad():
        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
    t = np.asarray(y[i], dtype=np.float32)
    if kind[i] == 0:
        gsel = gene_mask > 0
        pr_s.append(out[gsel]); tr_s.append(t[gsel])
    else:
        pr_s.append(out); tr_s.append(t)
print(f"训练见过扰动样本 (n={len(pr_s)}): PCC = {pcc_mean(pr_s, tr_s):+.4f}")

print("\n### 参考")
print("  论文 G2CP: 20% 化学训练下 PCC +375.4% 提升(基线 ~0.2 量级)")
print("  说明: PCC 是'预测表达 vs 真实测量'的相关系数, 0.3+ = 中强相关, 0.5+ = 强")
