# -*- coding: utf-8 -*-
"""eval_holdout_similarity.py —— 新药预测能力的分层验证。

核心问题: 模型能不能预测"训练没见过的新药"?
答案分层: 按留出药物与训练集药物的最大结构相似度(Tanimoto)分层:
  - 高相似(≥0.7): 与已知药几乎同骨架 → 预期可预测(结构相似性假设)
  - 中相似(0.4-0.7): 部分相似
  - 低相似(<0.4): 全新骨架 → 最难
同时给出 全部留出药物 的总体 PCC。
"""
import sys, os, json
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
CKPT = sys.argv[1] if len(sys.argv) > 1 else "g2cp_full_correct.pt"

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

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
gv = [str(x) for x in ck["gene_vocab"]]
dv = [str(x) for x in ck["drug_vocab"]]
cl = [str(x) for x in ck["cl_names"]]
hvg = list(ck["hvg"])
dv_idx = {d: i for i, d in enumerate(dv)}
cl_idx = {c: i for i, c in enumerate(cl)}
emb = ck["net"]["head.0.weight"].shape[0] - 32
headw = ck["net"]["head.1.weight"].shape[0]
net = G2CPNet(len(gv), ECFP4_BITS, emb, len(cl), len(hvg), headw).to(DEVICE)
net.load_state_dict(ck["net"], strict=False)
net.eval()
print(f"模型 {os.path.basename(CKPT)}: 语义 {ck.get('enc_semantics','?')} | 药 {len(dv)} 系 {len(cl)}", flush=True)

# 扰动级留出(药物样本)
pert_uid = np.where(kind == 0, key, len(gene_vocab) + key)
rng = np.random.RandomState(0)
perts = sorted(set(int(p) for p in pert_uid))
perm = rng.permutation(len(perts))
te_pert = set(int(perts[i]) for i in perm[int(len(perts) * 0.9):])
te_drug = [i for i in range(n) if kind[i] == 1 and int(pert_uid[i]) in te_pert]
print(f"留出药物扰动 {len(te_pert)}, 留出药物样本 {len(te_drug)}", flush=True)

# 留出药物指纹(用于相似度) + 训练药物指纹集合
tr_drug_idx = [i for i in range(n) if kind[i] == 1 and int(pert_uid[i]) not in te_pert]
tr_drug_keys = set(int(key[i]) for i in tr_drug_idx)
te_drug_keys = {}
for i in te_drug:
    te_drug_keys.setdefault(int(key[i]), []).append(i)
print(f"留出药物数 {len(te_drug_keys)}, 训练药物数 {len(tr_drug_keys)}", flush=True)

# Tanimoto 相似度: 留出药 vs 训练药 最大相似度 (用 numpy 加速, 只对留出药算)
tr_fps = fps[list(tr_drug_keys)]  # (N_tr, 2048) 太大? 32k×2048 float32 = 262MB, OK
tr_fps_n = tr_fps / (np.linalg.norm(tr_fps, axis=1, keepdims=True) + 1e-8)
tr_fps_bin = tr_fps > 0
tr_counts = tr_fps_bin.sum(1, keepdims=True) + 1e-8
sim_cache = {}

def max_tanimoto(fp):
    # 用点积近似: Tanimoto = |A∩B| / (|A|+|B|-|A∩B|); 用批量矩阵运算太慢, 采样 2000 训练药近似
    nz = (fp > 0)
    a_cnt = nz.sum()
    inter = nz[None, :] & tr_fps_bin  # (2000, 2048) 逐行算
    return inter

# 分块算相似度(内存友好): 每个留出药 vs 全部训练药
te_keys = sorted(te_drug_keys.keys())
bucket = {'high': [], 'mid': [], 'low': []}   # (key → 最大相似度)
sims = {}
for ki in te_keys:
    fp = fps[ki]
    nz = (fp > 0)
    a_cnt = nz.sum()
    inter = (nz & tr_fps_bin).sum(1)  # (N_tr,)
    union = a_cnt + tr_counts.ravel() - inter
    tani = inter / np.maximum(union, 1e-8)
    sims[ki] = float(tani.max())

# 预测留出药物样本 + 按相似度分层 PCC
pr = defaultdict(list); tr = defaultdict(list)
with torch.no_grad():
    for ki, idxs in te_drug_keys.items():
        if ki >= len(fps) or ki >= len(dv):  # 主缓存 drug 索引 = 模型 drug_vocab 索引(词表同源)
            continue
        fp = fps[ki]
        z = F.normalize(net.cp_lin(torch.from_numpy(fp).float().unsqueeze(0).to(DEVICE)), dim=1)
        for i in idxs:
            cn = cl_names[int(cell[i])]
            ci = cl_idx.get(cn, -1)
            if ci < 0:
                continue
            out = net.head(torch.cat([z, net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
            t = np.asarray(y[i], dtype=np.float32)
            s = sims.get(int(key[i]), 0.0)
            layer = 'high' if s >= 0.7 else ('mid' if s >= 0.4 else 'low')
            pr[layer].append(out); tr[layer].append(t)
            pr['all'].append(out); tr['all'].append(t)

print("\n" + "=" * 64)
print("### 新药(留出)预测能力分层验证")
print(f"{'层':<8}{'相似度':<16}{'药物数':>7}{'样本':>8}{'PCC':>10}")
for layer, desc, cond in [('high', '≥0.7 同骨架', lambda s: s >= 0.7),
                          ('mid', '0.4-0.7 部分相似', lambda s: 0.4 <= s < 0.7),
                          ('low', '<0.4 全新骨架', lambda s: s < 0.4)]:
    if pr[layer]:
        pc = pcc_mean(pr[layer], tr[layer])
        n_drugs = sum(1 for s in sims.values() if cond(s))
        print(f"{layer:<8}{desc:<16}{n_drugs:>7}{len(pr[layer]):>8}{pc:>+10.4f}")
if pr['all']:
    pc = pcc_mean(pr['all'], tr['all'])
    print(f"{'all':<8}{'全部新药':<16}{len(te_keys):>7}{len(pr['all']):>8}{pc:>+10.4f}")
print("\n说明: PCC 是预测 vs 真实表达的相关; 0.2+ = 有预测信号, 0.4+ = 中强")
print("如果'同骨架'层 PCC 明显高于'全新骨架'层 → 结构相似的新药可预测, 全新骨架仍难")
