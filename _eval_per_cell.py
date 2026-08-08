import os, sys, json, torch, numpy as np
import torch.nn.functional as Fn
from collections import defaultdict
sys.path.insert(0, ".")
from train_g2cp_contrast import G2CPNet, CACHE_DIR
from unipret.compound_encoder import ECFP4_BITS

DEV = torch.device("cuda")
CACHE = os.path.join(".", "data", "g2cp_cache_fullgene")
m = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
kind, key, cell = m["kind"], m["key"], m["cell"]
y = np.load(os.path.join(CACHE, "y.npy"), mmap_mode="r")
fps = np.load(os.path.join(CACHE, "drug_fps.npy"))
cl_names = [str(x) for x in m["cl_names"]]

ck = torch.load("g2cp_full_cpi_v7.pt", map_location="cpu", weights_only=False)
gv = [str(x) for x in ck["gene_vocab"]]
net = G2CPNet(len(gv), ECFP4_BITS, 512, len(ck["cl_names"]), len(ck["hvg"]), 1024)
net.load_state_dict(ck["net"], strict=False)
with torch.no_grad():
    net.cp_lin.main.weight.data.copy_(ck["net"]["cp_lin.weight"])
    net.cp_lin.main.bias.data.copy_(ck["net"]["cp_lin.bias"])
    net.cp_lin.res[2].weight.data.zero_()
    net.cp_lin.res[2].bias.data.zero_()
net.to(DEV).eval()

n = len(kind)
pert_uid = np.where(kind == 0, key, len(gv) + key)
perts = sorted(set(int(p) for p in pert_uid))
rng = np.random.RandomState(0)
perm = rng.permutation(len(perts))
te_pert = set(int(perts[i]) for i in perm[: int(len(perts) * 0.1)])

cl_pp = defaultdict(list)
cl_tt = defaultdict(list)
for i in range(n):
    if kind[i] != 1 or int(pert_uid[i]) not in te_pert:
        continue
    cn = cl_names[int(cell[i])]
    k = int(key[i])
    c = int(cell[i])
    z = Fn.normalize(net.cp_lin(torch.from_numpy(fps[k]).float().unsqueeze(0).to(DEV)), dim=1)
    with torch.no_grad():
        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([c]).to(DEV))], dim=1))[0].cpu().numpy()
    t = np.asarray(y[i], dtype=np.float32)
    msk = t != 0
    cl_pp[cn].append(out[msk])
    cl_tt[cn].append(t[msk])
    if i % 5000 == 0:
        print(f"  {i}/{n}", flush=True)

print("\n=== v7 单细胞系药物扰动 PCC (训练域留出) ===")
pccs = []
for cn in sorted(cl_pp.keys()):
    P = np.concatenate(cl_pp[cn])
    T = np.concatenate(cl_tt[cn])
    if len(P) > 100:
        pcc = np.corrcoef(P, T)[0, 1]
        pccs.append(pcc)

print(f"细胞系数: {len(pccs)} | 平均: {np.mean(pccs):.3f} | 中位: {np.median(pccs):.3f} | 范围: {min(pccs):.3f}~{max(pccs):.3f}")
print(f"\n论文 sciPlex3 UniPert (n_top=all): A549=0.981, K562=0.973, MCF7=0.986")
print(f"  注意: 论文字分数来自top-DEG且模型在sciPlex3上训练；我们的数字是全基因+扰动级完全留出")
for tgt in ["A549", "MCF7"]:
    if tgt in cl_pp:
        P = np.concatenate(cl_pp[tgt])
        T = np.concatenate(cl_tt[tgt])
        pcc = np.corrcoef(P, T)[0, 1]
        print(f"  {tgt}: PCC={pcc:.3f}")
