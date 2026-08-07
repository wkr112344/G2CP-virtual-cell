# -*- coding: utf-8 -*-
"""eval_20pct.py —— 论文协议: 20% 化学训练, 80% 化学留出 PCC(5 细胞系)。
对比 有 ESM(遗传先验) vs 无 ESM, 算相对提升, 对照论文 +375.4%。
留出划分与 train_g2cp_full.py --frac_chem 0.2 完全一致(seed 7)。
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, 'data', 'g2cp_cache_fullgene')
from train_g2cp_contrast import G2CPNet
from unipret.compound_encoder import ECFP4_BITS

DEVICE = torch.device('cuda')
FIVE = ['HT29', 'A375', 'A549', 'MCF7', 'PC3']

m = np.load(os.path.join(CACHE, 'meta.npz'), allow_pickle=True)
kind, key, cell = m['kind'], m['key'], m['cell']
gene_vocab = [str(x) for x in m['gene_vocab']]
cl_names = [str(x) for x in m['cl_names']]
y = np.load(os.path.join(CACHE, 'y.npy'), mmap_mode='r')
fps = np.load(os.path.join(CACHE, 'drug_fps.npy'))
n_out = y.shape[1]
n = len(kind)
n_gene = len(gene_vocab)
gene_cols = json.load(open(os.path.join(CACHE, 'gene_cols.json')))
gene_mask = np.zeros(n_out, dtype=np.float32)
gene_mask[gene_cols] = 1.0

def pcc(pp, tt):
    cs = []
    for a, b in zip(pp, tt):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        cs.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(cs)) if cs else 0.0

def eval_ckpt(path, tag):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    gv = [str(x) for x in ck['gene_vocab']]
    cl = [str(x) for x in ck['cl_names']]
    emb = ck['net']['head.0.weight'].shape[0] - 32
    headw = ck['net']['head.1.weight'].shape[0]
    net = G2CPNet(len(gv), ECFP4_BITS, emb, len(cl), len(ck['hvg']), headw).to(DEVICE)
    net.load_state_dict(ck['net'], strict=False)
    net.eval()
    gv_idx = {g: i for i, g in enumerate(gv)}
    cl_idx = {c: i for i, c in enumerate(cl)}

    # 与训练一致的 20% 化学扰动划分(seed 7, frac 0.2)
    pert_uid = np.where(kind == 0, key, n_gene + key)
    rng0 = np.random.RandomState(0)
    perts = sorted(set(int(p) for p in pert_uid))
    perm = rng0.permutation(len(perts))
    tr_pert = set(int(perts[i]) for i in perm[:int(len(perts) * 0.9)])
    chem_perts = sorted(p for p in tr_pert if p >= n_gene)
    rng_c = np.random.RandomState(7)
    n_keep = int(len(chem_perts) * 0.2)
    keep = set(int(chem_perts[i]) for i in rng_c.choice(len(chem_perts), n_keep, replace=False))
    te_pert = set(p for p in chem_perts if p not in keep)  # 80% 化学留出

    te = [i for i in range(n) if kind[i] == 1 and int(pert_uid[i]) in te_pert]
    print(f'\n===== {tag} =====', flush=True)
    print(f'20% 训练化学扰动 {len(keep)}, 80% 留出化学扰动 {len(te_pert)}, 留出样本 {len(te)}', flush=True)

    per_cell = {c: [[], []] for c in FIVE}
    allp, allt = [], []
    with torch.no_grad():
        for i in te:
            ki = int(key[i])
            if ki >= len(fps):
                continue
            z = F.normalize(net.cp_lin(torch.from_numpy(fps[ki]).float().unsqueeze(0).to(DEVICE)), dim=1)
            cn = cl_names[int(cell[i])]
            ci = cl_idx.get(cn, -1)
            if ci < 0 or cn not in FIVE:
                continue
            out = net.head(torch.cat([z, net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
            t = np.asarray(y[i], dtype=np.float32)
            allp.append(out); allt.append(t)
            per_cell[cn][0].append(out); per_cell[cn][1].append(t)
    pc = pcc(allp, allt)
    print(f'总体 PCC(5系): {pc:+.4f} | 样本 {len(allp)}')
    for cn in FIVE:
        if len(per_cell[cn][0]):
            print(f'  {cn}: PCC {pcc(per_cell[cn][0], per_cell[cn][1]):+.4f} (n={len(per_cell[cn][0])})')
    return pc

pc_no = eval_ckpt('g2cp_20pct_noesm.pt', '无 ESM(无遗传先验)')
pc_esm = eval_ckpt('g2cp_20pct_esm.pt', '有 ESM(遗传先验)')

rel = (pc_esm - pc_no) / abs(pc_no) * 100 if abs(pc_no) > 1e-6 else float('nan')
print('\n' + '=' * 60)
print(f'### 20% 化学训练 PCC 考试')
print(f'无 ESM 基线:    {pc_no:+.4f}')
print(f'有 ESM(遗传先验): {pc_esm:+.4f}')
print(f'相对提升:       {rel:+.1f}%   (论文 +375.4%)')
print(f'判定:           {"✅ 超论文" if rel >= 375.4 else "❌ 未达论文"}')
