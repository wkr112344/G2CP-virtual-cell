# -*- coding: utf-8 -*-
"""严格留出版 EF：剔除与训练 CPI 对重叠的 (compound, target) 正样本后重算。
审稿人意见1: EF 139 是否训练泄漏? 答案: 重叠仅 6%, 本脚本给出剔除重叠后的严格 EF。"""
import sys, os, csv, json
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import MolToSmiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
from train_g2cp_contrast import G2CPNet
from unipret.compound_encoder import ECFP4_BITS, smiles_to_ecfp4

DEVICE = torch.device('cuda')
CKPT = 'g2cp_full_cpi_v7.pt'

def canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return MolToSmiles(mol) if mol else None

# ---- 训练 CPI 对 (canon_smiles, gene) ----
cpi = np.load('data/g2cp_cache_fullgene/cpi_pairs_big.npy', allow_pickle=True)
m = np.load('data/g2cp_cache_fullgene/meta.npz', allow_pickle=True)
dv = [str(x) for x in m['drug_vocab']]
gv = [str(x) for x in m['gene_vocab']]
brd2smi = json.load(open('data/lincs_drug_smiles.json'))
train_pairs = set()
for fname in ['cpi_pairs.npy', 'cpi_pairs_big.npy']:
    for di, gi in np.load(f'data/g2cp_cache_fullgene/{fname}', allow_pickle=True):
        smi = brd2smi.get(dv[di])
        cs = canon(smi) if smi else None
        if cs:
            train_pairs.add((cs, gv[gi]))
print(f'训练 CPI 对并集(有效): {len(train_pairs)}', flush=True)

# ---- 模型 ----
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
gv_m = [str(x) for x in ck['gene_vocab']]
cl = [str(x) for x in ck['cl_names']]
emb = ck['net']['head.0.weight'].shape[0] - 32
headw = ck['net']['head.1.weight'].shape[0]
n_out = len(ck['hvg'])
net = G2CPNet(len(gv_m), ECFP4_BITS, emb, len(cl), n_out, headw).to(DEVICE)
_sd = dict(ck['net'])
if 'cp_lin.weight' in _sd and 'cp_lin.main.weight' not in _sd:
    _w, _b = _sd.pop('cp_lin.weight'), _sd.pop('cp_lin.bias')
    _sd['cp_lin.main.weight'] = _w
    _sd['cp_lin.main.bias'] = _b
net.load_state_dict(_sd, strict=False)
net.eval()
gv_idx = {g: i for i, g in enumerate(gv_m)}
print(f'模型 {CKPT} | 基因 {len(gv_m)}', flush=True)

# ---- Touchstone ----
ts_rows = list(csv.DictReader(open('data/g2cp/data/new_Touchstone.csv', encoding='utf-8')))
target2drugs = defaultdict(list)   # target gene -> [smiles]
drugs = []
for r in ts_rows:
    smi = (r['SMILES'] or '').strip()
    tg = (r['gene_targets'] or '').strip()
    if not smi or not tg:
        continue
    drugs.append((smi, (r['Name'] or '').strip()))
    for t in tg.split('|'):
        t = t.strip()
        if t in gv_idx:
            target2drugs[t].append(smi)
target2drugs = {t: v for t, v in target2drugs.items() if len(set(v)) >= 2}
print(f'靶基因(≥2药): {len(target2drugs)} | 化合物总数: {len(drugs)}', flush=True)

uniq_smis = sorted(set(s for s, _ in drugs))
z_cache = {}
with torch.no_grad():
    for s in uniq_smis:
        fp = smiles_to_ecfp4(s)
        if fp is None:
            continue
        z = F.normalize(net.cp_lin(torch.from_numpy(fp).float().unsqueeze(0).to(DEVICE)), dim=1)
        z_cache[s] = z.cpu().numpy()[0]
print(f'编码成功: {len(z_cache)}', flush=True)

def ef_at(cutoff, pos_ranks, n_total):
    n_pos = len(pos_ranks)
    if n_pos == 0:
        return float('nan')
    n_top = max(1, int(n_total * cutoff))
    hit = sum(1 for r in pos_ranks if r < n_top)
    return (hit / n_top) / (n_pos / n_total)

cuts = [0.005, 0.01, 0.05, 0.10]
def run(mode):
    efs = {c: [] for c in cuts}
    n_overlap = 0
    for t, smis in target2drugs.items():
        with torch.no_grad():
            g = F.normalize(net.gene_emb(torch.tensor([gv_idx[t]], device=DEVICE).long()), dim=1)
        g = g.cpu().numpy()[0]
        pos_set = set(smis)
        if mode == 'strict':
            keep = set()
            for s in smis:
                cs = canon(s)
                if cs and (cs, t) not in train_pairs:
                    keep.add(s)
            pos_set = keep
        n_overlap += len(set(smis)) - len(pos_set)
        scores = []
        for s, z in z_cache.items():
            scores.append((float(z @ g), s in pos_set))
        scores.sort(key=lambda x: -x[0])
        n_total = len(scores)
        pos_ranks = [i for i, (sc, is_pos) in enumerate(scores) if is_pos]
        for c in cuts:
            efs[c].append(ef_at(c, pos_ranks, n_total))
    print(f'[{mode}] 剔除重叠正样本: {n_overlap} 对')
    for c in cuts:
        me = np.nanmean(efs[c])
        print(f'  top {int(c*100)}%: EF = {me:.2f}')

print('\n=== 原始口径 (近似, 含重叠) ===')
run('orig')
print('\n=== 严格留出 (剔除与训练 CPI 重叠对) ===')
run('strict')
