# -*- coding: utf-8 -*-
"""对比用户分子 vs celecoxib 的 MAP7 完整预测值。"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_g2cp_contrast import G2CPNet, CACHE_DIR
from unipret.compound_encoder import smiles_to_ecfp4, ECFP4_BITS

ck = torch.load('g2cp_v10.pt', map_location='cpu', weights_only=False)
hvg = list(ck['hvg'])
emb = ck['net']['head.0.weight'].shape[0] - 32
headw = ck['net']['head.1.weight'].shape[0]
net = G2CPNet(len(ck['gene_vocab']), ECFP4_BITS, emb, len(ck['cl_names']), len(hvg), headw)
net.load_state_dict(ck['net'], strict=False)
net.eval()
fps = np.load(os.path.join(CACHE_DIR, 'drug_fps.npy'))
net._fps = torch.from_numpy(fps)
drug_vocab = [str(x) for x in ck['drug_vocab']]

import anndata as ad
a = ad.read_h5ad('data/g2cp/data/LINCS/pool/pool_gene_chem_ctrl_adata.h5ad', backed='r')
sub = a.obs[a.obs['pert_type'] == 'trt_cp'][['pert_id', 'cmap_name']].drop_duplicates()
nm2id = {str(r['cmap_name']).lower(): str(r['pert_id']) for _, r in sub.iterrows()}
a.file.close()
print('celecoxib 在训练集:', 'celecoxib' in nm2id)

def pred_map7(inp, cell=0):
    with torch.no_grad():
        if inp in drug_vocab:
            fp = fps[drug_vocab.index(inp)]
        else:
            fp = smiles_to_ecfp4(inp)
            if fp is None:
                return None
        z = F.normalize(net.cp_lin(torch.from_numpy(np.asarray(fp, dtype=np.float32)).unsqueeze(0)), dim=1)
        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([cell]).long())], dim=1))[0].numpy()
    if 'MAP7' in hvg:
        i = hvg.index('MAP7')
        return float(out[i]), int(np.argmax(out)) if False else None
    return None

u = pred_map7('Cc1ccc(S(=O)(=O)Nc2cc(C(F)(F)F)ccc2OC)cc1')
cel_id = nm2id.get('celecoxib')
c = pred_map7(cel_id) if cel_id else None
print(f'你的分子 MAP7 预测值: {round(u[0], 3) if u else "MAP7 不在 978 HVG"}')
print(f'celecoxib MAP7 预测值: {round(c[0], 3) if c else "无/不在"}')
print(f'MAP7 在 978 HVG: {"MAP7" in hvg}')
