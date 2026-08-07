# -*- coding: utf-8 -*-
"""评测 G2CP 模型：留出样本 PCC（按扰动类型分）。"""
import sys, os, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_g2cp_contrast import G2CPNet, CACHE_DIR
from unipret.compound_encoder import ECFP4_BITS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="g2cp_model_v2.pt")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gene_vocab = ck["gene_vocab"]
    drug_vocab = ck["drug_vocab"]
    emb = ck["net"]["head.0.weight"].shape[0] - 32   # 从权重推断 emb
    headw = ck["net"]["head.1.weight"].shape[0]
    net = G2CPNet(len(gene_vocab), ECFP4_BITS, emb, len(ck["cl_names"]), len(ck["hvg"]), headw).to(DEVICE)
    net.load_state_dict(ck["net"])
    net.eval()
    m = np.load(os.path.join(CACHE_DIR, "meta.npz"), allow_pickle=True)
    kind, key, cell = m["kind"], m["key"], m["cell"]
    valid = None  # prep_extra 已过滤
    yv = np.load(os.path.join(CACHE_DIR, "y.npy"))
    n = len(kind)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    fps_t = torch.from_numpy(np.load(os.path.join(CACHE_DIR, "drug_fps.npy"))).to(DEVICE)
    kv_t = torch.from_numpy(key).long().to(DEVICE)
    kind_t = torch.from_numpy(kind).float().to(DEVICE)
    cell_t = torch.from_numpy(cell).long().to(DEVICE)

    def run(idx):
        pr, tr = [], []
        with torch.no_grad():
            for s in range(0, len(idx), 512):
                b = torch.from_numpy(idx[s:s + 512]).long().to(DEVICE)
                k = kv_t[b]
                k0 = kind_t[b].unsqueeze(1)
                z = net.gene_emb(torch.clamp(k, 0, len(gene_vocab) - 1)) * k0 + \
                    net.cp_lin(fps_t[torch.clamp(k, 0, len(drug_vocab) - 1)]) * (1 - k0)
                z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
                out = net.head(torch.cat([z, net.cell_emb(cell_t[b])], dim=1)).cpu().numpy()
                pr.append(out)
                tr.append(yv[idx[s:s + 512]])
        return np.concatenate(pr), np.concatenate(tr)

    def pcc_mean(pp, tt):
        cs = [np.corrcoef(pp[i], tt[i])[0, 1] for i in range(len(pp))]
        return float(np.nanmean(cs))

    te = perm[int(n * 0.9):]
    P, T = run(te)
    print(f"留出样本 {len(P)} 总 PCC: {pcc_mean(P, T):+.4f}")
    tek = kind[te]
    for nm, mk in [("基因扰动", tek == 0), ("药物扰动", tek == 1)]:
        if mk.sum() > 20:
            print(f"  {nm} ({int(mk.sum())}) PCC: {pcc_mean(P[mk], T[mk]):+.4f}")
    tr2 = perm[:2000]
    P2, T2 = run(tr2)
    print(f"训练见过样本 (2000) PCC: {pcc_mean(P2, T2):+.4f}")

if __name__ == "__main__":
    main()
