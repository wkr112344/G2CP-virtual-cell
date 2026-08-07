# -*- coding: utf-8 -*-
"""eval_fusion.py —— 融合模型双指标评测：留出 PCC（分 5系/新系）+ SMD（论文口径）。"""
import sys, os, argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
FUSION = os.path.join(BASE, "data", "g2cp_cache_fusion")
from train_g2cp_contrast import G2CPNet
from unipret.compound_encoder import ECFP4_BITS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIVE = ["HT29", "A375", "A549", "MCF7", "PC3"]


def load_net(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    gene_vocab = [str(x) for x in ck["gene_vocab"]]
    drug_vocab = [str(x) for x in ck["drug_vocab"]]
    cl_names = [str(x) for x in ck["cl_names"]]
    emb = ck["net"]["head.0.weight"].shape[0] - 32
    headw = ck["net"]["head.1.weight"].shape[0]
    net = G2CPNet(len(gene_vocab), ECFP4_BITS, emb, len(cl_names), len(ck["hvg"]), headw).to(DEVICE)
    net.load_state_dict(ck["net"], strict=False)
    net.eval()
    return net, gene_vocab, drug_vocab, cl_names


def pcc_mean(pp, tt):
    cs = [np.corrcoef(pp[i], tt[i])[0, 1] for i in range(len(pp))]
    return float(np.nanmean(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="g2cp_fusion.pt")
    ap.add_argument("--pcl", default="data/g2cp/data/CMAP_mmc1.txt")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n_neg", type=int, default=0)
    args = ap.parse_args()

    net, gene_vocab, drug_vocab, cl_names = load_net(args.ckpt)
    m = np.load(os.path.join(FUSION, "meta.npz"), allow_pickle=True)
    kind, key, cell = m["kind"], m["key"], m["cell"]
    yv = np.load(os.path.join(FUSION, "y.npy"))
    fps = np.load(os.path.join(FUSION, "drug_fps.npy"))
    n = len(kind)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    fps_t = torch.from_numpy(fps).to(DEVICE)
    kv_t = torch.from_numpy(key).long().to(DEVICE)
    kind_t = torch.from_numpy(kind).float().to(DEVICE)
    cell_t = torch.from_numpy(cell).long().to(DEVICE)
    cl_names_l = list(cl_names)

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

    te = perm[int(n * 0.9):]
    P, T = run(te)
    print(f"\n===== PCC 评测（留出 {len(P)} 样本）=====")
    print(f"总 PCC: {pcc_mean(P, T):+.4f}")
    tek = kind[te]
    for nm, mk in [("基因扰动", tek == 0), ("药物扰动", tek == 1)]:
        if mk.sum() > 20:
            print(f"  {nm} ({int(mk.sum())}) PCC: {pcc_mean(P[mk], T[mk]):+.4f}")
    tec = cell[te]
    m5 = np.isin(tec, [cl_names_l.index(c) for c in FIVE if c in cl_names_l])
    if m5.sum() > 20:
        print(f"  5系 ({int(m5.sum())}) PCC: {pcc_mean(P[m5], T[m5]):+.4f}")
    mnew = (~m5) & (tek == 1)
    if mnew.sum() > 20:
        print(f"  新系药物 ({int(mnew.sum())}) PCC: {pcc_mean(P[mnew], T[mnew]):+.4f}")
    P2, T2 = run(perm[:2000])
    print(f"训练见过样本 (2000) PCC: {pcc_mean(P2, T2):+.4f}")

    # ===== SMD =====
    print(f"\n===== SMD 评测 =====")
    with torch.no_grad():
        z = net.cp_lin(torch.from_numpy(fps).to(DEVICE))
        z = F.normalize(z, dim=1).cpu().numpy()
    vset = set(drug_vocab)
    pcl = {}
    with open(args.pcl, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            drugs = [p for p in parts[4].split("|") if p.startswith("BRD-")]
            if len(drugs) >= 2:
                pcl[parts[0]] = drugs
    C = z @ z.T
    if args.beta < 1.0:
        nf = len(fps)
        T = np.zeros((nf, nf), dtype=np.float32)
        for s in range(0, nf, 2000):
            e = min(s + 2000, nf)
            A = fps[s:e] @ fps.T
            c1 = fps[s:e].sum(1)
            den = c1[:, None] + fps.sum(1)[None, :] - A
            T[s:e] = np.divide(A, den, out=np.zeros_like(A), where=den != 0)
        np.fill_diagonal(T, -1)
        S = args.beta * C + (1 - args.beta) * T
    else:
        S = C
    np.fill_diagonal(S, -1)
    smds, covered = [], 0
    for cid, drugs in pcl.items():
        ci = [drug_vocab.index(d) for d in drugs if d in vset]
        if len(ci) < 2:
            continue
        within = S[np.ix_(ci, ci)][np.triu_indices(len(ci), 1)]
        other = [i for i in range(len(drug_vocab)) if i not in ci]
        if args.n_neg > 0:
            rng2 = np.random.RandomState(0)
            k = min(args.n_neg, len(other))
            oi = rng2.choice(other, size=k, replace=False)
            between = S[ci][:, oi].ravel()
        else:
            between = S[ci][:, other].ravel()
            if len(between) > 20000:
                rng2 = np.random.RandomState(0)
                between = between[rng2.choice(len(between), 20000, replace=False)]
        w, b = within.astype(np.float64), between.astype(np.float64)
        pool = np.sqrt((w.var() + b.var()) / 2)
        if pool < 1e-6:
            continue
        smds.append(float((w.mean() - b.mean()) / pool))
        covered += 1
    print(f"可评测类别 {covered} | 平均 SMD: {np.mean(smds):+.3f}（β={args.beta}；论文 UniPert=1.85 / ECFP4=1.61）")


if __name__ == "__main__":
    main()
