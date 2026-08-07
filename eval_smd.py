# -*- coding: utf-8 -*-
"""SMD 评测 v2：标准 Cohen's d 口径（类内 vs 全部类间成对相似度）。
对齐论文：MoA 类别内药物嵌入相似度应显著高于类别间。"""
import sys, os, argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_g2cp_contrast import G2CPNet, CACHE_DIR
from unipret.compound_encoder import ECFP4_BITS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_drug_embs(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    gene_vocab = ck["gene_vocab"]
    drug_vocab = ck["drug_vocab"]
    emb = ck["net"]["head.0.weight"].shape[0] - 32
    headw = ck["net"]["head.1.weight"].shape[0]
    net = G2CPNet(len(gene_vocab), ECFP4_BITS, emb, len(ck["cl_names"]), len(ck["hvg"]), headw).to(DEVICE)
    net.load_state_dict(ck["net"], strict=False)
    net.eval()
    fps = np.load(os.path.join(CACHE_DIR, "drug_fps.npy"))
    with torch.no_grad():
        z = net.cp_lin(torch.from_numpy(fps).to(DEVICE))
        z = F.normalize(z, dim=1).cpu().numpy()
    return z, drug_vocab

def parse_pcl(path):
    pcl = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            drugs = [p for p in parts[4].split("|") if p.startswith("BRD-")]
            if len(drugs) >= 2:
                pcl[parts[0]] = drugs
    return pcl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="g2cp_model_v5.pt")
    ap.add_argument("--pcl", default="data/g2cp/data/CMAP_mmc1.txt")
    ap.add_argument("--beta", type=float, default=1.0, help="嵌入权重；<1 时与 ECFP4 Tanimoto 混合")
    ap.add_argument("--n_neg", type=int, default=0,
                    help="每类采样负样本数（论文口径：采样少量负对，>0 启用；0=全量负样本池）")
    args = ap.parse_args()
    z, drug_vocab = load_drug_embs(args.ckpt)
    vset = set(drug_vocab)
    pcl = parse_pcl(args.pcl)
    # 关键修复: 按唯一化合物去重(同一指纹/同一化合物多批次 BRD 只留一个代表)
    # → 消除重复批次污染(类间负样本若含同类化合物重复BRD, 相似度=1.0, SMD被压到0附近)
    fps_all = np.load(os.path.join(CACHE_DIR, "drug_fps.npy"))
    has_fp = fps_all.any(axis=1)
    _, uniq_first = np.unique(fps_all[has_fp], axis=0, return_index=True)
    uniq_idx = set(np.where(has_fp)[0][uniq_first].tolist())
    n_fp = len(uniq_idx)
    print(f"药物 {len(drug_vocab)} | 类别 {len(pcl)} | 唯一化合物 {n_fp} ({100*n_fp/len(drug_vocab):.1f}%)")
    # 全量相似度矩阵（嵌入余弦）
    C = z @ z.T
    if args.beta < 1.0:
        fps = fps_all
        n = len(fps)
        chunk = 2000
        T = np.zeros((n, n), dtype=np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
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
        ci = [i for i in ci if i in uniq_idx]   # 类内唯一化合物化(重复批次只留一个)
        if len(ci) < 2:
            continue
        within = S[np.ix_(ci, ci)][np.triu_indices(len(ci), 1)]
        other = [i for i in range(len(drug_vocab)) if i not in ci and i in uniq_idx]
        if args.n_neg > 0:
            # 论文口径：每类随机采样 n_neg 个异类药作负对（同类药数×负药数对）
            rng = np.random.RandomState(0)
            k = min(args.n_neg, len(other))
            oi = rng.choice(other, size=k, replace=False)
            between = S[ci][:, oi].ravel()
        else:
            between = S[ci][:, other].ravel()
            if len(between) > 20000:
                rng = np.random.RandomState(0)
                between = between[rng.choice(len(between), 20000, replace=False)]
        w, b = within.astype(np.float64), between.astype(np.float64)
        pool = np.sqrt((w.var() + b.var()) / 2)
        if pool < 1e-6:
            continue
        smds.append(float((w.mean() - b.mean()) / pool))
        covered += 1
    print(f"可评测类别 {covered} | 平均 SMD: {np.mean(smds):+.3f}（β={args.beta}；论文 UniPert=1.85 / ECFP4=1.61）")

if __name__ == "__main__":
    main()
