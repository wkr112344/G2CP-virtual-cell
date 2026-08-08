# -*- coding: utf-8 -*-
"""EF 评测：药物统一空间嵌入的 MoA 富集因子（对齐论文 EF@top0.5/1/5/10%）。
用 CMAP_mmc1 的 PCL 类别：同类药应互相排在最前面。"""
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
    _sd = dict(ck["net"])
    if "cp_lin.weight" in _sd and "cp_lin.main.weight" not in _sd:
        _w, _b = _sd.pop("cp_lin.weight"), _sd.pop("cp_lin.bias")
        _sd["cp_lin.main.weight"] = _w
        _sd["cp_lin.main.bias"] = _b
    net.load_state_dict(_sd, strict=False)
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
    args = ap.parse_args()
    z, drug_vocab = load_drug_embs(args.ckpt)
    dv_idx = {d: i for i, d in enumerate(drug_vocab)}
    vset = set(drug_vocab)
    pcl = parse_pcl(args.pcl)
    # 每个药的类别（一个药可能多类，取第一个用于评测去重）
    drug_cls = {}
    for cid, drugs in pcl.items():
        for d in drugs:
            if d in vset and d not in drug_cls:
                drug_cls[d] = cid
    print(f"药物 {len(drug_vocab)} | 类别 {len(pcl)} | 有类别标签药 {len(drug_cls)}")
    S = z @ z.T  # 相似度矩阵
    np.fill_diagonal(S, -1)
    for top in [0.005, 0.01, 0.05, 0.10]:
        k = max(1, int(len(drug_vocab) * top))
        hits, total_q = 0, 0
        for q, cid in drug_cls.items():
            qi = dv_idx[q]
            order = np.argsort(-S[qi])[:k]
            for j in order:
                if j < len(drug_vocab) and drug_vocab[j] in drug_cls and drug_cls[drug_vocab[j]] == cid:
                    hits += 1
                    break  # 每 query 只计 1 个命中
            total_q += 1
        base_rate = max(1, len(drug_cls) / len(drug_vocab) * 100)
        hit_rate = hits / max(total_q, 1)
        ef = hit_rate / (len(drug_cls) / max(len(drug_vocab), 1))
        print(f"EF@top{top*100:g}%: {ef:.2f}（命中 {hits}/{total_q}，随机期望 1.0）")
    print("论文 EF@top0.5/1/5/10% = 19.19/12.54/4.88/3.28")

if __name__ == "__main__":
    main()
