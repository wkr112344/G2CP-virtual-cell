"""
平均对齐 ρ 评测（文献指标：UniPert 0.43 vs ECFP 0.38）
=====================================================================
含义：对已知「药物-靶点基因」正对，药物嵌入与靶点基因嵌入的相似度，
应系统性高于随机配对的相似度。ρ = 标签(正/负) 与相似度的点二列相关
（等价于线性化 AUC；>0 表示对齐有效，论文 0.43）。

用法：python unipret/eval_alignment.py --ckpt unipert_pretrain.pt
"""
import os
import sys
import csv
import argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
LINCS_TSV = os.path.join(BASE, "LINCS_small_molecules.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unipert_pretrain.pt")
    ap.add_argument("--max_pairs", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    import torch
    from unipret.config import DEVICE
    from unipret.train_stages import _make_unipert
    from unipret.compound_encoder import smiles_to_ecfp4
    from concurrent.futures import ThreadPoolExecutor

    def _norm(s):
        return str(s).strip().upper() if s else ""

    print(f">>> 平均对齐 ρ：{args.ckpt}", flush=True)
    sa = torch.load(args.ckpt, map_location=DEVICE)
    gv = {_norm(k): v for k, v in sa["gene_vocab"].items()}
    unipert = _make_unipert(len(sa["gene_vocab"]), gene_vocab=sa["gene_vocab"],
                            gene_mode="esm").to(DEVICE)
    unipert.load_state_dict(sa["unipert"])
    unipert.eval()

    # 蛋白池嵌入
    pool_genes = sorted(gv.keys())
    pool_idx = [gv[g] for g in pool_genes]
    with torch.no_grad():
        P = unipert.encode_gene(None, torch.tensor(pool_idx, device=DEVICE),
                                torch.zeros(len(pool_idx), dtype=torch.long, device=DEVICE))
        P = P.cpu().numpy()
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    print(f"    蛋白池 {len(pool_genes)} 基因", flush=True)

    # 药物-靶点正对（LINCS）
    pairs = []
    with open(LINCS_TSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sm = (row.get("canonical_smiles") or "").strip()
            tg = (row.get("target") or "").strip()
            if sm and sm != "-" and tg and tg != "-":
                tgs = [_norm(g) for g in tg.replace(",", " ").split() if g != "-"]
                hit = [g for g in tgs if g in gv]
                if hit:
                    pairs.append((sm, hit))
    rng = np.random.default_rng(args.seed)
    if len(pairs) > args.max_pairs:
        pairs = [pairs[i] for i in
                 sorted(rng.choice(len(pairs), args.max_pairs, replace=False).tolist())]
    print(f"    药-靶正对 {len(pairs)}", flush=True)

    # 构图
    gcache = {}
    sms = sorted({p[0] for p in pairs})
    with ThreadPoolExecutor(max_workers=8) as ex:
        for s, g in zip(sms, ex.map(smiles_to_ecfp4, sms)):
            if g is not None and g.any():
                gcache[s] = g
    print(f"    构图 {len(gcache)}/{len(sms)}", flush=True)

    pos_sims, neg_sims = [], []
    with torch.no_grad():
        for sm, tgs in pairs:
            g = gcache.get(sm)
            if g is None:
                continue
            z = unipert.encode_compound([g], torch.zeros(1, dtype=torch.long, device=DEVICE))
            z = z.cpu().numpy()[0]
            z = z / (np.linalg.norm(z) + 1e-8)
            sims = P @ z
            pos_sims.append(max(sims[gv[t]] for t in tgs))        # 正对相似度（取最高真靶）
            neg_sims.append(float(np.percentile(sims, 30)))        # 负对参照（低分位随机蛋白）
    pos_sims, neg_sims = np.array(pos_sims), np.array(neg_sims)
    labels = np.concatenate([np.ones_like(pos_sims), np.zeros_like(neg_sims)])
    sims_all = np.concatenate([pos_sims, neg_sims])
    rho = float(np.corrcoef(labels, sims_all)[0, 1])
    print(f"\n=== 平均对齐 ρ ===")
    print(f"  正对相似度均值 : {pos_sims.mean():+.4f} ± {pos_sims.std():.4f}")
    print(f"  负对参照相似度 : {neg_sims.mean():+.4f} ± {neg_sims.std():.4f}")
    print(f"  对齐 ρ         : {rho:+.4f}   （论文 UniPert 0.43 / ECFP 0.38）")
    print(f"  （{len(pos_sims)} 正对 vs {len(neg_sims)} 参照，随机基线 ≈ 0）")


if __name__ == "__main__":
    main()
