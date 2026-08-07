"""
CPI-EF 大池评测（P4c：扩蛋白池到对比预训练全词表 + LINCS 药物查询）
=====================================================================
蛋白池 = unipert_pretrain.pt 的全部基因（905 个，对比预训练学过嵌入）
查询   = LINCS 药（有 SMILES 且 target 在池内）
EF@top k%（池 905：0.5%≈4.5 个、1%≈9、2%≈18、5%≈45、10%≈90）

用法：python unipret/eval_ef_lincs.py --ckpt unipert_pretrain.pt
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
    ap.add_argument("--max_drugs", type=int, default=6000,
                    help="LINCS 查询药上限（有 target 的药全量 ~1.5 万，采样加速）")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    import torch
    from unipret.config import DEVICE
    from unipret.data_bridge import CompoundGraphCache
    from unipret.train_stages import _make_unipert

    def _norm(s):
        return str(s).strip().upper() if s else ""

    print(f">>> CPI-EF 大池评测：{args.ckpt}", flush=True)
    sa = torch.load(args.ckpt, map_location=DEVICE)
    gv = {_norm(k): v for k, v in sa["gene_vocab"].items()}
    unipert = _make_unipert(max(sa["gene_vocab"].values()) + 1, gene_vocab=sa["gene_vocab"],
                            gene_mode="esm").to(DEVICE)
    unipert.load_state_dict(sa["unipert"])
    unipert.eval()

    # ---- 蛋白池：全词表基因嵌入 ----
    pool_genes = sorted(gv.keys())
    pool_idx = [gv[g] for g in pool_genes]
    gid = torch.tensor(pool_idx, device=DEVICE)
    cl = torch.zeros(len(pool_idx), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        P = unipert.encode_gene(None, gid, cl).cpu().numpy()
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    n_pool = len(pool_genes)
    print(f"    蛋白池：{n_pool} 基因，嵌入 {P.shape}", flush=True)

    # ---- LINCS 药物（target ∈ 池内）----
    drugs = []
    with open(LINCS_TSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sm = (row.get("canonical_smiles") or "").strip()
            tg = (row.get("target") or "").strip()
            if sm and sm != "-" and tg and tg != "-":
                tgs = [_norm(g) for g in tg.replace(",", " ").split() if g != "-"]
                hit = [g for g in tgs if g in gv]
                if hit:
                    drugs.append((row["pert_name"], sm, hit))
    if len(drugs) > args.max_drugs:
        idx = sorted(np.random.default_rng(1).choice(
            len(drugs), args.max_drugs, replace=False).tolist())
        drugs = [drugs[i] for i in idx]
    print(f"    LINCS 查询药：{len(drugs)}（target 在池内）", flush=True)

    # 并行构图
    from unipret.compound_encoder import smiles_to_ecfp4
    from concurrent.futures import ThreadPoolExecutor
    gcache = {}
    sms = [d[1] for d in drugs]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for s, g in zip(sms, ex.map(smiles_to_ecfp4, sms)):
            if g is not None and g.any():
                gcache[s] = g
    print(f"    构图 {len(gcache)}/{len(sms)}", flush=True)

    # ---- EF 评测 ----
    ranks = []
    with torch.no_grad():
        for name, sm, truth in drugs:
            g = gcache.get(sm)
            if g is None:
                continue
            z = unipert.encode_compound([g], torch.zeros(1, dtype=torch.long, device=DEVICE))
            z = z.cpu().numpy()[0]
            z = z / (np.linalg.norm(z) + 1e-8)
            sim = P @ z
            order = np.argsort(-sim)
            truth_idx = {gv[t] for t in truth}
            # 真靶点的最高排名（池内）
            rk = min(np.where(np.isin(order, list(truth_idx)))[0]) + 1
            ranks.append(rk)
    ranks = np.array(ranks)
    print(f"    有效查询 {len(ranks)}", flush=True)
    print(f"\n=== CPI-EF 大池 [ 蛋白池 {n_pool} 基因，{len(ranks)} 药 ] ===")
    for pct in (0.5, 1.0, 2.0, 5.0, 10.0):
        k = max(1, int(round(n_pool * pct / 100)))
        hit = (ranks <= k).mean()
        exp = k / n_pool
        ef = hit / exp if exp > 0 else float("nan")
        print(f"  EF@top {pct:>4.1f}% (前{k}个) = {ef:5.2f}   Hit={hit*100:5.1f}%  "
              f"(随机期望 {exp*100:.2f}%)")
    print(f"\n  中位排名: {np.median(ranks):.0f}（池 {n_pool}，随机期望中位 {n_pool/2:.0f}）")
    print(f"  论文参考: EF 19.19/12.54/4.88/3.28 @ top 0.5%/1%/5%/10%")


if __name__ == "__main__":
    main()
