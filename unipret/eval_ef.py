"""
CPI-EF 评测（第三个基准：药-靶检索富集）
=====================================================================
设定（受限于可编码基因，采用本地高可靠蛋白池）：
  蛋白池 = dataset.json 的 20 个基因（有 ESM 序列语义嵌入，质量最高）
  查询   = 本地 218 药（targets 标注为真实靶点）
  对每个药：药物嵌入 vs 全部 20 个蛋白嵌入 → 余弦相似度排序 →
          真实靶点的排名 → 命中率 / 富集因子 EF@top k%
  对照   = 随机基线（EF 期望 1）

用法：python unipret/eval_ef.py --ckpt stageB.pt
"""
import os
import sys
import argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
LOCAL = os.path.join(BASE, "dataset.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="stageB.pt")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    import torch
    from unipret.config import DEVICE
    from unipret.data_bridge import load_local_dataset, CompoundGraphCache
    from unipret.model import UniPert
    from unipret.train_stages import _make_unipert

    print(f">>> CPI-EF 评测：{args.ckpt}", flush=True)
    sa = torch.load(args.ckpt, map_location=DEVICE)
    gene_vocab = sa["gene_vocab"]
    local = load_local_dataset(LOCAL)
    genes = local["genes"]                       # 20 个蛋白池基因
    drugs = local["drugs"]                       # 218 个药物

    # 模型（esm 模式；基因编码器对本地基因有 ESM 语义特征；num_genes 与 ckpt 一致）
    unipert = _make_unipert(len(sa["gene_vocab"]), gene_vocab=gene_vocab,
                            gene_mode="esm").to(DEVICE)
    try:
        unipert.load_state_dict(sa["unipert"])
        print("    载入 ckpt 编码器", flush=True)
    except Exception as e:
        print(f"    ⚠️ unipert 载入失败（{e}），用随机初始化", flush=True)

    # 蛋白池嵌入（本地基因；用 ckpt 的新词表索引）
    cache = CompoundGraphCache()
    def _norm(s):
        return str(s).strip().upper() if s else ""
    gv = {_norm(k): v for k, v in sa["gene_vocab"].items()}
    pool_genes = [g for g in genes if _norm(g) in gv]
    pool_idx = [gv[_norm(g)] for g in pool_genes]
    if not pool_idx:
        print("    !! 蛋白池基因不在 ckpt 词表，退出", flush=True)
        return
    gid = torch.tensor(pool_idx, device=DEVICE)
    cl = torch.zeros(len(pool_idx), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        P = unipert.encode_gene(None, gid, cl).cpu().numpy()   # [N_pool, 256]
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    print(f"    蛋白池：{len(pool_genes)} 基因，嵌入 {P.shape}", flush=True)

    # 药物查询
    ranks_all, hits = [], []
    with torch.no_grad():
        for dr in drugs:
            sm = dr.get("smiles", "")
            g = cache.get(sm)
            if g is None:
                continue
            z = unipert.encode_compound([g], torch.zeros(1, dtype=torch.long, device=DEVICE))
            z = z.cpu().numpy()[0]
            z = z / (np.linalg.norm(z) + 1e-8)
            sim = P @ z                                        # [N_pool]
            order = np.argsort(-sim)                           # 相似度降序
            truth = {gv[_norm(t)] for t in dr.get("targets", []) if _norm(t) in gv}
            if not truth:
                continue
            rk = min(int(np.where(np.isin(order, list(truth)))[0][0]) + 1 for _ in [0])
            ranks_all.append(rk)
            hits.append((dr["name"], rk, len(truth)))
    ranks = np.array(ranks_all)
    n = len(pool_genes)
    print(f"    有效药物查询 {len(ranks)}（有 SMILES 且有真实靶点在池内）", flush=True)

    # 指标：Hit@k（真实靶点排进前 k） + EF@top k%（密度比随机期望）
    print(f"\n=== CPI-EF [ 蛋白池 {n} 基因，{len(ranks)} 药 ] ===")
    for k in (1, 2, 3, 5, 10):
        hit_rate = (ranks <= k).mean()
        exp = k / n
        ef = hit_rate / exp if exp > 0 else float("nan")
        print(f"  Hit@{k:<2d}={hit_rate*100:5.1f}%   EF@top {k}={ef:5.2f}   "
              f"(随机期望 {exp*100:.0f}%)")
    print(f"\n  中位排名: {np.median(ranks):.0f}（池 {n}，随机期望中位 {n/2:.0f}）")
    top = sorted(hits, key=lambda x: x[1])[:8]
    print("  真实靶点排最前的药：")
    for name, rk, ntr in top:
        print(f"    {name[:36]:38s} 真靶排名 #{rk}（池 {n}，{ntr} 靶）")


if __name__ == "__main__":
    main()
