"""
MoA-SMD 评测（对标论文 Figure 2C：UniPert 1.85 vs ECFP 1.61）
================================================================
大白话：一个好的分子/基因表征，应该让"作用机制(MoA)相同的药"在向量空间里
挤在一起，机制不同的离得远。SMD（标准化均值差）就量化"类内相似 - 类间相似"
有多明显。

做法（与论文 SMD 精神一致）：
  1. MoA 分组 = 共享同一蛋白靶点的药归一组（218 药 → 17 靶点组，>=2 药的组）
  2. 取每个药的嵌入（化合物编码器 256 维，可传权重文件或随机初始化）
  3. 对每个组：类内平均余弦相似度 vs 类间平均相似度 → SMD
  4. 输出所有组 SMD 的均值（论文报 1.61→1.85）

用法：
  python unipret/eval_metrics.py                       # 随机初始化基线（对照）
  python unipret/eval_metrics.py --ckpt stageB.pt      # 用训练后的权重
  python unipret/eval_metrics.py --ckpt unipert_pretrain.pt
"""
import os
import sys
import json
import argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

LOCAL = os.path.join(BASE, "dataset.json")


def load_drug_groups(chems_path=None):
    """MoA 分组：
    - 缺省：本地 dataset.json 218 药按共享靶点分组（测未见药，反映零样本泛化）
    - 给定 --chems（sciplex3_chems.json）：按 sciPlex3 自带 target 注释分组（测训练过的药）
    返回 (drugs_list, groups{MoA: [idx]})；drugs_list 每项 {name, smiles, target}
    """
    if chems_path:
        d = json.load(open(chems_path, encoding="utf-8"))
        drugs = [{"name": n, "smiles": v["smiles"], "target": v.get("target", "")}
                 for n, v in d.items() if v.get("smiles")]
        groups = {}
        for i, dr in enumerate(drugs):
            t = dr["target"].strip()
            if t:
                groups.setdefault(t, []).append(i)
        groups = {t: idxs for t, idxs in groups.items() if len(idxs) >= 2}
        return drugs, groups
    d = json.load(open(LOCAL, encoding="utf-8"))
    drugs = [{"name": x["name"], "smiles": x.get("smiles", ""), "target": ""}
             for x in d["drugs"]]
    groups = {}
    for i, dr in enumerate(d["drugs"]):
        for t in dr.get("targets", []):
            groups.setdefault(t, []).append(i)
    groups = {t: idxs for t, idxs in groups.items() if len(idxs) >= 2}
    return drugs, groups


def smd_score(sim_mat, groups, drug_idxs):
    """对每个组算 SMD = (类内均相似 - 类间均相似) / 合并标准差。"""
    results = []
    for t, idxs in groups.items():
        idxs = [i for i in idxs if i in drug_idxs]
        if len(idxs) < 2:
            continue
        inside, outside = [], []
        for a in idxs:
            for b in idxs:
                if a < b:
                    inside.append(sim_mat[a, b])
            for b in drug_idxs:
                if b not in idxs:
                    outside.append(sim_mat[a, b])
        if not inside or not outside:
            continue
        mu_i, mu_o = np.mean(inside), np.mean(outside)
        pooled = np.sqrt((np.var(inside) + np.var(outside)) / 2) + 1e-8
        results.append((t, (mu_i - mu_o) / pooled, len(idxs)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="权重文件（stageB.pt / unipert_pretrain.pt）；缺省=随机初始化基线")
    ap.add_argument("--chems", default=None,
                    help="sciplex3_chems.json：用 sciPlex3 target 分组测训练过的药；缺省用本地 218 药")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    import torch
    from unipret.config import EMBED_DIM
    from unipret.model import UniPert
    from unipret.compound_encoder import CompoundEncoder, smiles_to_ecfp4

    drugs, groups = load_drug_groups(args.chems)

    # 构建化合物编码器（pretrain/stageB 权重里含 compound_encoder）
    enc = CompoundEncoder().to(args.device)
    if args.ckpt and os.path.isfile(args.ckpt):
        sd = torch.load(args.ckpt, map_location=args.device)
        up = sd.get("unipert", sd)
        sub = {k[len("compound_encoder."):]: v for k, v in up.items()
               if k.startswith("compound_encoder.")}
        if sub:
            enc.load_state_dict(sub, strict=False)
            print(f"    ✅ 载入 {args.ckpt} 的 compound_encoder", flush=True)
        else:
            print(f"    ⚠️ {args.ckpt} 无 compound_encoder，用随机初始化", flush=True)

    # 逐个药物编码（图网络）
    feats = []
    with torch.no_grad():
        for dr in drugs:
            g = smiles_to_ecfp4(dr.get("smiles", ""))
            if g is None:
                feats.append(None)
                continue
            z = enc([g]).cpu().numpy()[0]      # [256]
            feats.append(z / (np.linalg.norm(z) + 1e-8))
    valid = [i for i, f in enumerate(feats) if f is not None]
    F = np.stack([feats[i] for i in valid])
    sim = F @ F.T                                    # 余弦相似度
    drug_idxs = set(valid)

    results = smd_score(sim, groups, drug_idxs)
    results.sort(key=lambda r: -r[1])
    mode = "sciPlex3 target 分组（训练过的药）" if args.chems else "本地 218 药共享靶点分组（未见药）"
    print(f"\n=== MoA-SMD [ {mode} ]：{len(results)} 组，{len(valid)} 药可构图 ===")
    for t, s, n in results:
        print(f"  {t:<10} SMD={s:+.2f}  ({n} 药)")
    mean_smd = np.mean([r[1] for r in results]) if results else 0
    print(f"\n  平均 SMD = {mean_smd:.2f}   （论文：ECFP 1.61 → UniPert 1.85，+14.4%）")
    print(f"  来源: {'随机初始化基线' if not args.ckpt else args.ckpt} "
          f"{'[sciPlex3 target 分组]' if args.chems else '[本地 218 药]'}")


if __name__ == "__main__":
    main()
