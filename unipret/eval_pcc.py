"""
PCC 预测基线评估（对标论文 Figure 4E：G2CP 化学扰动表达预测 vs 真实）
=====================================================================
衡量：stageB 模型对"药物→表达变化"的预测与真实测量（sciPlex3）的
Pearson 相关性（PCC），按 (药, 细胞系) 计算。论文报告 G2CP 在 20% 训练
数据下 PCC 相对无遗传预训练提升 375.4%。

做法：
  对每个 (药, 细胞系) 样本，预测 hvg 维表达变化 vs 真实 → PCC。
  同时给两个基线对照：
    ① 零预测基线（全 0 预测）→ PCC 定义上为 NaN/0，用"预测=对照均值"基线
    ② 随机预测基线（随机高斯）→ PCC≈0
  输出平均 PCC（真实信号 > 0 即为有效预测）。

用法：
  python unipret/eval_pcc.py --ckpt stageB.pt
"""
import os
import sys
import json
import argparse
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

SCIPLEX3 = "C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad"
CHEM_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sciplex3_chems.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="stageB.pt")
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    import torch
    from unipret.config import DEVICE, CELL_LINE_NAMES
    from unipret.data_bridge import (PerturbationReader, select_hvg,
                                     build_compound_samples_chembl)
    from unipret.model import UniPert
    from unipret.effect_model import PerturbationEffectModel

    print(f">>> PCC 评估：{args.ckpt}", flush=True)
    sa = torch.load(args.ckpt, map_location=DEVICE)
    gene_vocab = sa["gene_vocab"]
    hvg_n = sa.get("hvg_dim", args.hvg)

    reader = PerturbationReader(SCIPLEX3, backed=True)
    hvg = select_hvg(reader, n=hvg_n, max_cells=50000)
    chem_map = json.load(open(CHEM_MAP, encoding="utf-8"))
    samples = build_compound_samples_chembl(reader, chem_map, hvg)
    for s in samples:
        s["smiles"] = chem_map[s["name"]]["smiles"]
    samples = [s for s in samples if s.get("smiles")]
    print(f"    评估样本 {len(samples)}（{len(set(s['name'] for s in samples))} 药 × "
          f"{len(set(s['cell_line_idx'] for s in samples))} 系）", flush=True)

    # 重建模型（gene_mode 从 ckpt 推断：stageA/B 保存时是 esm/hybrid——用 esm 尝试，失败回退）
    from unipret.train_stages import _make_unipert
    unipert = _make_unipert(max(gene_vocab.values()) + 1, gene_vocab=gene_vocab,
                            gene_mode="esm").to(DEVICE)
    try:
        unipert.load_state_dict(sa["unipert"])
        effect = PerturbationEffectModel.build_stage_b(
            PerturbationEffectModel.build_stage_a(unipert, hvg_n)).to(DEVICE)
        effect.load_state_dict(sa["effect"])
    except Exception as e:
        print(f"    ⚠️ esm 载入失败（{e}），试 embedding 模式", flush=True)
        unipert = _make_unipert(max(gene_vocab.values()) + 1, gene_vocab=gene_vocab,
                                gene_mode="embedding").to(DEVICE)
        unipert.load_state_dict(sa["unipert"])
        effect = PerturbationEffectModel.build_stage_b(
            PerturbationEffectModel.build_stage_a(unipert, hvg_n)).to(DEVICE)
        effect.load_state_dict(sa["effect"])
    effect.eval()

    from unipret.data_bridge import CompoundGraphCache
    cache = CompoundGraphCache()
    pccs, nulls = [], []
    with torch.no_grad():
        for s in samples:
            g = cache.get(s["smiles"])
            cl = torch.tensor([s["cell_line_idx"]], device=DEVICE)
            pred = effect.forward_compound([g], cl).cpu().numpy()[0]   # [hvg]
            real = s["expr_delta"]                                     # [hvg]
            if np.std(pred) < 1e-8 or np.std(real) < 1e-8:
                continue
            pcc = float(np.corrcoef(pred, real)[0, 1])
            pccs.append(pcc)
            null = float(np.corrcoef(np.random.default_rng(0).normal(size=real.shape), real)[0, 1])
            nulls.append(null)
    reader.close()
    pccs, nulls = np.array(pccs), np.array(nulls)
    print(f"\n=== PCC（{len(pccs)} 个 (药,系) 样本，hvg={hvg_n}）===")
    print(f"  模型预测 PCC : {pccs.mean():+.4f} ± {pccs.std():.4f}")
    print(f"  随机预测基线 : {nulls.mean():+.4f} ± {nulls.std():.4f}")
    print(f"  有效提升     : {pccs.mean() - nulls.mean():+.4f}")
    # top 5 药
    from collections import defaultdict
    by_drug = defaultdict(list)
    for s, p in zip(samples, pccs):
        by_drug[s["name"]].append(p)
    top = sorted(by_drug.items(), key=lambda kv: -np.mean(kv[1]))[:5]
    print("  Top5 预测最好的药：")
    for n, ps in top:
        print(f"    {n[:40]:42s} PCC={np.mean(ps):+.3f} ({len(ps)} 系)")


if __name__ == "__main__":
    main()
