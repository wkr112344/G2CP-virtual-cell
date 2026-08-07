"""
残差学习版阶段 B（P4d：追平平均响应基线）
=====================================================================
动机：P3-A 留出评估发现模型未见药 PCC 0.587 < 平均响应基线 0.666。
      原因：模型把力气花在学"共性响应"上，而这部分被"猜平均"白捡。
方案（残差学习）：
  训练目标 = real − mean_delta[细胞系]   （药物特异性偏差）
  评测     = mean_delta[细胞系] + 模型预测  vs  real
  这样模型被迫学"特异性"，评测时平均响应由基线提供，二者叠加。

流程：stageA(esm) 迁移 → sciPlex3 136 药 × 3 系样本 → 按药 80/20 留出
      → 训练残差 → 未见药 PCC vs 平均基线 vs 随机基线。

用法：python unipret/train_stageb_residual.py --stageA stageA.pt --epochs 60
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

SCIPLEX3 = "C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad"
CHEM_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sciplex3_chems.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageA", default="stageA.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from unipret.config import DEVICE, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, WEIGHT_DECAY
    from unipret.data_bridge import (PerturbationReader, select_hvg,
                                     build_compound_samples_chembl, CompoundGraphCache)
    from unipret.train_stages import CompoundPertDataset, _train_loop, _make_unipert
    from unipret.effect_model import PerturbationEffectModel

    t0 = __import__("time").time()
    print(f">>> 残差学习版阶段 B（{args.epochs} epoch，留出 {args.holdout}）", flush=True)
    sa = torch.load(args.stageA, map_location=DEVICE)
    gene_vocab = sa["gene_vocab"]
    hvg_n = sa.get("hvg_dim", args.hvg)

    reader = PerturbationReader(SCIPLEX3, backed=True)
    hvg = select_hvg(reader, n=hvg_n, max_cells=50000)
    chem_map = json.load(open(CHEM_MAP, encoding="utf-8"))
    all_samples = build_compound_samples_chembl(reader, chem_map, hvg)
    for s in all_samples:
        s["smiles"] = chem_map[s["name"]]["smiles"]
    all_samples = [s for s in all_samples if s.get("smiles")]
    reader.close()
    print(f"    总样本 {len(all_samples)}（{len(set(s['name'] for s in all_samples))} 药）", flush=True)

    # 按药 80/20
    names = sorted({s["name"] for s in all_samples})
    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(len(names) * args.holdout))
    test_names = set(rng.choice(names, n_test, replace=False).tolist())
    train_samples = [s for s in all_samples if s["name"] not in test_names]
    test_samples = [s for s in all_samples if s["name"] in test_names]
    print(f"    训练 {len(train_samples)} 样本 / 测试 {len(test_samples)} 样本", flush=True)

    # 每细胞系平均响应（仅训练药）→ 残差目标
    by_line = defaultdict(list)
    for s in train_samples:
        by_line[s["cell_line_idx"]].append(s["expr_delta"])
    mean_delta = {ln: np.mean(v, axis=0) for ln, v in by_line.items()}
    for s in train_samples:
        s["expr_delta"] = s["expr_delta"] - mean_delta[s["cell_line_idx"]]
    for s in test_samples:
        s["expr_delta"] = s["expr_delta"] - mean_delta.get(s["cell_line_idx"],
                                                          next(iter(mean_delta.values())))
    print(f"    残差目标就绪（每细胞系平均已扣除，{len(mean_delta)} 系）", flush=True)

    # 模型 + 训练
    cache = CompoundGraphCache()
    ds = CompoundPertDataset(train_samples, cache)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: {
                            "graph": [x["graph"] for x in b],
                            "cell_line": torch.tensor([x["cell_line"] for x in b], dtype=torch.long),
                            "label": torch.stack([x["label"] for x in b])})
    unipert = _make_unipert(len(gene_vocab) + 1, gene_vocab=gene_vocab,
                            gene_mode="esm").to(DEVICE)
    unipert.load_state_dict(sa["unipert"])
    effect_a = PerturbationEffectModel.build_stage_a(unipert, hvg_n).to(DEVICE)
    effect_a.load_state_dict(sa["effect"])
    effect_b = PerturbationEffectModel.build_stage_b(effect_a).to(DEVICE)
    opt = torch.optim.AdamW(effect_b.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"    训练残差（{args.epochs} epoch）...", flush=True)
    _train_loop(effect_b, loader, opt, "residualB", args.epochs, is_gene=False)
    effect_b.eval()

    # 评测：pred_total = mean_delta[line] + residual_pred
    def pcc(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    pccs, randoms, means = [], [], []
    with torch.no_grad():
        for s in test_samples:
            g = cache.get(s["smiles"])
            cl = torch.tensor([s["cell_line_idx"]], device=DEVICE)
            res_pred = effect_b.forward_compound([g], cl).cpu().numpy()[0]
            base = mean_delta.get(s["cell_line_idx"], next(iter(mean_delta.values())))
            pred_total = base + res_pred
            real = s["expr_delta"] + base          # 真实全量
            p = pcc(pred_total, real)
            if p is None:
                continue
            pccs.append(p)
            randoms.append(pcc(np.random.default_rng(0).normal(size=real.shape), real))
            means.append(pcc(base, real))
    pccs, randoms, means = np.array(pccs), np.array(randoms), np.array(means)
    print(f"\n=== 残差版留出评估（{len(pccs)} 未见药样本，hvg={hvg_n}）===")
    print(f"  模型(平均+残差) PCC : {pccs.mean():+.4f} ± {pccs.std():.4f}")
    print(f"  平均响应基线 PCC   : {means.mean():+.4f} ± {means.std():.4f}")
    print(f"  随机预测基线 PCC   : {randoms.mean():+.4f} ± {randoms.std():.4f}")
    print(f"  相对平均基线       : {pccs.mean() - means.mean():+.4f}")
    print(f"  总耗时             : {__import__('time').time()-t0:.0f}s")
    by_drug = defaultdict(list)
    for s, p in zip(test_samples, pccs):
        by_drug[s["name"]].append(p)
    top = sorted(by_drug.items(), key=lambda kv: -np.mean(kv[1]))[:5]
    print("  Top5 未见药：")
    for n, ps in top:
        print(f"    {n[:40]:42s} PCC={np.mean(ps):+.3f} ({len(ps)} 系)")


if __name__ == "__main__":
    main()
