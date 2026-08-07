"""
留出评估（P3-A）：模型对"没训练过的药"的泛化 PCC
=====================================================================
设计（关键：按药划分，防泄漏）：
  sciPlex3 136 药 → 随机 80% 训练 / 20% 测试（同一药的全部细胞系样本
  一起进训练或测试）。阶段 A（基因预训练）与阶段 B 训练都只用训练药；
  测试药（模型完全没见过）评估 PCC。

对照基线：
  ① 随机预测（高斯噪声）→ PCC≈0
  ② 平均响应基线：预测 = 训练集所有药的平均表达变化（看模型是否胜过"猜平均"）

用法：
  python unipret/eval_holdout.py --stageA stageA.pt --epochs 30 --holdout 0.2
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--holdout", type=float, default=0.2, help="测试药比例")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from unipret.config import DEVICE, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, WEIGHT_DECAY
    from unipret.data_bridge import (PerturbationReader, select_hvg,
                                     build_compound_samples_chembl, CompoundGraphCache)
    from unipret.train_stages import (CompoundPertDataset, _train_loop, _make_unipert,
                                      load_local_dataset, LOCAL)
    from unipret.effect_model import PerturbationEffectModel

    print(f">>> 留出评估：测试药比例 {args.holdout}，seed {args.seed}", flush=True)
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

    # ---- 按药 80/20 划分（防泄漏）----
    names = sorted({s["name"] for s in all_samples})
    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(len(names) * args.holdout))
    test_names = set(rng.choice(names, n_test, replace=False).tolist())
    train_samples = [s for s in all_samples if s["name"] not in test_names]
    test_samples = [s for s in all_samples if s["name"] in test_names]
    print(f"    训练药 {len(set(s['name'] for s in train_samples))} "
          f"/ 测试药 {len(test_names)}（样本 {len(train_samples)}/{len(test_samples)}）", flush=True)
    print(f"    测试药例：{sorted(test_names)[:6]} ...", flush=True)

    # ---- 阶段 B 训练（只用训练药；基因编码器从 stageA 迁移）----
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
    print(f"    阶段 B 训练（{args.epochs} epoch，仅训练药）...", flush=True)
    _train_loop(effect_b, loader, opt, "holdoutB", args.epochs, is_gene=False)
    effect_b.eval()

    # ---- 测试药 PCC 评估 ----
    def pcc_of(pred, real):
        if np.std(pred) < 1e-8 or np.std(real) < 1e-8:
            return None
        return float(np.corrcoef(pred, real)[0, 1])

    pccs, randoms, means = [], [], []
    mean_delta = np.mean([s["expr_delta"] for s in train_samples], axis=0)  # 平均响应基线
    with torch.no_grad():
        for s in test_samples:
            g = cache.get(s["smiles"])
            cl = torch.tensor([s["cell_line_idx"]], device=DEVICE)
            pred = effect_b.forward_compound([g], cl).cpu().numpy()[0]
            real = s["expr_delta"]
            p = pcc_of(pred, real)
            if p is None:
                continue
            pccs.append(p)
            randoms.append(pcc_of(np.random.default_rng(0).normal(size=real.shape), real))
            means.append(pcc_of(mean_delta, real))   # 预测=训练平均响应（猜平均基线）
    pccs, randoms, means = np.array(pccs), np.array(randoms), np.array(means)
    print(f"\n=== 留出评估（{len(pccs)} 个未见药样本，hvg={hvg_n}）===")
    print(f"  模型预测 PCC : {pccs.mean():+.4f} ± {pccs.std():.4f}")
    print(f"  随机预测基线 : {randoms.mean():+.4f} ± {randoms.std():.4f}")
    print(f"  平均响应基线 : {means.mean():+.4f} ± {means.std():.4f}")
    print(f"  相对随机提升 : {pccs.mean() - randoms.mean():+.4f}")
    print(f"  相对平均基线 : {pccs.mean() - means.mean():+.4f}")
    by_drug = defaultdict(list)
    for s, p in zip(test_samples, pccs):
        by_drug[s["name"]].append(p)
    top = sorted(by_drug.items(), key=lambda kv: -np.mean(kv[1]))[:5]
    print("  Top5 预测最好的未见药：")
    for n, ps in top:
        print(f"    {n[:40]:42s} PCC={np.mean(ps):+.3f} ({len(ps)} 系)")


if __name__ == "__main__":
    main()
