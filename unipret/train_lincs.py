"""
LINCS 大样本训练 + 留出评测（P4：33,598 药 → 12,327 基因签名）
=====================================================================
目标：验证"大数据能否让模型学会药物特异性响应"（P3-A 发现模型输给平均基线）。
  · 数据：LINCS level5 consensus（药物级，无细胞系维度 → cell 全 0）
  · 划分：按药 80/20（防泄漏）
  · 模型：UniPert(esm) + head(12327)，从 stageA.pt 迁移编码器
  · 评测：未见药 PCC vs 随机基线 vs 平均响应基线

用法：python unipret/train_lincs.py --epochs 30
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
LINCS = os.path.join(BASE, "data", "lincs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageA", default="stageA.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--max_drugs", type=int, default=6000,
                    help="训练药上限（LINCS 全量 2.7 万 GPU 太慢；6000 药 × 12327 维已远超 sciPlex3）")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from unipret.config import (DEVICE, LEARNING_RATE,
                                WEIGHT_DECAY, SEED)
    from unipret.data_bridge import CompoundGraphCache
    from unipret.train_stages import _make_unipert
    from unipret.effect_model import PerturbationEffectModel

    t0 = __import__("time").time()
    print(f">>> LINCS 大样本训练（{args.epochs} epoch，留出 {args.holdout}）", flush=True)
    data = np.load(os.path.join(LINCS, "lincs_train.npz"), allow_pickle=True)
    names = data["names"]; smiles = data["smiles"]; X = data["X"]; genes = data["genes"]
    # 行标准化（关键修正）：LINCS level5 consensus 系数幅值 ~0.001，MSE 最优解是"预测 0"
    # → 退化成无预测。除以每药自身 std 后，模型被迫学"哪些基因上/下"的形状（PCC 恰好是形状相关）。
    std = X.std(1, keepdims=True)
    X = X / np.maximum(std, 1e-6)
    print(f"    数据 {X.shape[0]} 药 × {X.shape[1]} 基因（已行标准化，"
          f"{X.nbytes/2**30:.1f}GB）", flush=True)

    # ---- 按药 80/20 划分（可选子集加速）----
    rng = np.random.default_rng(args.seed)
    if args.max_drugs and len(set(names.tolist())) > args.max_drugs:
        keep = set(rng.choice(sorted(set(names.tolist())), args.max_drugs,
                              replace=False).tolist())
        idx_keep = [i for i, n in enumerate(names) if n in keep]
        names = names[idx_keep]; smiles = smiles[idx_keep]; X = X[idx_keep]
        print(f"    子集采样：{len(idx_keep)} 药", flush=True)
    uniq = sorted(set(names.tolist()))
    n_test = max(1, int(len(uniq) * args.holdout))
    test_names = set(rng.choice(uniq, n_test, replace=False).tolist())
    tr_i = [i for i, n in enumerate(names) if n not in test_names]
    te_i = [i for i, n in enumerate(names) if n in test_names]
    print(f"    训练 {len(tr_i)} 药 / 测试 {len(te_i)} 药", flush=True)

    # ---- 模型（从 stageA 迁移 UniPert；head 用 12327 维新建）----
    sa = torch.load(args.stageA, map_location=DEVICE)
    gene_vocab = sa["gene_vocab"]
    unipert = _make_unipert(len(gene_vocab) + 1, gene_vocab=gene_vocab,
                            gene_mode="esm").to(DEVICE)
    unipert.load_state_dict(sa["unipert"])
    effect_a = PerturbationEffectModel.build_stage_a(unipert, hvg_dim=X.shape[1]).to(DEVICE)
    effect = PerturbationEffectModel.build_stage_b(effect_a).to(DEVICE)
    print(f"    模型参数量 {effect.num_params()/1e6:.1f}M（head 输出 {X.shape[1]} 维）", flush=True)

    # ---- 训练（药物级样本；cell 全 0）----
    cache = CompoundGraphCache()
    def make_loader(idx):
        sms = [smiles[i] for i in idx]
        labels = [X[i] for i in idx]
        return sms, labels
    tr_sms, tr_lab = make_loader(tr_i)
    crit = torch.nn.MSELoss()
    opt = torch.optim.AdamW(effect.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # ---- 预构图（缓存；33.6k SMILES 用 8 线程并行，避免串行 50 分钟）----
    from concurrent.futures import ThreadPoolExecutor
    print("    预构图（SMILES → 分子图，8 线程）...", flush=True)
    from unipret.compound_encoder import smiles_to_ecfp4
    gcache = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for s, g in zip(tr_sms, ex.map(smiles_to_ecfp4, tr_sms)):
            gcache[s] = g
    for s in tr_sms:
        if gcache.get(s) is not None:
            cache.cache[s] = gcache[s]
    # 过滤构图失败/空图（None 或 atom_types 为空，如 LINCS 缺失 SMILES 的 "-"）
    valid_pos = [j for j, s in enumerate(tr_sms)
                 if gcache.get(s) is not None and len(gcache[s][0]) > 0]
    tr_sms = [tr_sms[j] for j in valid_pos]
    tr_pos = [tr_i[j] for j in valid_pos]           # 原始样本索引（对齐 X）
    print(f"    构图完成 {len(cache.cache)}（过滤后训练药 {len(tr_sms)}，"
          f"耗时 {__import__('time').time()-t0:.0f}s）", flush=True)
    if not tr_sms:
        print("    !! 全部构图失败，退出", flush=True)
        return
    effect.train()
    BS, GA = 128, 4   # 大 batch 加速（有效 512）；4GB 可吃 128×12327 输出头
    n_batch = (len(tr_i) + BS - 1) // BS
    for ep in range(args.epochs):
        perm = rng.permutation(len(tr_sms))
        run = 0.0
        opt.zero_grad(set_to_none=True)
        for st in range(0, len(perm), BS):
            pos = perm[st:st + BS]                       # 训练子集内位置
            graphs = [gcache[tr_sms[j]] for j in pos]    # 直接取已构图（valid_pos 保证非 None）
            lab = torch.tensor([X[tr_pos[j]] for j in pos], device=DEVICE)
            cl = torch.zeros(len(pos), dtype=torch.long, device=DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pred = effect.forward_compound(graphs, cl)
                loss = crit(pred, lab) / GA
            loss.backward()                              # 模型 13.9M，fp32 直接训（不用 GradScaler）
            run += loss.item() * GA
            if (st // BS + 1) % GA == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
        if len(perm) // BS + 1 % GA != 0:
            opt.step(); opt.zero_grad(set_to_none=True)
        print(f"  [lincs] epoch {ep+1}/{args.epochs} loss={run/n_batch:.5f} "
              f"({__import__('time').time()-t0:.0f}s)", flush=True)
    effect.eval()

    # ---- 测试药 PCC 评估（构图 + 过滤无效）----
    def pcc(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    te_sms = [smiles[i] for i in te_i]
    te_pos = []
    for j, s in enumerate(te_sms):
        g = gcache.get(s)
        if g is None or not g.any():
            g = smiles_to_ecfp4(s)
        if g is not None and g.any():
            gcache[s] = g
            te_pos.append(j)
    pccs, randoms, means = [], [], []
    mean_delta = X[[tr_pos[j] for j in range(len(tr_pos))]].mean(0)
    with torch.no_grad():
        for j in te_pos:
            i = te_i[j]
            g = gcache[te_sms[j]]
            cl = torch.zeros(1, dtype=torch.long, device=DEVICE)
            pred = effect.forward_compound([g], cl).cpu().numpy()[0]
            real = X[i]
            p = pcc(pred, real)
            if p is None:
                continue
            pccs.append(p)
            randoms.append(pcc(np.random.default_rng(0).normal(size=real.shape), real))
            means.append(pcc(mean_delta, real))
    pccs, randoms, means = np.array(pccs), np.array(randoms), np.array(means)
    print(f"\n=== LINCS 留出评估（{len(pccs)} 未见药，12,327 基因）===")
    print(f"  模型预测 PCC : {pccs.mean():+.4f} ± {pccs.std():.4f}")
    print(f"  随机预测基线 : {randoms.mean():+.4f} ± {randoms.std():.4f}")
    print(f"  平均响应基线 : {means.mean():+.4f} ± {means.std():.4f}")
    print(f"  相对随机提升 : {pccs.mean() - randoms.mean():+.4f}")
    print(f"  相对平均基线 : {pccs.mean() - means.mean():+.4f}")
    print(f"  总耗时        : {__import__('time').time()-t0:.0f}s")


if __name__ == "__main__":
    main()
