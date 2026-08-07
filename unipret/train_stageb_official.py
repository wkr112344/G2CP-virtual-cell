"""
阶段 B · 官方 ECFP4 表征版（P8）
================================
用户拍板方案：药物表征用官方预训练（ECFP4 + Linear，SMD 2.62 验证过），
表型预测头用我们 sciPlex3 高质量数据训。

流程：
  1) 加载 stageA.pt（基因路 ESM 迁移，strict=False 跳过旧 compound_encoder）
  2) 官方 cp_encoder 权重覆盖 compound_encoder.linear_layer（白捡官方表征）
  3) sciPlex3 136 药 × 3 系，按药 80/20 留出，训表型头
  4) 实时进度侦测：每 epoch 写 train_progress.json + print（无管道缓冲）
  5) 自动评测：未见药 PCC vs 平均基线 vs 随机基线

用法：python unipret/train_stageb_official.py --stageA stageA.pt --epochs 30
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

SCIPLEX3 = "C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad"
CHEM_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sciplex3_chems.json")
OFFICIAL_CKPT = os.path.join(BASE, "refs", "unipert_official", "current_model", "unipert_model.pt")
PROGRESS_FILE = os.path.join(BASE, "train_progress.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageA", default="stageA.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--save", default="stageB_official.pt")
    ap.add_argument("--freeze_cp", action="store_true",
                    help="冻结官方 ECFP4 编码器（只训表型头）")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from unipret.config import DEVICE, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, WEIGHT_DECAY
    from unipret.data_bridge import (PerturbationReader, select_hvg,
                                     build_compound_samples_chembl, CompoundGraphCache)
    from unipret.train_stages import CompoundPertDataset, _make_unipert
    from unipret.effect_model import PerturbationEffectModel

    t0 = time.time()
    print(f">>> 官方 ECFP4 表征版阶段 B（{args.epochs} epoch，留出 {args.holdout}）", flush=True)
    sa = torch.load(args.stageA, map_location=DEVICE)
    gene_vocab = sa["gene_vocab"]
    hvg_n = sa.get("hvg_dim", args.hvg)

    # ---- 数据（sciPlex3 136 药 × 3 系，按药 80/20）----
    reader = PerturbationReader(SCIPLEX3, backed=True)
    hvg = select_hvg(reader, n=hvg_n, max_cells=50000)
    chem_map = json.load(open(CHEM_MAP, encoding="utf-8"))
    all_samples = build_compound_samples_chembl(reader, chem_map, hvg)
    for s in all_samples:
        s["smiles"] = chem_map[s["name"]]["smiles"]
    all_samples = [s for s in all_samples if s.get("smiles")]
    reader.close()
    names = sorted({s["name"] for s in all_samples})
    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(len(names) * args.holdout))
    test_names = set(rng.choice(names, n_test, replace=False).tolist())
    train_samples = [s for s in all_samples if s["name"] not in test_names]
    test_samples = [s for s in all_samples if s["name"] in test_names]
    print(f"    总样本 {len(all_samples)} / 训练 {len(train_samples)} / 测试 {len(test_samples)}", flush=True)

    # ---- 模型：stageA 基因路 + 官方 ECFP4 覆盖 ----
    unipert = _make_unipert(len(gene_vocab) + 1, gene_vocab=gene_vocab,
                            gene_mode="esm").to(DEVICE)
    unipert.load_state_dict(sa["unipert"], strict=False)   # 跳过旧 GNN compound_encoder
    official = torch.load(OFFICIAL_CKPT, map_location=DEVICE, weights_only=True)
    unipert.compound_encoder.linear_layer.load_state_dict(
        {k.replace("linear_layer.", ""): v for k, v in official["cp_encoder"].items()})
    print("    官方 ECFP4 权重已覆盖 compound_encoder（SMD 基准 2.62）", flush=True)
    effect_a = PerturbationEffectModel.build_stage_a(unipert, hvg_n).to(DEVICE)
    effect_a.load_state_dict(sa["effect"], strict=False)   # 跳过旧 GNN compound_encoder
    effect_b = PerturbationEffectModel.build_stage_b(effect_a).to(DEVICE)
    if args.freeze_cp:
        # 只冻结官方 linear_layer（保留官方表征），LN 保持可训练（学归一化尺度）
        for p in effect_b.unipert.compound_encoder.linear_layer.parameters():
            p.requires_grad = False
        print("    已冻结官方 ECFP4 linear_layer（LN 可训练，只训表型头）", flush=True)
    opt = torch.optim.AdamW(effect_b.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()

    # ---- 数据加载 ----
    cache = CompoundGraphCache()
    ds = CompoundPertDataset(train_samples, cache)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: {
                            "graph": [x["graph"] for x in b],
                            "cell_line": torch.tensor([x["cell_line"] for x in b], dtype=torch.long),
                            "label": torch.stack([x["label"] for x in b])})

    # ---- 训练 + 实时进度侦测 ----
    print(f"    训练开始（进度实时写入 {PROGRESS_FILE}）...", flush=True)
    for ep in range(args.epochs):
        effect_b.train()
        run, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for batch in loader:
            gs = batch["graph"]
            cl = batch["cell_line"].to(DEVICE)
            lab = batch["label"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pred = effect_b.forward_compound(gs, cl)
                loss = crit(pred, lab) / GRAD_ACCUM
            loss.backward()
            run += loss.item() * GRAD_ACCUM
            n += 1
            if n % GRAD_ACCUM == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
        elapsed = time.time() - t0
        eta_min = elapsed / (ep + 1) * (args.epochs - ep - 1) / 60
        prog = {"epoch": ep + 1, "total": args.epochs,
                "loss": round(run / max(n, 1), 5),
                "elapsed_s": round(elapsed),
                "eta_min": round(eta_min, 1)}
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(prog, f)
        print(f"  [officialB] epoch {ep+1}/{args.epochs}  loss={prog['loss']:.5f}  "
              f"({elapsed:.0f}s, 预计剩余 {eta_min:.1f}min)", flush=True)

    torch.save({"unipert": unipert.state_dict(), "effect": effect_b.state_dict(),
                "gene_vocab": gene_vocab, "hvg_dim": hvg_n}, args.save)
    try:
        os.remove(PROGRESS_FILE)
    except Exception:
        pass
    print(f"    权重已存 {args.save}", flush=True)

    # ---- 评测：未见药 PCC（vs 平均基线 / 随机基线）----
    effect_b.eval()
    by_line = defaultdict(list)
    for s in train_samples:
        by_line[s["cell_line_idx"]].append(s["expr_delta"])
    mean_delta = {ln: np.mean(v, axis=0) for ln, v in by_line.items()}

    def pcc(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    pccs, randoms, means = [], [], []
    with torch.no_grad():
        for s in test_samples:
            g = cache.get(s["smiles"])
            cl = torch.tensor([s["cell_line_idx"]], device=DEVICE)
            pred = effect_b.forward_compound([g], cl).cpu().numpy()[0]
            real = s["expr_delta"]
            base = mean_delta.get(s["cell_line_idx"], next(iter(mean_delta.values())))
            p = pcc(pred, real)
            if p is None:
                continue
            pccs.append(p)
            randoms.append(pcc(np.random.default_rng(0).normal(size=real.shape), real))
            means.append(pcc(base, real))
    pccs, randoms, means = np.array(pccs), np.array(randoms), np.array(means)
    print(f"\n=== 官方 ECFP4 版留出评估（{len(pccs)} 未见药样本）===")
    print(f"  模型 PCC           : {pccs.mean():+.4f} ± {pccs.std():.4f}")
    print(f"  平均响应基线 PCC   : {means.mean():+.4f} ± {means.std():.4f}")
    print(f"  随机预测基线 PCC   : {randoms.mean():+.4f} ± {randoms.std():.4f}")
    print(f"  相对平均基线       : {pccs.mean() - means.mean():+.4f}")
    by_drug = defaultdict(list)
    for s, p in zip(test_samples, pccs):
        by_drug[s["name"]].append(p)
    top = sorted(by_drug.items(), key=lambda kv: -np.mean(kv[1]))[:5]
    print("  Top5 未见药：")
    for n, ps in top:
        print(f"    {n[:40]:42s} PCC={np.mean(ps):+.3f} ({len(ps)} 系)")
    print(f"  总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
