"""
药物特异性差异验证（P8）
========================
直击核心：模型有没有学到"药与药之间的细微差异"？

方法：对留出的未见药（同一细胞系），取两两药物对：
  - 真实差异向量 = 药A真实表达变化 - 药B真实表达变化
  - 预测差异向量 = 药A模型预测 - 药B模型预测
  - 计算两个差异向量的 Pearson 相关（逐基因）
如果模型只学会了"共性响应"（所有药都预测成差不多），
预测差异≈0 → 相关性≈0 → 特异性没学出来。
如果相关性显著 >0 → 模型真的在区分药物。

对照组：
  - 随机基线：随机噪声差异 vs 真实差异（应≈0）
  - 平均基线：预测差异=0（恒为 0，不参与）
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCIPLEX3 = "C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="stageB.pt")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--n_pairs", type=int, default=300)
    args = ap.parse_args()

    import torch
    from unipret.config import DEVICE
    from unipret.data_bridge import (PerturbationReader, select_hvg,
                                     build_compound_samples_chembl)
    from unipret.train_stages import _make_unipert
    from unipret.effect_model import PerturbationEffectModel

    # ---- 训练/测试划分（与 eval_holdout 一致：按药 80/20）----
    reader = PerturbationReader(SCIPLEX3, backed=True)
    hvg = select_hvg(reader, n=2000, max_cells=50000)
    chem_map = json.load(open("unipret/sciplex3_chems.json", encoding="utf-8"))
    all_s = list(build_compound_samples_chembl(reader, chem_map, hvg))
    reader.close()
    names = sorted({s["name"] for s in all_s})
    rng = np.random.default_rng(20260803)
    n_test = max(1, int(len(names) * 0.2))
    test_names = set(rng.choice(names, n_test, replace=False).tolist())
    test = [s for s in all_s if s["name"] in test_names]
    train = [s for s in all_s if s["name"] not in test_names]
    print(f"    测试药 {len(test_names)} 个 / {len(test)} 样本 | 训练药 {len(train)} 样本", flush=True)

    # ---- 模型 ----
    sb = torch.load(args.ckpt, map_location=DEVICE)
    gv = sb["gene_vocab"]
    hvg_n = sb.get("hvg_dim", 2000)
    unipert = _make_unipert(len(gv) + 1, gene_vocab=gv, gene_mode="esm").to(DEVICE)
    unipert.load_state_dict(sb["unipert"])
    ea = PerturbationEffectModel.build_stage_a(unipert, hvg_n).to(DEVICE)
    effect = PerturbationEffectModel.build_stage_b(ea).to(DEVICE)
    effect.load_state_dict(sb["effect"])
    effect.eval()

    # ---- 对测试样本做预测 ----
    from unipret.data_bridge import CompoundGraphCache
    cache = CompoundGraphCache()
    preds, reals = [], []
    with torch.no_grad():
        for s in test:
            sm = chem_map.get(s["name"], {}).get("smiles", "")
            g = cache.get(sm)
            if g is None or not g.any():
                continue
            p = effect.forward_compound(
                [g], torch.tensor([s["cell_line_idx"]], device=DEVICE)).cpu().numpy()[0]
            preds.append(p)
            reals.append(s["expr_delta"])
    preds = np.stack(preds)
    reals = np.stack(reals)
    print(f"    可预测测试样本 {len(preds)}", flush=True)

    # ---- 药物对差异相关（同一细胞系内、不同药的两两配对）----
    pairs = [(i, j) for i in range(len(test)) for j in range(len(test))
             if i < j
             and test[i]["cell_line_idx"] == test[j]["cell_line_idx"]
             and test[i]["name"] != test[j]["name"]]
    print(f"    合法药物对（同系异药）: {len(pairs)}", flush=True)
    if len(pairs) > args.n_pairs:
        rngp = np.random.default_rng(7)
        keep = rngp.choice(len(pairs), args.n_pairs, replace=False)
        pairs = [pairs[k] for k in keep]

    def pair_corr(A, B):
        """对每对药物 (i,j)：真实差异 A[i]-A[j] vs 预测差异 B[i]-B[j] 的逐基因相关。"""
        corrs = []
        for i, j in pairs:
            da = A[i] - A[j]
            db = B[i] - B[j]
            if np.std(da) < 1e-9 or np.std(db) < 1e-9:
                continue
            corrs.append(np.corrcoef(da, db)[0, 1])
        return np.array(corrs)

    real_diff_corrs = pair_corr(reals, preds)      # 真实差异 vs 模型预测差异
    rng2 = np.random.default_rng(1)
    noise = rng2.normal(size=preds.shape)
    rand_corrs = pair_corr(reals, noise)           # 对照：真实差异 vs 随机差异
    mean_pred = preds.mean(0)
    mean_corrs = pair_corr(reals, np.tile(mean_pred, (len(preds), 1)))  # 对照：猜平均的差异≈0

    print(f"\n===== 药物特异性差异验证（{args.ckpt}）=====")
    print(f"  模型预测差异 vs 真实差异:  PCC = {np.mean(real_diff_corrs):+.4f} ± {np.std(real_diff_corrs):.4f}  (n={len(real_diff_corrs)})")
    print(f"  随机基线差异 vs 真实差异:  PCC = {np.mean(rand_corrs):+.4f} ± {np.std(rand_corrs):.4f}")
    print(f"  猜平均基线差异 vs 真实差异: PCC = {np.mean(mean_corrs):+.4f} ± {np.std(mean_corrs):.4f}")
    gain = np.mean(real_diff_corrs) - np.mean(rand_corrs)
    print(f"\n  特异性信号增益 = 模型 - 随机 = {gain:+.4f}")
    if gain > 0.05:
        print("  判定：模型开始区分药与药（有特异性信号）")
    elif gain > 0.01:
        print("  判定：有微弱特异性信号，但很弱")
    else:
        print("  判定：特异性基本没学出来（只会猜共性）")


if __name__ == "__main__":
    main()
