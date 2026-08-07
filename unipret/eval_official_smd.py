"""
官方 UniPert 预训练模型 SMD 评测（P8）
=====================================
加载官方 cp_encoder（ECFP4 → Linear(2048→256)，无 LN），对 sciPlex3 136 药算嵌入，
按 MoA target 分组测 SMD，与我们 GNN 版（0.86）和随机基线（0.66）对比。

结论性目的：官方 ECFP4 表征在"同机制药聚类"上到底多强，验证我们 ECFP4 改造方向。
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

OFFICIAL_CKPT = os.path.join(BASE, "refs", "unipert_official", "current_model", "unipert_model.pt")
CHEMS = os.path.join(BASE, "unipret", "sciplex3_chems.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=OFFICIAL_CKPT)
    ap.add_argument("--chems", default=CHEMS)
    args = ap.parse_args()

    from unipret.compound_encoder import smiles_to_ecfp4, ECFP4_BITS
    from unipret.eval_metrics import load_drug_groups, smd_score

    # ---- 官方 cp_encoder（Linear 2048→256，无 LayerNorm）----
    w = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    lin = nn.Linear(ECFP4_BITS, 256)
    sd = {k.replace("linear_layer.", ""): v for k, v in w["cp_encoder"].items()}
    lin.load_state_dict(sd)
    lin.eval()
    print(f"    官方 cp_encoder 已加载（Linear {ECFP4_BITS}→256）", flush=True)

    # ---- sciPlex3 药物 ECFP4 → 官方嵌入 ----
    drugs, groups = load_drug_groups(args.chems)
    print(f"    sciPlex3 药物 {len(drugs)}，MoA 组 {len(groups)}", flush=True)
    feats = []
    valid = []
    with torch.no_grad():
        for i, dr in enumerate(drugs):
            fp = smiles_to_ecfp4(dr["smiles"])
            if fp is None:
                feats.append(None)
                continue
            x = torch.from_numpy(fp).float().unsqueeze(0)      # [1, 2048]
            z = lin(x).numpy()[0]                              # [256]
            feats.append(z)
            valid.append(i)
    F = np.stack([feats[i] for i in valid])
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    sim = F @ F.T
    print(f"    有效嵌入 {len(valid)}/{len(drugs)}", flush=True)

    results = smd_score(sim, groups, set(valid))
    results.sort(key=lambda r: -r[1])
    mean_smd = float(np.mean([r[1] for r in results])) if results else 0.0
    print(f"\n===== 官方 UniPert SMD（sciPlex3 MoA 分组）=====")
    for t, s, n in results[:10]:
        print(f"  {t:<24} SMD={s:+.2f}  ({n} 药)")
    print(f"\n  平均 SMD = {mean_smd:.2f}")
    print(f"  对照：随机基线 0.66 | 我们 GNN 版 0.86 | 论文 UniPert 1.85")


if __name__ == "__main__":
    main()
