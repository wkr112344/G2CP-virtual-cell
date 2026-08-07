"""
基因敲除预测评测（P8）：Norman 留出基因 → forward_gene 预测 vs 真实表达变化
=============================================================================
回应"基因预测准不准"：给一个实测 PCC 数字（模型 vs 平均基线 vs 随机）。
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

NORMAN = "C:/Users/wkr20/Desktop/virtual_cell_real_data/genetic/NormanWeissman2019_filtered.h5ad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="stageB.pt")
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    import torch
    from unipret.config import DEVICE
    from unipret.data_bridge import PerturbationReader, select_hvg, build_gene_samples_for_pretrain
    from unipret.train_stages import _make_unipert
    from unipret.effect_model import PerturbationEffectModel

    sb = torch.load(args.ckpt, map_location=DEVICE)
    gene_vocab = sb["gene_vocab"]
    hvg_n = sb.get("hvg_dim", args.hvg)
    unipert = _make_unipert(len(gene_vocab) + 1, gene_vocab=gene_vocab,
                            gene_mode="esm").to(DEVICE)
    unipert.load_state_dict(sb["unipert"], strict=False)
    effect = PerturbationEffectModel.build_stage_a(unipert, hvg_n).to(DEVICE)
    effect.load_state_dict(sb["effect"], strict=False)
    effect.eval()

    reader = PerturbationReader(NORMAN, backed=True)
    hvg = select_hvg(reader, n=hvg_n, max_cells=50000)
    samples, _gv = build_gene_samples_for_pretrain(reader, hvg, max_cells=4000, max_genes=None)
    reader.close()
    for s in samples:
        s["name"] = s["gene_names"][0] if s.get("gene_names") else ""
    samples = [s for s in samples if s["name"] in gene_vocab]
    names = sorted({s["name"] for s in samples})
    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(len(names) * args.holdout))
    test_names = set(rng.choice(names, n_test, replace=False).tolist())
    test = [s for s in samples if s["name"] in test_names]
    train = [s for s in samples if s["name"] not in test_names]
    print(f"    基因样本 {len(samples)}（{len(names)} 基因）；测试基因 {len(test_names)}", flush=True)

    mean_d = defaultdict(list)
    for s in train:
        mean_d[s["cell_line_idx"]].append(s["expr_delta"])
    mean_delta = {ln: np.mean(v, axis=0) for ln, v in mean_d.items()}

    def pcc(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    pccs, randoms, means = [], [], []
    with torch.no_grad():
        for s in test:
            gid = gene_vocab.get(s["name"])
            if gid is None:
                continue
            cl = torch.tensor([s["cell_line_idx"]], device=DEVICE)
            pred = effect.forward_gene(torch.tensor([gid], device=DEVICE),
                                       seqs=None, cell_line_idx=cl).cpu().numpy()[0]
            real = s["expr_delta"]
            base = mean_delta.get(s["cell_line_idx"], next(iter(mean_delta.values())))
            p = pcc(pred, real)
            if p is None:
                continue
            pccs.append(p)
            randoms.append(pcc(np.random.default_rng(0).normal(size=real.shape), real))
            means.append(pcc(base, real))
    pccs, randoms, means = np.array(pccs), np.array(randoms), np.array(means)
    print(f"\n=== 基因敲除预测评测（{len(pccs)} 留出基因样本，ckpt={args.ckpt}）===")
    print(f"  模型预测 PCC : {pccs.mean():+.4f} ± {pccs.std():.4f}")
    print(f"  平均基线 PCC : {means.mean():+.4f} ± {means.std():.4f}")
    print(f"  随机基线 PCC : {randoms.mean():+.4f} ± {randoms.std():.4f}")
    print(f"  相对平均基线 : {pccs.mean() - means.mean():+.4f}")


if __name__ == "__main__":
    main()
