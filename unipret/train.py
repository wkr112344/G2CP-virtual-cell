"""
UniPert 训练骨架（两阶段，对标 G2CP 的"先基因后药物"迁移）
==========================================================
阶段 A（遗传预训练）：在大规模 CRISPR 基因扰动数据（Norman / sciPlex3 基因部分）上，
           先把基因编码器训扎实（自监督：预测扰动后的表达变化 / 对比同基因不同条件）。
阶段 B（化学微调）：用 UniPert 对比学习把"基因扰动"和"化合物"对齐到统一空间，
           再用 sciPlex3 真实转录组做表型头微调与验证，带细胞系条件。

本文件即可运行： `python train.py --demo`  会用你本地 dataset.json 的
"药物→靶点"正样本对，在 3050Ti 上跑几步对比学习，验证骨架前向/反向能通。
（无需等 sciPlex3 下载完。）
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# 让脚本能 import 同目录的 unipret 包（包根目录是 virtual-cell/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unipret import (  # noqa: E402
    DEVICE, SEED, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, WEIGHT_DECAY,
    UniPertModel, info_nce,
)
from unipret.data_bridge import (  # noqa: E402
    load_local_dataset, build_positive_pairs, LocalPairDataset, collate,
)

torch.manual_seed(SEED)


def build_model(dataset):
    # 论文级基因编码器：用基因嵌入(GEARS 风格)而非序列 CNN，能吃下全部基因扰动
    model = UniPertModel(
        num_genes=len(dataset["genes"]), gene_encoder_mode="embedding"
    ).to(DEVICE)
    return model


def train_unipert(model, loader, epochs, desc="UniPert"):
    """通用训练循环：AMP 混合精度 + 梯度累积（4GB 显存友好）。"""
    optim = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    model.train()
    for ep in range(epochs):
        running = 0.0
        acc = 0
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            gene_ids = batch["gene_ids"].to(DEVICE)
            cell_lines = batch["cell_lines"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                g, c = model(
                    batch["gene_seqs"], gene_ids, batch["compound_graphs"], cell_lines
                )
                loss = info_nce(g, c) / GRAD_ACCUM
            scaler.scale(loss).backward()
            running += loss.item() * GRAD_ACCUM
            acc += 1
            if acc % GRAD_ACCUM == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                acc = 0
        # epoch 末尾把剩余累积梯度也更新一次（避免小数据集上梯度被丢弃）
        if acc > 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
        print(f"[{desc}] epoch {ep+1}/{epochs}  loss={running/len(loader):.4f}")


def run_demo(data_path, epochs=5):
    """本地可跑的最小验证：用"药物→靶点"正样本对做对比学习。"""
    print(f"DEVICE = {DEVICE}  (RTX 3050 Ti 4GB 自动启用 CUDA)")
    dataset = load_local_dataset(data_path)
    pairs, meta = build_positive_pairs(dataset)
    print(f"本地基因数={len(dataset['genes'])}  药物数={len(dataset['drugs'])}  "
          f"正样本对={len(pairs)}")
    if not pairs:
        print("没有可用正样本对（dataset.json 的 drug.targets 为空？）"); return
    ds = LocalPairDataset(dataset, pairs, meta)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    model = build_model(dataset)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量 ≈ {n_params:.2f}M")
    train_unipert(model, loader, epochs, desc="demo-contrastive")
    print("✅ 骨架前向+反向通过，UniPert 统一表征层可训练。")


# ---------------- 阶段 A / B 骨架（等 sciPlex3 到货后填充数据读取） ----------------
def stage_A_genetic_pretrain(dataset, h5ad_genetic):
    """TODO: 读 Norman/sciPlex3 基因扰动 h5ad，自监督预训练基因编码器。"""
    raise NotImplementedError(
        "阶段 A 待实现：用 SciPlex3Loader/ anndata 读入基因扰动表达，"
        "对基因编码器做自监督预训练（如预测扰动前后表达差）。"
    )


def stage_B_chemical_finetune(dataset, h5ad_sciplex3, epochs):
    """TODO: 用 sciPlex3 真实转录组做 UniPert 对比对齐 + 表型头微调。"""
    raise NotImplementedError(
        "阶段 B 待实现：sciPlex3 到货后，把化合物名对齐到本地 drug.name、"
        "guide 对齐到基因，跑 UniPert 对比学习 + 真实转录组表型头微调。"
    )


def main():
    ap = argparse.ArgumentParser(description="UniPert 训练骨架")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "dataset.json"))
    ap.add_argument("--demo", action="store_true", help="用本地数据跑最小验证（不需 sciPlex3）")
    ap.add_argument("--stage", choices=["A", "B"], help="跑阶段 A 基因预训练 / 阶段 B 药物微调")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--genetic-h5ad", default=None, help="Norman/sciPlex3 基因扰动 h5ad 路径")
    ap.add_argument("--sciplex3-h5ad", default=None, help="sciPlex3.h5ad 路径")
    args = ap.parse_args()

    data_path = os.path.abspath(args.data)
    dataset = load_local_dataset(data_path)

    if args.demo:
        run_demo(data_path, epochs=args.epochs)
    elif args.stage == "A":
        stage_A_genetic_pretrain(dataset, args.genetic_h5ad)
    elif args.stage == "B":
        stage_B_chemical_finetune(dataset, args.sciplex3_h5ad, args.epochs)
    else:
        # 默认就跑 demo，方便一键验证骨架
        run_demo(data_path, epochs=args.epochs)


if __name__ == "__main__":
    main()
