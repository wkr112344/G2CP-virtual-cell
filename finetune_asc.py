# -*- coding: utf-8 -*-
"""finetune_asc.py —— 用 GSE61302(真实 ASC 脂肪分化) 微调全基因模型。

目标: 教会模型 ASC 脂肪分化程序(FABP4/ADIPOQ/LEP 上调), 修正"癌细胞外推"偏差。

方法:
- 加载 g2cp_full.pt(143系全基因)
- 构造 ASC 脂肪分化"合成扰动样本":
  将 15 个 GSE61302 样本表达(基因级)转成 12,328 维向量, 对齐到全基因缓存
- 每个样本作为 ASC 细胞系的一个"表达状态样本", 与 gctx 的 ASC 药物样本一起,
  用 MSE + PCC 微调 head, 编码器小 lr 微调
"""
import sys, os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "data", "g2cp_cache_fullgene")
from train_g2cp_contrast import G2CPNet
from unipret.compound_encoder import ECFP4_BITS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--from", dest="load_from", default="g2cp_full.pt")
    ap.add_argument("--save", default="g2cp_full_asc.pt")
    args = ap.parse_args()

    # ---------- 全基因缓存 ----------
    m = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
    kind, key, cell = m["kind"], m["key"], m["cell"]
    gene_vocab = [str(x) for x in m["gene_vocab"]]
    drug_vocab = [str(x) for x in m["drug_vocab"]]
    cl_names = [str(x) for x in m["cl_names"]]
    n_gene, n_drug = len(gene_vocab), len(drug_vocab)
    y = np.load(os.path.join(CACHE, "y.npy"), mmap_mode="r")
    fps = np.load(os.path.join(CACHE, "drug_fps.npy"))
    n_out = y.shape[1]
    n = len(kind)
    log(f"全基因数据: {n} 样本 | 基因 {n_gene} | 药 {n_drug} | 系 {len(cl_names)} | 输出 {n_out}")

    # ---------- 加载预训练 ----------
    net = G2CPNet(n_gene, ECFP4_BITS, 512, len(cl_names), n_out, 1024).to(DEVICE)
    net.cls_out = 1
    net._fps = torch.from_numpy(fps).to(DEVICE)
    ck = torch.load(args.load_from, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["net"], strict=False)
    log(f"加载 {args.load_from}")

    # ---------- GSE61302 ASC 表达样本 ----------
    geo = json.load(open(os.path.join(BASE, "data", "geo_asc", "gse61302_gene_expr.json")))
    geo_expr = geo["expr"]  # {symbol: [15值]}
    labels = geo["labels"]  # undiff/d7/d21
    n_geo = len(geo["samples"])
    # 对齐到 12,328 基因
    hvg = json.load(open(os.path.join(CACHE, "hvg.json")))
    # 表达值转 log2(芯片值本身已 log), 直接标准化到模型输出尺度
    # 模型输出是 z-score 样(均值0方差~1), GEO 是芯片强度(几百到上万) → 需转换
    # 方案: 每个基因在 15 样本内 z-score(保留样本间差异), 乘缩放 + 均值对齐
    gene_mat = np.zeros((n_geo, n_out), dtype=np.float32)
    for j, g in enumerate(hvg):
        if g in geo_expr:
            v = np.array(geo_expr[g], dtype=np.float64)
            # log2 变换压动态范围
            v = np.log2(np.clip(v, 1, None))
            gene_mat[:, j] = v
    # 对每个基因: z-score(15样本)
    gm = gene_mat.mean(0, keepdims=True)
    gs = gene_mat.std(0, keepdims=True) + 1e-6
    gene_mat_z = (gene_mat - gm) / gs
    # 有值的基因掩码(GEO 覆盖的)
    has_data = np.array([1 if g in geo_expr else 0 for g in hvg], dtype=np.float32)
    log(f"GSE61302: {n_geo} 样本 | 基因对齐 {int(has_data.sum())}/{n_out} | z-score 完成")

    # ASC 细胞系索引
    if "ASC" not in cl_names:
        log("错误: ASC 不在细胞系词表!")
        return
    asc_idx = cl_names.index("ASC")
    log(f"ASC 细胞系索引: {asc_idx}")

    # ---------- 微调 ----------
    # 编码器小 lr, head 正常 lr
    enc_params = [p for nm, p in net.named_parameters() if not nm.startswith("head")]
    head_params = [p for nm, p in net.named_parameters() if nm.startswith("head")]
    opt = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr},
        {"params": enc_params, "lr": args.lr * 0.1},
    ], weight_decay=1e-4)

    geo_t = torch.from_numpy(gene_mat_z).to(DEVICE)
    has_t = torch.from_numpy(has_data).to(DEVICE)
    cell_t = torch.tensor([asc_idx] * n_geo, device=DEVICE).long()

    # 锚点: 用现有 ASC 药物样本的细胞嵌入
    for ep in range(args.epochs):
        net.train()
        total, tl = 0, 0.0
        opt.zero_grad()
        # batch 洗牌
        perm = torch.randperm(n_geo, device=DEVICE)
        B = 8
        for s in range(0, n_geo, B):
            bi = perm[s:s + B]
            # 用随机药物/基因嵌入作为扰动编码(ASC 微调主要是让 head 学会脂肪状态)
            # 简化: 用 ASC 细胞嵌入 + 随机扰动嵌入, 目标 = GEO 样本表达
            kb = torch.randint(0, n_gene, (len(bi),), device=DEVICE).long()
            z = F.normalize(net.gene_emb(kb), dim=1)
            c = net.cell_emb(cell_t[bi])
            out = net.head(torch.cat([z, c], dim=1))
            yb = geo_t[bi]
            # 损失: 只对有数据的基因
            mask = has_t.unsqueeze(0)
            om = out * mask
            ym = yb * mask
            lm = F.mse_loss(om, ym)
            # PCC(有数据基因)
            oc = om - om.mean(1, keepdim=True)
            yc = ym - ym.mean(1, keepdim=True)
            den = oc.norm(dim=1) * yc.norm(dim=1) + 1e-8
            pcc = (oc * yc).sum(1) / den
            lp = (1 - pcc).mean()
            loss = lm + lp
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += len(bi)
            tl += loss.item() * len(bi)
        log(f"epoch {ep+1}/{args.epochs} | loss {tl/total:.4f}")
        if (ep + 1) % 5 == 0:
            torch.save({"net": net.state_dict(), "gene_vocab": gene_vocab, "drug_vocab": drug_vocab,
                        "cl_names": cl_names, "hvg": hvg}, args.save)
            log(f"  保存 {args.save}")
    torch.save({"net": net.state_dict(), "gene_vocab": gene_vocab, "drug_vocab": drug_vocab,
                "cl_names": cl_names, "hvg": hvg}, args.save)
    log(f"微调完成 {args.save}")


if __name__ == "__main__":
    main()
