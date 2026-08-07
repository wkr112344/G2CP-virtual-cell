"""
ESM2-8M 基因/蛋白编码器接入模块（P1：补齐论文"增强蛋白编码器"）
=====================================================================
背景（大白话）：
  论文 UniPert 的基因分支 = ESM 蛋白大模型 + MSA 相似图 + GNN，把"蛋白序列语义"
  压进统一空间。我们 4GB 装不下大 ESM，但装得下最小的 **ESM2-8M（esm2_t6_8M，
  7.5M 参数，320 维嵌入）**——这是最贴近论文、且本机可跑的最小 ESM 家族成员。

设计决策：
  1. ESM 权重冻结，只当"特征提取器"：对每个有序列的基因跑一次 ESM → 320 维向量，
     缓存到磁盘（esm_cache.pt）。训练时查表即可，ESM 不需要参与反向传播
     （省显存、快，且符合论文"预训练蛋白编码器"思想）。
  2. EsmGeneEncoder 结构与原 GeneEncoder 一致（gene_id_emb + 特征分支 + fuse）：
       - 有序列的基因：ESM 320 维 → 线性投影 → 与基因 ID 嵌入融合 → 256 维统一空间
       - 无序列的基因：ESM 分支为 0，退化为纯 ID 嵌入（GEARS 风格，兼容全基因）
     这样 sciPlex3/Norman 全量基因（大多无序列）也能用，本地 20 个靶蛋白享受 ESM 语义。
  3. 无缝替换：forward 签名与 GeneEncoder 相同 (seqs, gene_ids)，model.py 调用不变。

用法：
  python unipret/esm_encoder.py          # 构建 esm_cache.pt（读 dataset.json 的 20 条蛋白序列）
"""
import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from .config import EMBED_DIM, GENE_HIDDEN, GENE_ID_DIM

ESM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                        "esm2_t6_8M_UR50D.pt")
ESM_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                              "esm_cache.pt")
ESM_EMBED_DIM = 320          # esm2_t6_8M 的隐藏维度（fixed，见 .pt 内 encoder_embed_dim）
ESM_NAME = "esm2_t6_8M_UR50D"


def load_esm(model_path=ESM_PATH):
    """加载 ESM2-8M 权重（官方 .pt）。返回 (model, alphabet)。

    torch>=2.6 默认 weights_only=True，官方 fairseq checkpoint 含 argparse.Namespace，
    故手动 torch.load(weights_only=False)（权重来自 Meta 官方源，可信），再调
    fair-esm 2.0 的 load_model_and_alphabet_core（跳过可选的 contact-regression 头）。
    """
    import argparse as _ap
    import esm as _esm
    torch.serialization.add_safe_globals([_ap.Namespace])
    data = torch.load(model_path, map_location="cpu", weights_only=False)
    model, alphabet = _esm.pretrained.load_model_and_alphabet_core(ESM_NAME, data, None)
    model.eval()
    return model, alphabet


def esm_mean_embedding(model, alphabet, seqs, device="cpu", batch=2):
    """批量蛋白序列 → [B, 320] 均值池化嵌入（非 pad token 求平均，标准做法）。

    4GB 显存保护：ESM transformer 的 attention 中间量随 batch×序列长增长，
    一次打包全部序列会 OOM（实测 20×600aa → 9.7GB）。故按小 batch 分批前向，
    每条序列 <0.1s（CUDA），20 条总计几秒，缓存一次性构建可接受。
    """
    if not seqs:
        return torch.zeros(0, ESM_EMBED_DIM)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    outs = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        labels, strs, tokens = batch_converter(list(enumerate(chunk)))
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[model.num_layers])
            rep = out["representations"][model.num_layers]          # [B, L, 320]
            mask = (tokens != alphabet.padding_idx).unsqueeze(-1)
            emb = (rep * mask).sum(1) / mask.sum(1).clamp(min=1)    # [B, 320]
        outs.append(emb.cpu())
    return torch.cat(outs, dim=0)


def build_esm_cache(local_dataset_path, out_path=ESM_CACHE_PATH, device="cpu"):
    """把本地基因/蛋白序列一次性编码成 ESM 嵌入，存 {基因符号: 320向量}。

    local_dataset 结构（dataset.json）：proteins = {GENE: {acc, seq}, ...}
    """
    with open(local_dataset_path, encoding="utf-8") as f:
        local = json.load(f)
    seqs, names = [], []
    for name, p in (local.get("proteins") or {}).items():
        if isinstance(p, dict) and p.get("seq"):
            seqs.append(p["seq"]); names.append(name)
    if not seqs:
        print("!! 本地无蛋白序列，跳过 ESM 缓存构建", flush=True)
        return None
    model, alphabet = load_esm()
    embs = esm_mean_embedding(model, alphabet, seqs, device=device)   # [N, 320]
    cache = {n.upper(): embs[i].numpy().astype(np.float32) for i, n in enumerate(names)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(cache, out_path)
    print(f"✅ ESM 缓存已存 {out_path}：{len(cache)} 个基因，维度 {EMBED_DIM} (esm {ESM_EMBED_DIM})",
          flush=True)
    return out_path


def _norm(s):
    return str(s).strip().upper() if s is not None else ""


class EsmGeneEncoder(nn.Module):
    """ESM 增强基因编码器：gene_id_emb + ESM 特征（查表冻结）融合 → EMBED_DIM。

    forward(seqs, gene_ids) 签名与 GeneEncoder 一致（seqs 仅兼容，不参与计算，
    特征全部来自预计算的 ESM 缓存）。无序列基因的 ESM 行为 0，退化为纯 ID 嵌入。
    """
    def __init__(self, num_genes, gene_vocab=None, esm_cache=None,
                 gene_id_dim=GENE_ID_DIM, esm_dim=ESM_EMBED_DIM):
        super().__init__()
        self.gene_id_emb = nn.Embedding(num_genes, gene_id_dim)
        # ESM 特征表 [num_genes, esm_dim]（无序列基因行为 0）——buffer 不进梯度
        feat = torch.zeros(num_genes, esm_dim)
        if esm_cache and gene_vocab:
            hit = 0
            for sym, idx in gene_vocab.items():
                v = esm_cache.get(_norm(sym))
                if v is None:
                    v = esm_cache.get(str(sym).upper())
                if v is not None:
                    feat[idx] = torch.as_tensor(v, dtype=torch.float32)
                    hit += 1
            print(f"    EsmGeneEncoder：ESM 特征覆盖 {hit}/{len(gene_vocab)} 个基因", flush=True)
        self.register_buffer("esm_feat", feat)
        self.esm_proj = nn.Linear(esm_dim, GENE_HIDDEN)
        self.fuse = nn.Linear(GENE_HIDDEN + gene_id_dim, EMBED_DIM)
        self.ln = nn.LayerNorm(EMBED_DIM)
        self.act = nn.GELU()

    def forward(self, seqs=None, gene_ids=None):
        device = next(self.parameters()).device
        if gene_ids.dim() == 1:
            gid = self.gene_id_emb(gene_ids)                  # [B, gene_id_dim]
            esm = self.esm_feat[gene_ids]                     # [B, esm_dim]
        else:  # 多基因扰动：ESM 特征与 ID 都均值聚合
            gid = self.gene_id_emb(gene_ids).mean(dim=1)
            esm = self.esm_feat[gene_ids].mean(dim=1)
        x = self.act(self.esm_proj(esm.to(device)))           # [B, GENE_HIDDEN]
        out = self.fuse(torch.cat([x, gid], dim=1))
        return self.ln(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="构建 ESM 缓存（P1）")
    ap.add_argument("--local", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset.json"))
    ap.add_argument("--out", default=ESM_CACHE_PATH)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> 构建 ESM 缓存（device={dev}）", flush=True)
    build_esm_cache(args.local, args.out, device=dev)
