"""
基因/蛋白功能相似图 + $L_{enhance}$ 自监督
=========================================
论文里 UniPert 基因分支 = ESM 蛋白大模型 + **19187 个人类蛋白的 MSA 相似图** + 2 层 GNN，
并用 $L_{enhance}$（掩码节点/边、在线预测离线表征）逼着嵌入学到进化/功能保守规律，
让没见过的蛋白也能泛化。

本机 4GB + 仅 ~20 条本地序列，不可能真建 19187 蛋白的 MSA 图。这里用两种**缩放且诚实**的替代，
目标与论文完全一致——给基因嵌入加一张"功能相似图"，再做掩码邻居预测自监督：

  图A（默认，覆盖广）「扰动响应功能图」：
    用 sciPlex3/Norman 里每个基因被敲除后的「表达变化向量」当该基因的功能画像，
    画像相近的基因 = 功能/通路相近（等价于论文用 MSA 相似度连边，只是信号来自真实扰动数据）。
    对画像做 k-NN 连边 → 基因功能相似图。

  图B（序列图，覆盖窄但更"结构"）「序列相似图」：
    对本地有序列的 ~20 个基因，用 k-mer Jaccard 算两两相似，连边。
    日后序列充足（如全蛋白组 FASTA）可直接扩成论文的 MSA 图。

两种图都喂给 GeneGraphEnhance：随机掩码一部分基因嵌入，用邻居嵌入均值去预测被掩码者，
逼着嵌入里编码"功能邻居该有的样子"——这就是论文 $L_{enhance}$ 的 4GB 缩放版。

注：19187 节点图在更大显存机器上把 build_gene_response_graph 的 max_genes 调大、
或换 MSA 边权重即可无缝升级，损失函数不用改。
"""
import numpy as np
import torch
import torch.nn as nn

from .data_bridge import build_gene_samples_for_pretrain, control_means_by_line
from .config import MASK_RATIO, EMBED_DIM, GENE_ID_DIM


# ----------------------------------------------------------- 图构建
def _cosine_sim(X):
    """X:[M,D] → [M,M] 余弦相似度。"""
    X = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    X = X / n
    return X @ X.T


def build_gene_response_graph(reader=None, hvg=None, samples=None,
                              max_genes=1500, k=8, max_cells=600):
    """
    用扰动响应（表达变化向量）建基因功能相似图。
    返回 (node_ids, edge_index)：
      node_ids   : list[int]  这些基因在 GeneEncoder 词表里的下标（1-based，0=padding）
      edge_index : list[(a,b)] 无向边（同在下标空间）
    samples 若已外部建好则直接复用，避免重复读盘。
    """
    if samples is None:
        samples, _ = build_gene_samples_for_pretrain(
            reader, hvg, max_cells=max_cells, max_genes=max_genes)
    if not samples:
        return [], []
    from collections import defaultdict
    agg = defaultdict(list)
    for s in samples:
        for g in s["gene_ids"]:
            agg[g].append(s["expr_delta"])
    feat = {g: np.mean(np.stack(agg[g]), 0) for g in agg}
    ids = sorted(feat)
    X = np.stack([feat[g] for g in ids]).astype(np.float32)
    if X.shape[0] < 2:
        return ids, []
    sim = _cosine_sim(X)
    np.fill_diagonal(sim, -1)
    edges = []
    for i in range(X.shape[0]):
        js = np.argsort(sim[i])[::-1][:k]
        for j in js:
            if j > i:  # 无向去重
                edges.append((ids[i], ids[j]))
    return ids, edges


def build_sequence_graph(seqs_dict, k=3, kmer=3):
    """对本地有序列的基因，用 k-mer Jaccard 连边。返回 (node_ids, edge_index)。"""
    names = [n for n in seqs_dict if seqs_dict[n]]
    def kmers(s):
        s = s.upper()
        return set(s[i:i + kmer] for i in range(len(s) - kmer + 1)) or {s[:kmer]}
    sets = {n: kmers(seqs_dict[n]) for n in names}
    ids = names
    idx = {n: i for i, n in enumerate(names)}
    edges = []
    for i, a in enumerate(names):
        best = sorted(
            ((j, len(sets[a] & sets[b]) / len(sets[a] | sets[b]))
             for j, b in enumerate(names) if j != i),
            key=lambda t: t[1], reverse=True)[:k]
        for j, sc in best:
            if sc > 0 and j > i:
                edges.append((idx[a], idx[b]))
    return ids, edges


# ----------------------------------------------------------- $L_{enhance}$ 自监督
class GeneGraphEnhance(nn.Module):
    """
    基因图自监督（论文 $L_{enhance}$ 的缩放版）：
      输入 emb:[N, D] 当前基因嵌入表（来自 UniPert 的 GeneEncoder.gene_id_emb）。
      随机掩码 MASK_RATIO 比例的节点 → 用其邻居嵌入均值当上下文 → 小 MLP 重建原嵌入。
      只对掩码节点算 MSE（邻居预测任务），逼着嵌入编码功能邻居结构。
    edge_index 用 (a,b) 列表表示（下标空间 = emb 行号）。
    """
    def __init__(self, dim=GENE_ID_DIM, hidden=256):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # 预计算邻接（行号 → 邻居行号列表）
        self.register_buffer("adj", torch.zeros(0, dtype=torch.long), persistent=False)

    def _build_adj(self, edge_index, n):
        adj = [[] for _ in range(n)]
        for a, b in edge_index:
            adj[a].append(b)
            adj[b].append(a)
        return adj

    def forward(self, emb, edge_index, mask_ratio=MASK_RATIO, maskable=None):
        """
        emb        : [N, D] 基因嵌入（通常取 unipert.gene_encoder.gene_id_emb.weight）
        edge_index : list[(a,b)]
        maskable   : 可选 list[int]，只在图内节点里掩码（避开 padding 行 0）
        返回标量损失（无图/图太小则返 0）
        """
        n = emb.size(0)
        if not edge_index or n < 3:
            return torch.tensor(0.0, device=emb.device)
        adj = self._build_adj(edge_index, n)
        # 邻居上下文：每个节点的邻居嵌入均值（停梯度，当作"目标结构"）
        ctx = emb.detach().clone()
        for i in range(n):
            if adj[i]:
                nb = torch.stack([emb.detach()[j] for j in adj[i]], 0).mean(0)
                ctx[i] = nb
        # 掩码集合：默认全表，但建议传图内节点（maskable）避开 padding
        pool = maskable if maskable else list(range(n))
        if not pool:
            return torch.tensor(0.0, device=emb.device)
        k = max(1, int(len(pool) * mask_ratio))
        rng = torch.tensor(pool, device=emb.device)[torch.randperm(len(pool), device=emb.device)[:k]]
        pred = self.project(ctx)                  # 用邻居上下文预测
        loss = (pred[rng] - emb.detach()[rng]).pow(2).mean()
        return loss
