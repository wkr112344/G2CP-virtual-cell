"""
基因 / 蛋白 / guide 序列编码器（轻量版，不依赖 ESM 大模型）
=====================================================================
思路（大白话）：
  一个基因扰动，论文里是用"这个基因的序列 + 基因身份"来表示的。
  我们把序列变成字符 → 用 1D 卷积(CNN)抓氨基酸 k-mer 局部模式 → 池化 → MLP，
  再叠加一张"基因身份嵌入表"（每个基因符号一个可学习向量，记住序列之外的身份/功能信息）。

为什么不用 ESM：ESM-2 大模型在 4GB 显存装不下，这里用轻量 CNN 替身，
结构(统一空间 + 对比学习)和论文一致，模型小但思想相同。
"""
import torch
import torch.nn as nn

from .config import (EMBED_DIM, GENE_HIDDEN, GENE_SEQ_MAX_LEN, CHAR_VOCAB)


class CharEmbedding(nn.Module):
    """把一串蛋白/DNA 序列字符变成整数下标，再嵌入成向量。"""
    def __init__(self, vocab, dim):
        super().__init__()
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.emb = nn.Embedding(len(vocab), dim, padding_idx=self.stoi["<pad>"])

    def encode(self, seqs):
        idx = []
        for s in seqs:
            t = [self.stoi.get(c, self.stoi["<unk>"]) for c in s[:GENE_SEQ_MAX_LEN]]
            idx.append(t)
        maxlen = max((len(t) for t in idx), default=1)
        for t in idx:
            t += [self.stoi["<pad>"]] * (maxlen - len(t))
        return torch.tensor(idx, dtype=torch.long)

    def forward(self, seqs):
        return self.emb(self.encode(seqs))   # [B, L, dim]


class GeneEncoder(nn.Module):
    """混合基因编码器（默认 gene_encoder_mode='hybrid'）——
    序列 CNN 特征 与 基因 ID 嵌入 融合，4GB 内逼近论文 ESM+蛋白图思想。
      有序列的基因：序列 CNN 抓 k-mer 局部模式 + 身份嵌入 → 带结构线索的向量
      无序列的基因：序列分支退化为 0，只剩身份嵌入（与 GEARS 纯 ID 嵌入一致）
    这样既能用本地 ~20 条真实序列，也能无缝覆盖 sciPlex3 全部 ~7500 个 KO 基因。
    """
    def __init__(self, num_genes, gene_id_dim=64):
        super().__init__()
        self.char_emb = CharEmbedding(CHAR_VOCAB, 32)
        # 多尺度 CNN：同时看 3/5/7 长度的局部模式
        self.convs = nn.ModuleList([
            nn.Conv1d(32, GENE_HIDDEN // 2, k, padding=k // 2) for k in (3, 5, 7)
        ])
        self.act = nn.GELU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.seq_proj = nn.Linear(GENE_HIDDEN // 2 * 3, GENE_HIDDEN)
        # 基因身份嵌入表：每个基因符号一个向量（始终可用）
        self.gene_id_emb = nn.Embedding(num_genes, gene_id_dim)
        self.fuse = nn.Linear(GENE_HIDDEN + gene_id_dim, EMBED_DIM)
        self.ln = nn.LayerNorm(EMBED_DIM)

    def forward(self, seqs, gene_ids):
        """
        seqs     : list[str] | None   蛋白/DNA 序列（None/空/多基因扰动时退化为纯 ID 嵌入）
        gene_ids : LongTensor[B] 或 [B, k]   单基因下标，或多基因扰动(如双敲)的下标集合
        返回 [B, EMBED_DIM]
        多基因扰动按 GEARS 风格对 ID 嵌入均值池化；序列分支仅在「单基因且有序列」时启用。
        """
        device = next(self.parameters()).device
        # 基因身份嵌入：单基因直接查，多基因均值池化
        if gene_ids.dim() == 1:
            gid = self.gene_id_emb(gene_ids)               # [B, gene_id_dim]
        else:
            gid = self.gene_id_emb(gene_ids).mean(dim=1)   # [B, gene_id_dim] 多基因均值
        # 序列分支：仅单基因 + 有序列时启用（多基因/无序列 → 退化为纯 ID 嵌入）
        use_seq = (seqs is not None) and (gene_ids.dim() == 1)
        if use_seq:
            x = self.char_emb.emb(self.char_emb.encode(seqs).to(device))  # [B, L, 32]
            x = x.transpose(1, 2)                   # [B, 32, L]
            feats = []
            for conv in self.convs:
                feats.append(self.act(conv(x)))      # [B, H/2, L]
            x = torch.cat([self.pool(f).squeeze(-1) for f in feats], dim=1)  # [B, H/2*3]
            x = self.act(self.seq_proj(x))           # [B, GENE_HIDDEN]
        else:
            x = torch.zeros(gid.size(0), GENE_HIDDEN, device=device)
        out = self.fuse(torch.cat([x, gid], dim=1))
        return self.ln(out)


class GeneEmbeddingEncoder(nn.Module):
    """
    基因嵌入编码器（GEARS 风格，论文级精度路径，不依赖蛋白序列）。
    ===================================================================
    为什么用它（大白话）：论文 G2CP 的下游预测 backbone(GEARS) 给每个基因一个
    可学习向量，而不是去读蛋白序列——这样能直接用"成千上万个被敲除的基因"来
    预训练，数据量是序列 CNN 的上百倍。我们本地只有 20 个基因有序列，
    用 CNN 几乎没数据可训；用嵌入则能吃下 sciPlex3 全部 ~7500 个基因扰动。

    输入 gene_ids: [B] 单基因，或 [B, k] 多基因扰动(如双敲)，取均值聚合
    （符合 GEARS "扰动=基因集表示" 的做法）。
    输出 [B, EMBED_DIM]，与化合物编码器输出维度一致，进入统一对比空间。
    """
    def __init__(self, num_genes, hidden=EMBED_DIM):
        super().__init__()
        self.emb = nn.Embedding(num_genes, hidden)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, gene_ids):
        if gene_ids.dim() == 1:
            z = self.emb(gene_ids)               # [B, hidden]
        else:
            z = self.emb(gene_ids).mean(dim=1)   # [B, hidden] 多基因均值
        return self.ln(z)
