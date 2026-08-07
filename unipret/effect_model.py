"""
扰动-表型预测模型（PerturbationEffectModel）—— G2CP 下游核心
================================================================
【关键重构：UniPert 与 G2CP 真正接起来了】
  论文的灵魂：UniPert 把基因/药物编码进统一 256 维空间（因空间），
  G2CP 直接吃这个统一表征做「扰动 → 表型」回归（果空间）。
  本实现里 PerturbationEffectModel **复用 UniPert 的编码器**（权重共享）：
    · forward_gene    = UniPert.encode_gene   → head_gene  → 表达变化
    · forward_compound = UniPert.encode_compound → head_comp → 表达变化
  所以 G2CP 的输入就是 UniPert 的统一嵌入；阶段 A 训出的基因编码器在阶段 B 被
  原样复用（遗传→化学迁移），不再是两套互不相干的嵌入表。

两阶段训练（train_stages.py）：
  · 阶段 A（遗传预训练）：在 sciPlex3/Norman 全部基因 CRISPR 扰动上，
    训 UniPert 基因编码器 + head_gene，学「基因扰动 → 表达变化」。+ $L_{enhance}$ 图自监督。
  · 阶段 B（化学微调）：冻结/复用阶段 A 的 UniPert（基因编码器 + head_gene），
    加化合物路（UniPert 化合物编码器 + head_comp），用对齐的化合物真实转录组微调。

适配 4GB 显存：HVG 默认 2000 维标签、AMP、梯度累积、batch 小。
"""
import torch
import torch.nn as nn

from .config import DEVICE, EMBED_DIM, PROJECTION_DIM


class PerturbationEffectModel(nn.Module):
    def __init__(self, unipert, hvg_dim, hidden=512, dropout=0.1,
                 with_compound_head=True):
        """
        unipert           : 已构建的 UniPert 实例（基因/化合物统一编码器，权重被本模型共享）
        hvg_dim           : 表达变化标签维度（默认 2000）
        with_compound_head: 阶段 A 可设为 False（不建化合物头，省参）；阶段 B 必须 True
        """
        super().__init__()
        self.unipert = unipert
        self.hvg_dim = hvg_dim
        # 基因路：UniPert 统一基因嵌入(256) → head
        self.head_gene = nn.Sequential(
            nn.Linear(EMBED_DIM, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hvg_dim),
        )
        # 化合物路：UniPert 统一化合物嵌入(256) → head（维度与基因路一致，便于迁移）
        if with_compound_head:
            self.head_comp = nn.Sequential(
                nn.Linear(EMBED_DIM, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hvg_dim),
            )

    # ---------- 基因扰动表示：直接复用 UniPert 统一编码（含细胞系条件） ----------
    def forward_gene(self, gene_ids, seqs=None, cell_line_idx=None):
        z = self.unipert.encode_gene(seqs=seqs, gene_ids=gene_ids,
                                     cell_line_idx=cell_line_idx)   # [B, 256] 统一嵌入
        return self.head_gene(z)                                   # [B, hvg_dim]

    def forward_compound(self, graphs, cell_line_idx=None):
        z = self.unipert.encode_compound(graphs, cell_line_idx=cell_line_idx)  # [B, 256]
        return self.head_comp(z)                                    # [B, hvg_dim]

    @classmethod
    def build_stage_a(cls, unipert, hvg_dim, with_compound_head=False):
        """阶段 A：只用基因路（化合物头可选不建，省参）。"""
        return cls(unipert, hvg_dim, with_compound_head=with_compound_head)

    @classmethod
    def build_stage_b(cls, stage_a_effect):
        """阶段 B：复用阶段 A 的 UniPert（基因编码器 + head_gene），加化合物头。
        基因编码器的权重通过共享同一 unipert 对象自然迁移，无需手动 copy。"""
        m = cls(stage_a_effect.unipert, stage_a_effect.hvg_dim,
                with_compound_head=True)
        m.head_gene.load_state_dict(stage_a_effect.head_gene.state_dict())
        return m

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
