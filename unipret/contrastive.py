"""
对比对齐损失（InfoNCE）—— UniPert 的核心
=========================================
目标：把"基因扰动"和"作用于同一通路的化合物"编码进同一个向量空间，让它们靠得近。

一句话类比：
  论文里基因和药物被放进"同一本词典"。比如"敲掉 ESR1 基因"和"用他莫昔芬抑制 ESR1"
  说的是同一回事，那这两个向量在统一空间里就该挨着；不相关的就该离得远。
  我们用 InfoNCE（对比学习经典损失）来训练这件事：
  - 正样本：同一个 batch 里第 i 个基因 ↔ 第 i 个化合物（已知作用于同一靶点/通路）
  - 负样本：batch 里其它所有化合物/基因
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TEMPERATURE, PROJECTION_DIM, EMBED_DIM


class ContrastiveAlign(nn.Module):
    """投影头：把基因/化合物表征各自投到统一共享空间（带细胞系条件），并 L2 归一化。"""
    def __init__(self):
        super().__init__()
        self.gene_head = nn.Sequential(
            nn.Linear(EMBED_DIM, PROJECTION_DIM), nn.GELU(),
            nn.Linear(PROJECTION_DIM, PROJECTION_DIM),
        )
        self.comp_head = nn.Sequential(
            nn.Linear(EMBED_DIM, PROJECTION_DIM), nn.GELU(),
            nn.Linear(PROJECTION_DIM, PROJECTION_DIM),
        )

    def forward(self, gene_z, comp_z, cell_cond=None):
        if cell_cond is not None:
            gene_z = gene_z + cell_cond
            comp_z = comp_z + cell_cond
        g = F.normalize(self.gene_head(gene_z), dim=-1)
        c = F.normalize(self.comp_head(comp_z), dim=-1)
        return g, c


def info_nce(g, c, temperature=TEMPERATURE):
    """
    g, c: [B, D] 同一 batch 内第 i 个基因与第 i 个化合物互为正样本。
    返回对称的对比损失（基因→化合物 与 化合物→基因 两个方向平均）。
    """
    B = g.size(0)
    logits = g @ c.t() / temperature            # [B, B]
    labels = torch.arange(B, device=g.device)    # 对角线 = 正样本对
    loss_g = F.cross_entropy(logits, labels)
    loss_c = F.cross_entropy(logits.t(), labels)
    return (loss_g + loss_c) / 2
