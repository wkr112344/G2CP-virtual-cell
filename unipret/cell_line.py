"""
细胞系条件嵌入
==============
同一药物在不同细胞系（如 K562 白血病 vs 293T 肾胚）里扰动出的转录组不同，
所以 UniPert 把"细胞系"也编码成一个向量，加到基因/化合物表征上（条件化）。
"""
import torch.nn as nn

from .config import NUM_CELL_LINES, EMBED_DIM


class CellLineCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NUM_CELL_LINES, EMBED_DIM)

    def forward(self, idx):
        """idx: LongTensor[B] 细胞系下标 → [B, EMBED_DIM]"""
        return self.emb(idx)
