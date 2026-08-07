"""
UniPert 统一模型 —— 把上面所有模块拼起来
=========================================
结构（对标论文的统一分子表征层）：
  基因编码器 ─┐
              ├─→ 统一共享潜空间（对比对齐）← 细胞系条件
  化合物图编码器 ─┘

基因编码器有两种模式（论文级精度用 embedding）：
  - gene_encoder_mode="embedding"：可学习基因嵌入(GEARS 风格)，能吃下 sciPlex3
    全部 ~7500 个基因扰动做预训练，不依赖蛋白序列 → 论文级精度路径。
  - gene_encoder_mode="sequence" ：轻量序列 CNN（保留给离线轻量推理）。

对外暴露：
  - encode_gene(seqs, gene_ids, cell_line_idx)      → 基因扰动向量
  - encode_compound(graphs, cell_line_idx)         → 化合物向量
  - forward(...)                                    → 投影+归一化后的 (g, c)，供对比损失用
"""
import torch.nn as nn

from .config import EMBED_DIM, GENE_ENCODER_MODE, ESM_ENABLED, ESM_CACHE
from .gene_encoder import GeneEncoder, GeneEmbeddingEncoder
from .esm_encoder import EsmGeneEncoder
from .compound_encoder import CompoundEncoder
from .cell_line import CellLineCondition
from .contrastive import ContrastiveAlign


def _load_esm_cache_or_none():
    """尝试读 ESM 缓存；失败返回 None（调用方自动退回 hybrid）。"""
    if not ESM_ENABLED:
        return None
    try:
        import os
        import torch
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, ESM_CACHE)
        if not os.path.isfile(path):
            return None
        # 缓存为纯 numpy 数组 + dict，无不可信对象；weights_only=False 以兼容 torch>=2.6
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"    ⚠️ ESM 缓存加载失败（退回 hybrid）：{e}", flush=True)
        return None


class UniPert(nn.Module):
    def __init__(self, num_genes, gene_encoder_mode=GENE_ENCODER_MODE,
                 gene_vocab=None, esm_cache=None):
        super().__init__()
        # 'esm'：ESM2-8M 冻结特征 + 基因ID（P1，最贴近论文增强蛋白编码器）；
        # 'embedding'：纯 ID 查表(GEARS风格)；'hybrid'/'sequence'：带 ID 融合的序列编码器
        if gene_encoder_mode == "esm":
            if esm_cache is None:
                esm_cache = _load_esm_cache_or_none()
            if esm_cache is not None:
                self.gene_encoder = EsmGeneEncoder(num_genes, gene_vocab=gene_vocab,
                                                   esm_cache=esm_cache)
            else:
                print("    ⚠️ 未找到 ESM 缓存，基因编码器回退为 hybrid", flush=True)
                self.gene_encoder = GeneEncoder(num_genes)
        elif gene_encoder_mode == "embedding":
            self.gene_encoder = GeneEmbeddingEncoder(num_genes)
        else:  # 'hybrid' 或 'sequence' 都用带 ID 融合的序列编码器（无序列时退化为 ID）
            self.gene_encoder = GeneEncoder(num_genes)
        self.compound_encoder = CompoundEncoder()
        self.cell_line = CellLineCondition()
        self.align = ContrastiveAlign()

    def encode_gene(self, seqs=None, gene_ids=None, cell_line_idx=None):
        z = self._encode_gene(seqs, gene_ids)
        if cell_line_idx is not None:
            z = z + self.cell_line(cell_line_idx)
        return z

    def _encode_gene(self, seqs, gene_ids):
        if isinstance(self.gene_encoder, (GeneEmbeddingEncoder, EsmGeneEncoder)):
            return self.gene_encoder(None, gene_ids)
        # hybrid/sequence：序列可缺省（缺省时退化为纯 ID 嵌入）
        return self.gene_encoder(seqs, gene_ids)

    def encode_compound(self, graphs, cell_line_idx=None):
        z = self.compound_encoder(graphs)
        if cell_line_idx is not None:
            z = z + self.cell_line(cell_line_idx)
        return z

    def forward(self, gene_seqs, gene_ids, compound_graphs, cell_line_idx=None):
        gz = self._encode_gene(gene_seqs, gene_ids)
        cz = self.compound_encoder(compound_graphs)
        cell_cond = self.cell_line(cell_line_idx) if cell_line_idx is not None else None
        g, c = self.align(gz, cz, cell_cond)
        return g, c

    def gene_id_embedding(self):
        """返回基因 ID 嵌入表权重 [num_genes, dim]，供 $L_{enhance}$ 图自监督用。
        hybrid/sequence 模式存在 gene_id_emb；纯 embedding 模式存在 emb。"""
        ge = self.gene_encoder
        if hasattr(ge, "gene_id_emb"):
            return ge.gene_id_emb.weight
        return ge.emb
