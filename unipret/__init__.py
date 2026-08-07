"""
UniPert 包入口：统一导出关键类，方便外部 `from unipret import ...`
"""
from .config import (
    DEVICE, EMBED_DIM, PROJECTION_DIM, OFFICIAL_EMBED_DIM, GNN_TYPE,
    NUM_CELL_LINES, CELL_LINE_NAMES,
    BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, WEIGHT_DECAY, TEMPERATURE, SEED,
)
from .gene_encoder import GeneEncoder
from .compound_encoder import CompoundEncoder, smiles_to_graph
from .cell_line import CellLineCondition
from .contrastive import ContrastiveAlign, info_nce
from .model import UniPert as UniPertModel
from .interface import UniPertClient
from . import io_adapters

# 对外暴露与官方 UniPert 包一致的门面：`from unipret import UniPert; unipert = UniPert()`
UniPert = UniPertClient

__all__ = [
    "DEVICE", "EMBED_DIM", "PROJECTION_DIM", "OFFICIAL_EMBED_DIM", "GNN_TYPE",
    "NUM_CELL_LINES", "CELL_LINE_NAMES",
    "BATCH_SIZE", "GRAD_ACCUM", "LEARNING_RATE", "WEIGHT_DECAY", "TEMPERATURE", "SEED",
    "GeneEncoder", "CompoundEncoder", "smiles_to_graph",
    "CellLineCondition", "ContrastiveAlign", "info_nce",
    "UniPertModel", "UniPertClient", "UniPert", "io_adapters",
]
