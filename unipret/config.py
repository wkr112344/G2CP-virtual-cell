"""
UniPert 统一表征层 —— 全局配置
（对标 Li et al., Cell 2026 的 UniPert-G2CP 架构，按本机 RTX 3050 Ti 4GB 显存调小）
"""
import torch

# 设备：自动选 GPU（3050Ti 4GB），否则退回 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 统一潜空间 ----
EMBED_DIM = 256          # 基因/化合物编码器各自输出的维度（论文用 256–512，4GB 取 256）
PROJECTION_DIM = 256     # 对比学习投影头输出 = 统一共享空间维度
GENE_HIDDEN = 256
COMPOUND_HIDDEN = 256
GENE_ID_DIM = 64        # 基因 ID 嵌入维度（GeneEncoder.gene_id_emb 默认；L_enhance 节点维度）

# ---- 与官方 UniPert 包对齐的硬规格（github.com/TencentAILabHealthcare/UniPert）----
# 官方 encode_genes / encode_compounds 输出的扰动向量固定为 256 维，
# 下游 GEARS / CPA 按此维度读取 adata.uns['UniPert_reps']。我们的统一空间也取 256。
OFFICIAL_EMBED_DIM = 256

# 化合物编码器实现说明（诚实对照论文）：
#   论文原文：SMILES → RDKit ECFP4 拓扑指纹(2048维二元) → 全连接 → 256维稠密嵌入。
#   我们本机无 RDKit（且要纯 PyTorch 可控），改用等效的「分子图 → GINE 图网络 → 256维」。
#   GINE 是表达力更强的图编码器（学习原子/边消息），属于"图路线"而非论文的"指纹+FC 路线"，
#   但目标一致：把分子压到统一 256 维共享空间。若日后装了 RDKit，可在 compound_encoder.py
#   加一个 ECFP4 分支切换回论文原路线。下面 GNN_TYPE 仅作记录，不代表论文化合物用 GCN。
GNN_TYPE = "GINE"        # 本实现化合物编码器 = GINE（图路线）；论文原路线为 ECFP4+FC
COMPOUND_ENCODER_NOTE = ("paper_uses_ECFP4_FC; ours_is_GINE_graph_net (more expressive, no RDKit)")

# ---- 基因编码器模式（4GB 内逼近论文 ESM+蛋白图）----
#   "hybrid"  (默认)：序列 CNN 特征 + 基因 ID 嵌入 融合。有序列的基因带结构信息，
#                      无序列的基因退化为纯 ID 嵌入（与 GEARS 一致）。4GB 可行，逼近论文思想。
#   "embedding"：纯基因 ID 查表（GEARS 风格），不读序列，能吃全部 ~7500 基因扰动做预训练。
#   "sequence"：只用轻量序列 CNN（需每个基因有序列，本地仅 ~20 个，数据太少不推荐单用）。
#   "esm"（P1 新增）：ESM2-8M 冻结特征（320 维，查表缓存）+ 基因 ID 嵌入 融合。
#                     最贴近论文"增强蛋白编码器"；无序列基因退化为纯 ID 嵌入（全基因兼容）。
# 论文基因分支 = ESM 蛋白大模型 + 19187 蛋白 MSA 相似图 + 2 层 GNN。4GB 装不下大 ESM，
# 故 hybrid/esm 是折中：至少让向量带序列/结构线索，而非纯随机 ID。
GENE_ENCODER_MODE = "hybrid"

# ---- ESM2-8M（P1：最贴近论文增强蛋白编码器的最小 ESM，4GB 可装）----
ESM_ENABLED = True          # 训练时是否尝试用 ESM 特征（无缓存/加载失败自动退回 hybrid）
ESM_PATH = "models/esm2_t6_8M_UR50D.pt"     # 相对 unipret/ 目录；已下载（30MB，官方源）
ESM_CACHE = "models/esm_cache.pt"           # 预计算 ESM 嵌入缓存 {基因符号: 320维}

# ---- 损失权重（多任务：表型回归 + 跨域对比 + 基因图自监督）----
W_ALIGN = 0.2            # L_align 跨域对比（基因↔药物 CPI 正样本）权重
W_ENHANCE = 0.1          # L_enhance 基因图自监督权重（掩码邻居预测）
W_MOA = 0.05             # L_moa 同机制药对对比权重（P5-B：0.15 使 SMD 回落，调小到 0.05）
MASK_RATIO = 0.3         # L_enhance 掩码比例

# ---- 细胞系条件（sciPlex3 = MCF7/A549/K562 三系；Norman 单系 K562 取下标 0 兼容）----
NUM_CELL_LINES = 3
CELL_LINE_NAMES = ["K562", "A549", "MCF7"]

# ---- 序列编码 ----
GENE_SEQ_MAX_LEN = 1024   # 蛋白序列截断长度（CRISPR guide 仅 20nt，远短于此）
KMER = 3                  # CNN 卷积核（氨基酸 k-mer）
# 蛋白字母表 + 特殊符（用于把基因/蛋白序列字符转成数字）
CHAR_VOCAB = list("ACDEFGHIKLMNPQRSTVWYBXZ") + ["<pad>", "<unk>"]

# ---- 训练 ----
BATCH_SIZE = 32           # 4GB 显存，单卡小 batch
GRAD_ACCUM = 16           # 梯度累积步数 → 有效 batch = 32*16 = 512（模拟论文大 batch）
LEARNING_RATE = 2e-4
TEMPERATURE = 0.07        # InfoNCE 温度系数（0.1→0.07 更锐利，攻 EF top0.5% 精度区间）
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
SEED = 20260802
