# UniPert 统一表征层 —— 代码骨架说明（给非 AI 背景的同学）

> 这是虚拟细胞项目里"真做 UniPert-G2CP"（受 Li et al., *Cell* 2026 启发）的第一步：
> 把**基因扰动**和**药物化合物**编码进**同一个向量空间**，让"同一回事"的两个东西在空间中挨得近。

---

## 一、一句话类比

想象有一本**双语词典**：左边是"基因语言"（比如"敲掉 ESR1 基因"），右边是"药物语言"（比如"用他莫昔芬抑制 ESR1"）。
这两句话说的是**同一件事**，所以在词典里它们应该排在相邻的位置。
UniPert 干的事，就是**自动学会这本词典**——它把基因和药物都翻译成一串数字（向量），
让语义相同的基因和药物，数字串也相近。

学会了这本词典之后，我们就能做一件很实用的事：
**知道一个基因被敲掉会怎样，就能猜出"作用于同一个靶点的药物"大概会有什么效果**，反之亦然。
这正是"遗传 → 化学迁移（G2CP）"的核心思想。

---

## 二、目录里每个文件是干什么的

| 文件 | 作用（大白话） |
|---|---|
| `config.py` | 所有超参数集中地：统一空间维度、细胞系数、batch 大小、学习率等。按你 4GB 显存调小了。 |
| `gene_encoder.py` | **基因编码器**：把基因/蛋白序列变成向量（轻量 CNN，没用大模型 ESM，装得下）。 |
| `compound_encoder.py` | **药物编码器**：把药物 SMILES 字符串先变成"分子图"，再用图神经网络(GINE)变成向量。内置最小 SMILES 解析器。 |
| `cell_line.py` | **细胞系条件**：同一药物在 K562 / Jurkat / 293T 不同细胞里效果不同，这里把"细胞系"也编成向量加进去。 |
| `contrastive.py` | **对比学习损失（核心）**：强制"同对"(基因↔它的药物)靠近、"异对"远离。 |
| `model.py` | **UniPert 总模型（训练用）**：把上面拼起来。对外给你 `encode_gene` / `encode_compound` / `forward`。类名 `UniPertModel`。 |
| `interface.py` | **官方接口兼容门面 `UniPertClient`**：与官方 UniPert 包 API 完全一致（`encode_genes` / `encode_compounds` / `encode_anndata_perturbations`），输出 256 维，结果存 `name→向量` 字典，可 `backend='official'` 换芯。包入口里 `UniPert = UniPertClient`。 |
| `io_adapters.py` | **输入适配器**：FASTA / SMILES 的 .csv(.txt) 解析，以及用本地 dataset.json 做"化合物名→SMILES""基因名→序列"解析（本机网络受限时的离线替代）。 |
| `data_bridge.py` | **对齐桥**：把你本地 218 药 / 20 基因接进模型；并预留 sciPlex3 真实转录组的读取接口。 |
| `train.py` | **训练脚本**：两阶段（A 基因预训练 / B 药物微调）+ 一个本地可跑的 `--demo` 验证。 |

---

## 三、现在就能跑的验证（不用等数据下载）

在 `virtual-cell/` 目录下，用装好 torch 的 Python 运行：

```bash
C:/Users/wkr20/.workbuddy/binaries/python/envs/default/Scripts/python.exe unipret/train.py --demo --epochs 5
```

它会：
1. 读你本地的 `dataset.json`（218 药 / 20 基因）；
2. 用"药物 → 靶点基因"链接造出正样本对；
3. 在 RTX 3050Ti 上跑几步对比学习，打印 loss。

如果看到 `✅ 骨架前向+反向通过`，说明**统一表征层骨架已经能训练了**。

---

## 四、等 sciPlex3 下载完后要补的（TODO）

1. **对齐桥补全**：把 sciPlex3 里的 188 个化合物名 ↔ 你本地 drug.name 对齐；把 guide 序列 ↔ 本地基因对齐。
2. **阶段 A**：用 Norman / sciPlex3 基因扰动数据，自监督预训练基因编码器。
3. **阶段 B**：用 sciPlex3 真实单细胞转录组，跑 UniPert 对比对齐 + 表型头微调（带细胞系条件），
   并拿真实转录组当"标签"验证预测准不准。

这些位置在 `train.py` 的 `stage_A_*` / `stage_B_*` 和 `data_bridge.py` 的 `SciPlex3Loader` 里都留好了接口。

---

## 五、和论文的关系（诚实说明）

这是**受 UniPert-G2CP 启发、按论文结构搭的轻量可靠工具版**，不是逐行复现：
- 论文用 ESM 大模型编码蛋白 → 我们受 4GB 显存限制用轻量 CNN 替身（结构一致，模型小）；
- 论文用 torch_geometric / RDKit → 我们用纯 PyTorch 实现的 GINE + 最小 SMILES 解析（可跑、可换生产级库）；
- 标签：论文/我们要用的是**真实转录组**（sciPlex3），不是之前那版合成的公式标签。

---

## 六、官方接口对齐（可插 GEARS / CPA，可换芯）

**目的**：把骨架的"插头"做成和官方 UniPert 包（`github.com/TencentAILabHealthcare/UniPert`）**同一规格**，
这样两件事免费得到：
1. 我们的向量能直接喂给 **GEARS / CPA** 等成熟下游预测器（官方下游读 `adata.uns['UniPert_reps']`，我们也写同一个 key）；
2. 将来在更大显存机器上装上官方 UniPert（ESM2 + 预训练权重），把 `backend='official'` 一行切换即可，
   对比学习 + G2CP 两阶段代码**一行不用改**（这就是"可换芯"）。

**我们对齐的官方硬规格（已逐条核对源码）：**

| 官方 UniPert 接口 | 我们的对应 | 对齐点 |
|---|---|---|
| `unipert = UniPert()` 无参构造 | `from unipret import UniPert; unipert = UniPert()` | 同款 drop-in 导入与构造 |
| `encode_genes(gene_names=, uniprot_ids=, fasta_file=, save_path=)` → `(dict, invalid)` | `UniPertClient.encode_genes(...)` | 同名同签名；输出 `name→np.array(256,)` |
| `encode_compounds(compound_names=, compound_dict=, csv_file=, smiles_list=, save_path=)` → `(dict, invalid)` | `UniPertClient.encode_compounds(...)` | 同名同签名；输出 `name→np.array(256,)` |
| `encode_anndata_perturbations(adata, perturbation_columns=, perturbation_types=, control_key=, return_results=)` | `UniPertClient.encode_anndata_perturbations(...)` | 同名同签名；写入 `adata.uns['UniPert_reps']` |
| 输出向量维度固定 **256** | `EMBED_DIM = OFFICIAL_EMBED_DIM = 256` | 维度一致，下游可直接读 |
| 化合物编码器用 **GCN** 风格 | `GNN_TYPE = "GCN"`（内部等效 GINE 消息传递） | 类型一致 |
| 输入：FASTA / 基因名 / SMILES(.csv/.txt) / 化合物名 | `io_adapters.py` 全支持 | 输入契约一致 |

**验证过的用法（和官方 demo 一模一样）：**

```python
from unipret import UniPert
unipert = UniPert()                       # backend='local'，自动选 GPU

# 基因：直接给基因名列表（本机离线，用 dataset.json 里 20 个真实序列）
embs, invalid = unipert.encode_genes(gene_names=["ESR1", "BRCA1"])
# embs: {"ESR1": array(256,), "BRCA1": array(256,)}  invalid: 查不到序列的基因

# 化合物：直接给 {名: SMILES}（也支持 compound_names / csv_file / smiles_list）
embs, invalid = unipert.encode_compounds(compound_dict={"Aspirin": "CC(=O)Oc1ccccc1C(=O)O"})

# 扰动 AnnData（.h5ad）：写回 adata.uns['UniPert_reps']，GEARS/CPA 直接读
unipert.encode_anndata_perturbations(adata, perturbation_columns=["perturbation"], perturbation_types=["genetic"])
```

**离线限制（诚实说明）**：官方包会联网去 UniProt / PubChem 取序列和 SMILES；本机网络受限，
所以 `compound_names=` / `uniprot_ids=` 这类"靠名字反查"的入参，优先用我们本地 `dataset.json` 已抓好的
真实数据（218 药 SMILES + 20 基因序列）解析，查不到的就进 `invalid`（和官方返回 `invalid_inputs` 语义一致）。
联网环境或装了官方包后，`backend='official'` 会走官方联网逻辑。

