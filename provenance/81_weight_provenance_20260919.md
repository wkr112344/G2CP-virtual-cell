# #16 主模型权重训练集出处（#13 发现 2 的 A/B/C 判定）（2026-09-19）

脚本 `81_weight_provenance.py` ｜ 门 **6 条**（G81f 为如实未闭项）｜ placeholder **1** 条 ｜ 耗时 0.5 s

> 待答问题（用户 11:53 点名）：**主模型权重训练集出处——决定发现 2 是 A/B/C 哪种。**
> 取法：**不猜，从代码里找回划分语句，再逐字复算。**

## 0. 一句话

**是 C（并且 A 也排除）：主模型的域内适配把本轮留出的那 3,204 个药全部留出，两个留出集 md5 逐位相同；那 1,909 个「落在旧训练集」的药属于另一个实验（跨源共同可评子集对照）的划分，与主模型无关。**

→ 主模型臂读数**不需要**加 `pending_leakage_audit`，应改写为「域内适配留出集与本轮留出逐位相同（md5 `29fcd23a0cf1d3dfa0712b8f035dfde6`），重叠 **0 / 3204**」。

## 1. 四份权重各自的训练脚本

| 权重 | 臂 | 适配缓存 | 训练脚本 | 脚本在案 |
|---|---|---|---|---|
| `g2cp_full_cpi_v7.pt` | base_pretrained | — | `—（官方免安装包内置）` | **否** |
| `g2cp_v7_beta_ft.pt` | main_domainadapted | g2cp_cache_beta_all | `verify_ft_20260911.py` | 是 |
| `g2cp_v7_beta_ft2_both.pt` | control_v2cache_domainadapted | g2cp_cache_beta_v2 | `verify_ft2_20260911.py` | 是 |
| `g2cp_v7_dcic_ft.pt` | dcic2021_domainadapted | g2cp_cache_2021 | `p3_dcic_adapt_20260914.py` | 是 |

划分语句（逐字）：`pert_uid = np.where(kind == 0, key, NGENE + key); rng = np.random.RandomState(0); perts = sorted(set(int(p) for p in pert_uid)); perm = rng.permutation(len(perts)); te_set = 后 10%`。

`g2cp_full_cpi_v7.pt` 是官方免安装包内置的基座，**训练脚本/划分不在案** —— 这是本步唯一的未闭项。

## 2. 复算各缓存上「原划分」与「该权重的适配训练药物集」

| 缓存 | 行数 | 药物 kind | 扰动数 | 留出扰动 | 适配训练药物 | 全部药物 |
|---|---|---|---|---|---|---|
| `g2cp_cache_beta_all` | 157522 | 1 | 32039 | 3204 | 28835 | 32039 |
| `g2cp_cache_beta_v2` | 158094 | 1 | 32039 | 3204 | 28835 | 32039 |
| `g2cp_cache_2021` | 156931 | 1 | 32039 | 3204 | 28835 | 32039 |
| `g2cp_cache_rebuilt` | 202200 | 0 | 20370 | 2037 | 18333 | 20370 |

> 两套缓存的 `kind` 约定相反（逐字取自代码）：beta 系缓存 `train_g2cp_contrast.py::prep_extra` 用 `kind==1` 表示药物；`g2cp_cache_rebuilt`（`build_y.py` L192 注释「0 = drug branch」）用 `kind==0` 表示药物。复算时按各自约定取药物行，二者 `drug_vocab` 均**字典序有序**（G81b），故「按整数 key 排序」与「按药物名字排序」得到的划分**逐位相同**。

## 3. 新留出 ∩ 各权重的适配训练药物

| 权重 | 臂 | **∩ 适配训练药物** | ∩ 适配留出药物 | 不在该缓存覆盖内 |
|---|---|---|---|---|
| `g2cp_v7_beta_ft.pt` | main_domainadapted | **0（0.0%）** | 3204 | 0 |
| `g2cp_v7_beta_ft2_both.pt` | control_v2cache_domainadapted | **0（0.0%）** | 3204 | 0 |
| `g2cp_v7_dcic_ft.pt` | dcic2021_domainadapted | **0（0.0%）** | 3204 | 0 |

三份权重（主 / 对照 / dcic）的适配留出集都**逐位等于**新留出：

- 新留出 ids md5：`29fcd23a0cf1d3dfa0712b8f035dfde6`
- 主权重适配留出 md5：`29fcd23a0cf1d3dfa0712b8f035dfde6` ⇒ **相同**（G81d）

## 4. 旧划分（`H=2,037`）复算 + 与 #13 对账

| 量 | 本步复算 | #13 的 L10 | 一致 |
|---|---|---|---|
| 旧药物宇宙 | 20370 | 20370 | 是 |
| 旧留出 H | 2037 | 2037 | 是 |
| 旧训练 | 18333 | 18333 | 是 |
| 新留出 ∩ 旧训练 | 1909 | 1909 | 是 |
| 新留出 ∩ 旧留出 | 192 | 192 | 是 |
| 旧留出 md5 | `84a999451255e44c2943afc1d0fed993` | `84a999451255e44c2943afc1d0fed993` | **逐位一致** |

⇒ #13 的 L10 读数经独立复算复核通过（G81c）。

## 5. 判定 A / B / C

**结论：非 A、非 B，是 C**

**依据**：主权重 `g2cp_v7_beta_ft.pt` 的域内适配脚本是 `verify_ft_20260911.py`，数据集 `g2cp_cache_beta_all`，划分语句 = `RandomState(0)` + `sorted(set(pert_uid))` + 取后 10%。该脚本逐字可复算 ⇒ 复算得：新留出 3,204 药里**0 个落在适配训练段**、3204 个落在适配留出段、0 个不在该 cache 覆盖内。而且**两个留出集 md5 逐位相同**（`29fcd23a0cf1d3dfa0712b8f035dfde6`）。

- A 不成立：主模型域内适配把这 3,204 药**全部留出**，它在这批药的行上没有做过任何梯度更新。
- B 不成立：记录其实是清楚的。`20_split_provenance.md` §1 已把 20,370 / H=2,037 逐行定位到代码（`step36_official_split.py` L26/L37、`step55_v2_beta2020.py` L29/L70-L85、`controlled_swap_source_sensitivity.py` L26/L64-L88），并给出复跑实测（rebuilt 202,200 行 / 观测药 20,370 / H=2,037 / P*=11,275）。
- C 成立：`H=2,037` 是**跨源共同可评子集**实验（三源对照：C*=64 系 / D*=20,370 药）的划分，作用在 `g2cp_cache_rebuilt` 上；它不是主模型的训练集，也与主口径划分无关。新留出 ∩ 旧训练 = 1909（59.6%）这个数之所以大，只是因为**两次抽样落在同一个 32,039 药物空间的不同子集上**，并非泄漏。

**处置**：L2/L4 主模型臂读数**不需要**加 `pending_leakage_audit`，改为写「主权重域内适配的留出集与本轮药物级留出逐位相同（md5 `29fcd23a0cf1d3dfa0712b8f035dfde6`），重叠 0/3,204」。唯一剩下的未闭项是**上游基座** `g2cp_full_cpi_v7.pt` 的训练划分（G81f）。

## 6. 关键后果（口径要改的两处）

**后果一（好）**：主模型臂在这 3,204 药上确是我们的冷启动——域内适配恰好把它们留出，且是**同一份**留出集（md5 `29fcd23a0cf1d3dfa0712b8f035dfde6`）。原先挂的 `pending_leakage_audit` 可以摘掉，换成一条带数字的正面陈述。

**后果二（须改口径）**：`self_constructed_drug_holdout_v1` 与「32,039 药物词表上 `RandomState(0)` 取后 10%」**逐位相同** ⇒ 它不是一次独立的新划分，而是该构造在完整词表上的自然结果。因此：

- **不能**把它当作「独立第二个划分」来支撑「换划分也稳健」的论证；
- `self_constructed` 这个命名会误导，稿里与 sidecar 里应改为「32,039 药物词表上的 `RandomState(0)` 后 10% 留出（= 主权重域内适配留出集）」。

## 7. 未闭项

- 上游基座 `g2cp_full_cpi_v7.pt` 的训练划分：官方包未附训练脚本/划分记录，本机不在案 ⇒ 「基座是否见过这批留出药物」仍未闭；若稿中要把主模型臂当冷启动评估，必须补这一条或改口径。

## 8. 门

| 门 | 结果 | 关键读数 |
|---|---|---|
| G81a | OK | `{"n": 4, "missing": ["g2cp_full_cpi_v7.pt"]}` |
| G81b | OK | `{"vocab_sorted": {"g2cp_cache_beta_all": true, "g2cp_cache_beta_v2": true, "g2cp_cache_2021": true, "g2cp_cache_rebuilt": true}, "note": "drug_vocab 有序 ⇒ sorted(set(key))（整数）与 sorted(set(drug_id))（字符串）给出同一顺序 ⇒ 两种复算口径等价"}` |
| G81c | OK | `{"l10_md5": "84a999451255e44c2943afc1d0fed993", "here_md5": "84a999451255e44c2943afc1d0fed993", "note": "两处独立复算的旧留出集 md5 一致 ⇒ #13 的 L10 读数可引用"}` |
| G81d | OK | `{"new_holdout_md5": "29fcd23a0cf1d3dfa0712b8f035dfde6", "main_ft_te_md5": "29fcd23a0cf1d3dfa0712b8f035dfde6", "verdict": "新留出 == 主权重 FT 留出集"}` |
| G81e | OK | `{"new_held_in_main_ft_train": 0, "new_held_in_main_ft_test": 3204, "new_held_not_in_cache": 0, "verdict": "非 A、非 B，是 C"}` |
| G81f | **未过（如实未闭）** | `{"reason": "上游基座训练划分不在案（如实标注的未闭项，不计入 problems）"}` |

## 9. 哈希

| 文件 | md5 |
|---|---|
| `81_weight_provenance.py` | `0733f2d40068f1a76e0eb48113ca8bd5` |
| `81_weight_provenance.json` | `cba7b1de11ebae59f06874e9f230139a` |
| `81_render_md.py` | `5659a1773056fb4b664591fe825d0fed` |
| `78_leakage_audit.json` | `db9ca718b634fa4aed7258e21806ee86` |
| `verify_ft_20260911.py` | `32297cd97622a139e593eacd63addd47` |
| `verify_ft2_20260911.py` | `c325c9ff8a563378634e7a4a9e330269` |
| `p3_dcic_adapt_20260914.py` | `666df0d4336b5374fefec941ee7ea7df` |
| `step36_official_split.py` | `8f2fc2a011be079b88292c2b9462f1d6` |
| `build_y.py` | `b2259c554b8b0a7e0cb3c7a77266271c` |
| `holdout/beta2020/drug_holdout.ids.txt` | `29fcd23a0cf1d3dfa0712b8f035dfde6` |
