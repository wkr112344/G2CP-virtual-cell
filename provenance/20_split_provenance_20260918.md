# 旧划分取证定案（三样东西全翻到）· 2026-09-18

> 指令来源：用户第 13 条消息「beta2020 旧划分必须翻，而且要找三样东西」。
> 纪律：本文只记**读到的代码与复跑得到的数**，不引旧稿自述、不凭记忆。
> 全部结论均已在本机复跑验证（复跑脚本与输出见 §5）。

---

## 0. 一句话结论

| 问题 | 结论 |
|---|---|
| 20,370 怎么来的 | **A 臂缓存 `g2cp_cache_rebuilt` 的观测药物覆盖面**（= 观测药(rebuilt) ∩ 观测药(cache_2021)）。不是 10% 产物，不是官方给的数。 |
| 15,994 vs 15,157 | **两个都是本工程复算**，无一来自原文脚本。15,994 = 复刻本机旧 `G2CP_repo/eval_full.py` 写法；15,157 = 本工程早期自建写法。 |
| 官方划分脚本 | **找到了**（GEARS `PertData.prepare_split`，`default_rng` + `choice`），**但不可套用**（官方是 5 个细胞系各自独立、seed 非 0）⇒ 我们的划分无官方对应物。 |
| `split_name` | **`self_constructed_drug_holdout_v1`**。已撤回此前「官方口径 / 官方划分」的标签。 |

---

## 1. 第一样：20,370 的构造代码

**构造处（在读的代码里逐字定位）：**

```python
# controlled_swap_source_sensitivity.py  L26 / L64-L88
DRB = os.path.join(BASE, "data", "g2cp_cache_rebuilt")          # L26
mb  = np.load(os.path.join(DRB, "meta.npz"), allow_pickle=True) # L65
a_key = mb["key"].astype(np.int64)                              # L66
a_cell = mb["cell"].astype(np.int64)                            # L67
a_cl   = mb["cl_names"];  a_dv = mb["drug_vocab"]               # L68-L69
a_cn = [a_cl[c] for c in a_cell];  a_dn = [a_dv[k] for k in a_key]
cellsA = set(a_cn); drugA = set(a_dn)          # L76-L77
Cstar = sorted(cellsA & cellsB)                # L78   共同细胞系
Dstar = sorted(drugsA & drugsB)                # L79   共同药物
perts = sorted(Dstar); n_te = int(len(perts) * 0.9)             # L84-L85
perm = np.random.RandomState(0).permutation(len(perts))         # L86
H = set(perts[i] for i in perm[n_te:])                          # L87   留出 = 后 10%
```

同一定义在 `step55_v2_beta2020.py` L29/L70-L85 里被**逐行照抄**（该脚本 docstring 明写「设计保持旧口径不变：C*=64 系 / D*=20,370 药 / H=2,037 留出 / P*=11,275 评测对」，并在 L86 用 `assert (64, 20370, 2037, 11275)` 卡住）；最早出处是 `step36_official_split.py` L26/L37（`CACHE = data/g2cp_cache_rebuilt`）与 `step35_alias_and_official_split.py` L25/L111。

**复跑实测（本机，2026-09-18）：**

| 量 | 值 | 怎么得到的 |
|---|---|---|
| rebuilt 缓存规模 | 202,200 行 / 65 系 / `len(drug_vocab)`=32,039 | `meta.npz` 直读 |
| rebuilt 观测药物数 | **20,370** | `set(drug_vocab[key])` |
| rebuilt `kind` 取值 | **全部 = 0**（202,200 行） | 直读；故 `pert_uid = key`，无 4,994 偏移 |
| C*（共同细胞系） | **64** | 65 ∩ 125；rebuilt 有 5 个系不在 cache_2021 里 |
| D*（共同药物） | **20,370** | 20,370 ∩ 32,039 = 20,370（rebuilt 的药 100% ⊆ cache_2021） |
| H（留出药物） | **2,037** | `RandomState(0).permutation(20370)[18333:]` |
| n_test（rebuilt 上全部留出行） | **20,031** | 与 `stage35/36` 记录一致 |
| P*（跨源对级交集 ∧ C* ∧ H） | **11,275** | 与 `step55` 记录一致 |

⇒ **五位数字全部逐位复现**，20,370 的出处不再是悬案。

**两点附加事实（前轮结论的补强）：**

1. `2,037 = 10% × 20,370` 成立，但 20,370 **不是**「beta2020 全集的 10% 反向推出来的数」，而是 rebuilt 这个**早期重建缓存**的药物覆盖面（32,039 的一个子集：20,370 ⊂ 32,039）。
2. rebuilt 的 `drug_vocab` 在槽位升序下**恰好是字典序**（实测 `dv[uk] == sorted(set(...))` 逐位 True），所以 `step35/36`（按整数 key 排序）与 `step55`（按药物名字排序）拿到的是**同一个 2,037 留出集**（实测交集 = 2,037）。此处**无隐藏的第二套划分**，前轮的怀疑可撤。

---

## 2. 第二样：15,994 vs 15,157 的出处

**两个数都出自 `results/stage45_official_split_2021.json`，由 `step45_official_split_2021.py` 一次跑出，两个键**（`official` / `ours`）**都是本工程复算。**

| 键名 | n_pert | n_test | 口径（读 `step45_official_split_2021.py`） |
|---|---|---|---|
| `official` | 3,204 | **15,994** | L50-L56：`pert_uid = np.where(kind==0, key, NGENE+key)`；`perts = sorted(set(...))`；`RandomState(0).permutation`；取**最后** 10% |
| `ours` | 3,204 | **15,157** | L59-L64：`u = np.unique(key)`；`default_rng(0).permutation`；取**前** 10% |

**本机逐字复跑（cache_2021，156,931 行 / 32,039 药 / `kind` 全 = 1）：**

```
official: n_pert=32039 held=3204 n_test=15994      ← 与 stage45 一致
ours    : n_pert=32039 held=3204 n_test=15157      ← 与 stage45 一致
两口径测试行重合 = 1398  (Jaccard = 1398/29753 = 0.047)
```

⇒ 它们**不是同一个划分的两个近似**，是两个不同划分（RNG 不同 + 取尾/取头相反），测试行几乎不重叠。

**此处必须撤回一个既有命名：**

- 那段被我们（以及脚本注释、`stage45` 的键名）称作「**官方**口径」的 `RandomState(0)` 写法，它的实际出处在 `G2CP_repo/eval_full.py` L68-74。
- 而 `G2CP_repo/` **不是原文仓库**：该目录下的 `train_g2cp_contrast.py` 自述「G2CP 论文级重训 v4」、`from unipret.compound_encoder import ECFP4_BITS`（`unipret` 是本工程自带包）、`eval_full.py` 自述「全基因模型 g2cp_full_v3 的扰动级留出评测」并读本工程的 `data/g2cp_cache_fullgene`。且 **`eval_full.py` / `train_g2cp_contrast.py` 都不在原文仓库的 81 个文件里**（原文仓库副本 = `external_data/g2cp_code/`，目录只有 `analysis_assets/`、`G2CP/`、`UniPert/`）。
- ⇒ 该写法是「**本工程早期自己的口径**」。**「官方口径」这个标签是错误命名，撤回。** 后文统一称 `inherited_randstate0_last10`（本工程旧口径）。

**顺带一个有利事实**：`beta_all` 与 `cache_2021` 的药物全集**逐字完全相同**（32,039，同序，`np.array_equal` = True），故两版的 `self_constructed_drug_holdout_v1` ID 列表是**同一份**（md5 同为 `f5333fac8ba1d833aa27c8b03b29c308`）。受控换源设计里两臂共用同一留出集，是**优点**（差异只能归因于数据源与覆盖度），但正文必须写明"两臂同划分"，不能写成"两套独立随机划分"。

---

## 3. 第三样：原文官方划分脚本

**找到了，位置与实现：**

- 调用点 `external_data/g2cp_code/G2CP/train_G2CP_cpf.py` L84-89：
  ```python
  pert_data.prepare_split(split=TRAIN_MODE, seed=SEED,
                          train_ptbg_set_frac = FINETUNE_FRAC,
                          pretrain_ptbg_set_frac = PRETRAIN_FRAC,
                          finetune_ptbg_set_frac = FINETUNE_FRAC,
                          val_ptbg_set_frac = VAL_FRAC)
  ```
  `SEED` 默认 0、`VAL_FRAC` 默认 0.1（L22/L25）。
- 实现体 `external_data/g2cp_code/G2CP/backbone_model/GEARS/gears/pertdata.py`，`'cpf'` 分支：
  ```python
  rng_test = np.random.default_rng(seed=self.seed)
  test_ptbgs = rng_test.choice(self.chem_ptbgs,
                               int(len(self.chem_ptbgs) * (1-self.finetune_ptbg_set_frac)),
                               replace=False)
  train_val_ptbgs = np.setdiff1d(self.chem_ptbgs, test_ptbgs)
  val_ptbgs = rng_test.choice(train_val_ptbgs, int(len(train_val_ptbgs)*val_ptbg_set_frac), replace=False)
  ```
  ⇒ **是 `default_rng(seed)` + `choice`，不是 `RandomState(0)` + `permutation`**；且先切 test、再从余下切 val。
  另有 split 落盘缓存机制（`split_file` 存在即直接 load）——若要严格复现某次实验的划分，需拿到当时的 `.pkl`。
- 实验网格 `external_data/g2cp_code/G2CP/run_scripts/run_LINCS_cpf_exps.sh`：
  - 数据集 = **LINCS-A549 / LINCS-A375 / LINCS-MCF7 / LINCS-HT29 / LINCS-PC3**（五个细胞系**各自独立**建数据集）
  - seeds = **1310 / 229 / 52 / 1519 / 564**（**不是 0**）
  - `finetune_fracs` 扫 0.9 → 0.1，`pretrain_fracs` 扫 1.0 → 0.0

**为什么不能套到我们这边：**

1. **划分空间不同**：官方在「单个细胞系 × 该系自己的 perturbagen 列表」上做 90/10；我们的缓存是**跨 125 个细胞系混池**的 `(cell, drug)` 对级矩阵。同一个官方 seed 在我们的池子上没有定义。
2. **seed 不同**：官方用的是一组非 0 seed；`seed=0` 是我们自己的选择。
3. **扰动单位不同**：官方 `chem_ptbgs` 是该系下的化合物列表；我们的 `universe` 是全池唯一 `drug_id`。

⇒ **不存在可用的官方划分脚本。** 按用户预设规则，`split_name` 写 `self_constructed_drug_holdout_v1`，**不静默冒充官方**。此规则已写进 `holdout/<source>/drug_holdout.json` 的 `split_name_basis` 字段，任何人读文件即可看到依据。

---

## 4. 两套对账表（**分两套，不混**）

### 套一 · 受控换源子集（A 臂 = `g2cp_cache_rebuilt`）

| 项 | 值 |
|---|---|
| A_def 缓存 | `g2cp_cache_rebuilt`（GSE92742 Level5 MODZ 重建，202,200 行 / 65 系） |
| C* 细胞系 | 64（rebuilt 65 ∩ cache_2021 125） |
| D* 药物 | 20,370（观测药(rebuilt) ∩ 观测药(cache_2021)） |
| H 留出药物 | 2,037（`RandomState(0).permutation(20,370)[18333:]`） |
| n_test（rebuilt 全行） | 20,031 |
| P* 评测对 | 11,275 |
| 已有读数 | A 0.3551 / B 0.1057 / ΔPCC +0.2494 [0.2440, 0.2553] |
| 用途 | 把「数据源效应」与「覆盖度效应」分离 |

### 套二 · 全集合划分（各版本自己的全药物空间）

| source | 全集 | n_pert | n_test | 口径名 |
|---|---|---|---|---|
| dcic2021 | 32,039 | 3,204 | 15,994 | `inherited_randstate0_last10`（本工程旧口径，**非官方**） |
| dcic2021 | 32,039 | 3,204 | 15,157 | `self_constructed_defaultrng0_first10`（本工程早期自建） |
| dcic2021 | 32,039 | 3,204 | 15,994 | `self_constructed_drug_holdout_v1`（与第一行同实现同数） |
| beta2020 | 32,039 | 3,204 | 16,065 | `self_constructed_drug_holdout_v1` |
| GSE70138 | 1,001 | 100 | 794 | `self_constructed_drug_holdout_v1` |

### 两套之间的桥（写进文件，防混用）

20,370 ⊂ 32,039；`2,037 = 10% × 20,370`，`3,204 = 10% × 32,039`。
⇒ **正文任何位置不得把 2,037 与 3,204 互换，也不得把两套并称「同一个留出集」。**

---

## 5. 复现与留痕

- 复跑脚本：本轮全部为**只读探针**（直读 `meta.npz` / `pairs_meta.npz` + 纯 numpy 重算），未改动任何缓存。
- 取证读到的代码文件与行号：见 §1–§3，逐条给了路径 + 行号 + 代码片段。
- 被改动的产物（**有备份，未删任何东西**）：
  - `_scratch/v31_mvpa/11_build_holdout.py`（加 `split_name` / `split_name_basis` / `frozen: false` / `provenance`；改准 beta2020 与 dcic2021 的 `notes`）
    - 备份：`11_build_holdout.py.bak_pre_prov_20260918`
  - `_scratch/v31_mvpa/holdout/**/drug_holdout.json`（由上述脚本重跑生成）
    - 备份：`holdout.bak_pre_prov_20260918/`
- **ID 列表零漂移**（判据：重跑前后 `drug_holdout.ids.txt` 的 md5 逐条相同）

  | source | md5 | 状态 |
  |---|---|---|
  | beta2020 | `f5333fac8ba1d833aa27c8b03b29c308` | 未变 |
  | dcic2021 | `f5333fac8ba1d833aa27c8b03b29c308` | 未变 |
  | GSE70138 | `31545873e01b9c5b27fb3900241d922d` | 未变 |

- `frozen` 一律 `false`：按用户指令，步 3 的 `holdout.json` **先不冻结**。
