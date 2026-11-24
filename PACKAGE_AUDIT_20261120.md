# 复现包核对记录（2026-11-20）

> **状态（2026-11-24 更新）**：本文件保留 11-20 当时的核对记录，第五节列出的五项遗留问题均已处理，见文末
> 「7. 遗留问题状态」。当前仍然存在的缺口只有四项，也列在该节，不要与已解决的历史条目混淆。

对象：`C:\Users\wkr20\WorkBuddy\Claw\GigaScience_reproducibility_release`
方法：以 `MANIFEST.csv` 逐条对照磁盘文件；补充表编号对照 11-20 稿正文引用；与 10-13 案卷 `ALIGNMENT_v3_20261113.md` 的三条诊断逐项比对。

## 1. 核对前状态

| 项 | 数值 |
|---|---|
| MANIFEST 行数 | 349（github 96 / zenodo 199，另有 3 条 not_included、51 条 skipped-duplicate） |
| 磁盘文件 | github 91 / zenodo 199 |
| 补充表 | 51 个，编号止于 S26b |
| S27–S30 | **缺失**（只有脚本，无输出 CSV；MANIFEST 无条目） |

## 2. 本次补齐

**输出表（6 个，从 10-13 的 aligned 包移入 `zenodo/supplementary/`）**

| 文件 | 对应结论 |
|---|---|
| `S27_sdst_beta2020_condition_audit.csv` | SDST × beta2020 条件/孔位核查 |
| `S28_xpert_transigen_matched_summary.csv` | 表 4 六个正常匹配值 |
| `S28_permutation_scheme_comparison.csv` | 错配方案 A/B 双方案 12 行 |
| `S28_matched_vs_mispaired_pearson_historical_schemeA.csv` | 历史方案 A（已被上一行取代） |
| `S29_paired_advantage_bootstrap.csv` | 1,343 药物配对优势 bootstrap（0.01453 / −0.00298） |
| `S30_lincs_model_comparison_practices.csv` | LINCS 模型比较实践调查 |

**重建脚本（6 个，已在上一提交进入版本控制，本次补入 MANIFEST）**

`github/analysis/supplementary_rebuild_S23_S28/`：README、build_S25_gene_counts、build_S27_condition_audit、build_S28_permutation、build_S29_paired_bootstrap、config_paths。

## 3. 核对后状态

| 项 | 数值 |
|---|---|
| MANIFEST 行数 | 361（github 102 / zenodo 205） |
| 计入校验条目 | 307 |
| 缺失文件 | **0** |
| 磁盘有、清单无 | **0** |
| zenodo/supplementary | 57 个文件，S1–S30 齐 |
| 包体积 | 约 815 MB |

## 4. 与 10-13 案卷的对照

`ALIGNMENT_v3_20261113.md` 当时提出三条：

1. 复现包有脚本、无 S27–S30 输出 CSV，MANIFEST 也没有条目 → **本次已解决**
2. 上传版与交付包停在 S26b，比六表稿旧一代 → 未处理（不影响 GitHub 复现包）
3. aligned 包最全但不在发布流程里 → 内容已并入本仓库

## 5. 遗留问题

**5.1 论文引用的 S18 在包内不存在**

11-20 稿 Analyses 段：「…fell within the beta2020 bootstrap 95% empirical ranges… (Supplementary Tables S3, S17, **S18**, and S24)」。
包内无 S18，最接近的是 `S17b_coverage_ge3_rank_stability.csv`（覆盖度 ≥3 的 rank stability）与 `S22_topk_random_expectation.csv`（随机期望）。
需确认是改稿内编号，还是在包内补一个 S18 命名文件。**未擅自改动。**

**5.2 历史编号仍在包内**

`S17b`、`S17c`、`S26b` 三个带字母后缀的表仍在包中，而 11-20 稿正文已不再引用（各 0 次）。保留无害，但若追求编号整洁可加一条 note 说明其为历史编号。

**5.3 S20 已不被引用**

11-20 稿正文 S20 出现 0 次，包内亦无 S20，无需处理。

**5.4 仓库名与论文引用不一致**

- git remote：`https://github.com/wkr112344/VirtualCellWorkbench.git`
- 论文文献 [27]：`https://github.com/wkr112344/G2CP-virtual-cell`

两者需统一，否则审稿人点开 [27] 会 404。**未擅自改 remote。**

## 6. 未执行的动作

- 未推送 GitHub（网络不通：git 配置代理 `127.0.0.1:26561` 无监听；直连被 reset）
- 未改动任何论文正文
- 未删除任何历史文件

## 7. 遗留问题状态（2026-11-24）

### 五项历史条目：全部已解决

| 原条目 | 状态 | 依据 |
|---|---|---|
| 5.1 论文引用的 S18 在包内不存在 | 已解决 | 投稿包现有 `S18_rank_concordance_summary.csv`（跨矩阵等级一致性，Spearman 0.5675 / Kendall τ 0.4110）；旧编号 `S18_coverage_ge3_rank_stability.csv` 与 `S17b` 内容完全相同，属重复表，映射见 `REPRODUCTION_MAP.md` |
| 5.2 S17b / S17c / S26b 历史编号仍在包内 | 已解决 | 改为在 `REPRODUCTION_MAP.md` 明确映射（S17b、S17c 保留在 `detail/`；S26b 对应 `detail/S23b_*`），不作为遗留问题 |
| 5.3 S20 已不被引用 | 已解决 | 投稿包现有 `S20_lincs_perdrug_raw_vs_residualized.csv`，正文 Table 6 相关引用已对齐 |
| 5.4 仓库名与论文 [27] 不一致 | 已解决 | remote 现为 `https://github.com/wkr112344/G2CP-virtual-cell.git`，与正文 [27] 一致 |
| 6 未推送 GitHub | 已解决 | 已推送至 `master-with-repro`，本文件所在提交即公开快照 |

### 当前仍然存在的四项缺口

1. **`X_TPM.npy` 不随包发布**（908 × 19,177 表达矩阵）。已提供确定性重建脚本
   `github/analysis/rebuild_X_TPM.py`：按 `cell_order.txt` 取行、列序即基因轴、float32，输出 sha256。
   重建是否正确的判据是重跑正对照后 S13–S15 是否复现（脚本会打印形状与哈希）。`run_all.sh` 在该文件缺失时
   跳过此步并给出提示，其余步骤照常完成。
2. **S28 置换表的生成脚本不在仓库内**。`build_S28_v2.py` 是当时的驱动脚本，未随包提供，因此
   `S28_*` 只能从 CSV 核对（观测值、零分布均值、区间、经验 P 值、记录数），不能在仓库内端到端重跑。
   已在 `REPRODUCTION_MAP.md` 就地标注。
3. **GTEx 原始 per-sample 矩阵未随包发布**（两个 5.9 GB 下载）。`06_score_2x2.py` 因此无法从零重跑；
   它产出的 `results/per_sample_scores.csv` 已归档，`run_all.sh` 用该中间文件跑 `07_bootstrap.py`，
   并由第 1 步重生成 S8、S10–S12。
4. **`github/analysis/` 下部分探索性脚本仍保留作者机上的绝对路径常量**（约 30 个，不在 `run_all.sh`
   调用链内）。正式复现入口 `run_all.sh` 及其调用的脚本已全部改为仓库相对路径，并已在本地完整跑通
   （74 项校验 0 不符）。

### 2026-11-24 实跑记录

- `run_all.sh` 在本机跑通：第 1 步重生成 S5/S6/S8/S10–S12，与归档逐值相同（最大绝对差 0）；
  第 4 步 `07_bootstrap.py` 复现 interaction Pearson 0.070957、Spearman 0.061869；
  第 6 步 `compared 74 checks, 0 mismatched`。
- 第 2、3、4 步各有一处显式跳过（原始 DepMap pkg、X_TPM、GTEx 原始矩阵），原因同上，均在日志中写明。
