# 复现包核对记录（2026-11-20）

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
