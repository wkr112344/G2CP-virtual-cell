# 研究提案：二甲双胍经脂肪酸合成抑制下调 mTORC1 信号的机制研究

**——基于 G2CP 虚拟细胞模型的预测与验证**

*提案类型：本科科研 / 基础实验研究*
*关联工具：G2CP 虚拟细胞模型（v7，162 细胞系全基因版）*
*日期：2026-08-06*

---

## 一、研究背景

非酒精性脂肪性肝病（NAFLD，现称代谢相关脂肪性肝病 MASLD）已成为全球最常见的慢性肝病，与肥胖、2 型糖尿病密切相关。二甲双胍（Metformin）作为全球处方量最大的降糖药物，其对肝脏脂质代谢的影响备受关注，但具体分子机制仍存在争议。

已有共识指出二甲双胍的核心机制是**激活 AMPK**、抑制线粒体复合物 I、进而抑制 **mTORC1 信号通路**；同时多项研究提示二甲双胍可减少肝细胞脂滴沉积。然而，**二甲双胍的"降脂效应"是否部分经由抑制脂肪酸合成（FA synthesis）通路介导、进而下调 mTORC1 信号**，尚缺乏直接证据。脂肪酸合酶（FASN）是脂肪酸从头合成的关键限速酶，其产物（棕榈酸等）与 mTORC1 激活密切相关，提示"FASN—脂质信号—mTORC1"存在潜在的因果链条。

## 二、计算预测（G2CP 虚拟细胞模型）

利用本课题组的 G2CP 全基因虚拟细胞模型（训练数据：LINCS L1000 32,039 药 × 162 细胞系 + DepMap CRISPR 基因扰动，与 UniPert-G2CP 论文同架构），在肝癌细胞系 HEPG2 上分别模拟**二甲双胍给药**与**FASN 基因敲除**，得到转录组级效应预测（12,328 基因）。

**结果一：二甲双胍 × HEPG2 预测下调基因（节选）**

| 基因 | 功能注释 | 与已知机制一致性 |
|---|---|---|
| SLC7A5（LAT1） | 氨基酸转运体，mTORC1 通路关键上游 | ✓ 二甲双胍→AMPK→抑制 mTORC1→SLC7A5 下调 |
| PLIN2 | 脂滴包被蛋白，脂质储存核心 | ✓ 脂滴减少 |
| MPC2 | 线粒体丙酮酸载体 | ✓ 线粒体代谢抑制 |
| CCNB2 | 细胞周期蛋白 B2 | ✓ 抗增殖效应 |
| GNAS | G 蛋白 α 亚基 | ✓ 代谢信号 |

**结果二：FASN 敲除 × HEPG2 预测下调基因（节选）**

| 基因 | 功能注释 |
|---|---|
| SLC7A5（LAT1） | 氨基酸转运体 / mTORC1 |
| CCNB2 | 细胞周期 |
| GNAS | 代谢信号 |
| IARS2 | 线粒体翻译 |

**结果三（核心发现）：两组预测的下调基因存在显著重叠（≥4 个，含 mTORC1 关键靶基因 SLC7A5）**

> 二甲双胍的预测转录组效应 ≈ FASN 敲除的预测效应
> → 提示：**二甲双胍对肝细胞的代谢/抗增殖效应，可能部分通过"抑制脂肪酸合成 → 脂质信号减弱 → mTORC1 信号下调（SLC7A5）"这一链条介导。**

## 三、科学假说

> **H：二甲双胍通过抑制脂肪酸合成（FASN 依赖通路），下调肝细胞 mTORC1 信号及下游靶基因 SLC7A5 的表达，从而发挥降脂与抗增殖效应。**

推论（可分别检验）：
- H1：二甲双胍处理后，HEPG2 细胞 SLC7A5 mRNA 表达显著下调；
- H2：siRNA 敲低 FASN 可复现二甲双胍对 SLC7A5 的下调（模拟二甲双胍的"降脂→mTOR"效应）；
- H3：外源性棕榈酸（FASN 产物）可部分挽救二甲双胍导致的 SLC7A5 下调（因果性检验）。

## 四、研究目标

1. 验证二甲双胍对 HEPG2 细胞脂滴含量与 SLC7A5/mTORC1 信号的影响；
2. 检验 FASN 敲低是否模拟二甲双胍的转录组效应（重点 SLC7A5）；
3. 探索脂肪酸补充对二甲双胍效应的拮抗作用，确证"脂质信号—mTORC1"因果链。

## 五、实验设计

### 5.1 细胞与处理
- 细胞：HepG2（人肝癌细胞系，ATCC HB-8065），含 10% FBS 的 DMEM 培养。
- 分组：
  - 对照组（PBS/溶剂）
  - 二甲双胍组（5 mM 与 10 mM，24 h / 48 h）
  - siFASN 组（siRNA 敲低 FASN，48 h）
  - siFASN + 二甲双胍组（联合处理）
  - 二甲双胍 + 棕榈酸组（100–200 µM，palmitate-BSA 偶联物）

### 5.2 检测指标
| 层次 | 指标 | 方法 |
|---|---|---|
| 表型 | 脂滴含量 | 油红 O（Oil Red O）染色 + 定量 |
| mRNA | SLC7A5、PLIN2、FASN、CCNB2、GNAS | RT-qPCR（GAPDH 内参） |
| 蛋白 | FASN、p-AMPK/AMPK、p-S6/S6（mTORC1 下游） | Western blot |
| 功能 | 细胞增殖/活力 | CCK-8 |

### 5.3 关键对照逻辑
- siFASN 组用于回答：**FASN 敲低是否复现二甲双胍的 SLC7A5 下调**（H2）；
- 棕榈酸回补组用于回答：**补充脂肪酸能否挽救二甲双胍的效应**（H3，因果性）；
- 若 H2、H3 均成立，则"二甲双胍→FASN/脂肪酸合成→mTORC1→SLC7A5"因果链得到实验支持。

## 六、预期结果与解读

1. **预期 1**：二甲双胍降低 HepG2 脂滴含量，SLC7A5 表达下调 → 支持模型预测；
2. **预期 2**：siFASN 复现 SLC7A5 下调，且与二甲双胍无显著叠加 → 支持"二甲双胍效应部分经 FASN 通路"；
3. **预期 3**：棕榈酸回补部分挽救 SLC7A5 下调 → 支持"脂质信号—mTORC1"因果链。

**如果预期全部达成**：可得出结论——二甲双胍抑制肝细胞脂质从头合成并下调 mTORC1 信号，为二甲双胍治疗 MASLD/肝细胞脂毒性提供新的机制依据，并可申请进一步动物实验（高脂饮食小鼠模型）。

## 七、创新点与意义

1. **计算先行、实验验证**：采用虚拟细胞模型先做全基因组扰动预测，再以最小实验集验证核心假说，降低试错成本；
2. **机制新角度**：首次系统提出并检验"二甲双胍—FASN/脂质合成—mTORC1—SLC7A5"链条（检索未见直接报道）；
3. **临床转化价值**：为二甲双胍在 MASLD 人群的个体化应用提供分子依据。

## 八、可行性

- 所需设备均为常规细胞学实验配置（细胞培养、qPCR、WB、显微镜）；
- 周期估计：2–3 个月（含重复与数据分析）；
- 风险控制：若 SLC7A5 在 HepG2 基础表达较低，可改测 PLIN2/FASN 及 p-S6 蛋白作为主要读出。

## 九、主要参考文献

1. Zhou G, Myers R, Li Y, et al. Role of AMP-activated protein kinase in mechanism of metformin action. *J Clin Invest*. 2001;108(8):1167-1174.
2. Rena G, Hardie DG, Pearson ER. The mechanisms of action of metformin. *Diabetologia*. 2017;60(9):1577-1585.
3. Hardie DG, Ross FA, Hawley SA. AMPK: a nutrient and energy sensor that maintains energy homeostasis. *Nat Rev Mol Cell Biol*. 2012;13(4):251-262.
4. Nicklin P, Bergman P, Zhang B, et al. Bidirectional transport of amino acids regulates mTOR and autophagy. *Cell*. 2009;136(3):521-534.
5. Menendez JA, Lupu R. Fatty acid synthase and the lipogenic phenotype in cancer pathogenesis. *Nat Rev Cancer*. 2007;7(10):763-777.
6. Foretz M, Guigas B, Bertrand L, et al. Metformin: from mechanisms of action to therapies. *Cell Metab*. 2014;20(6):953-966.
7. Li Y, et al. UniPert-G2CP bridges genetic and chemical screens from molecular representation to phenotype. *Cell*. 2026. doi:10.1016/j.cell.2026.06.005.

---

*附注：本提案中的计算预测来自 G2CP v7 虚拟细胞模型（SMD 论文口径 1.302，为论文 UniPert 的 70.4%；EF 富集因子为随机 139 倍）。预测结果为计算建议，最终结论以实验验证为准。*
