/* =====================================================================
 * 论文级通路互作模板（P6 · 虚拟细胞工作台）
 * 每个通路：节点（蛋白：功能/结构域/修饰位点/互作）+ 边（机制级）
 *           + 药物介入点 + 表型输出
 * 机制信息参考 UniProt / 经典信号转导教材（Alberts / 药理机制），
 * 关键修饰位点（磷酸化残基等）为文献公认。
 * ===================================================================== */
const PATHWAYS = {
  pka: {
    name: "cAMP-PKA 信号通路",
    summary: "激素/神经递质 → 受体 → Gs → 腺苷酸环化酶 → cAMP → PKA → 转录因子，是调控代谢、增殖、心肌收缩的经典通路。",
    layers: ["GPCR / 受体", "第二信使", "激酶级联", "转录因子", "基因表达"],
    nodes: [
      { id: "adrb2", symbol: "ADRB2", name: "β₂-肾上腺素受体", type: "GPCR",
        function: "肾上腺素/去甲肾上腺素的 G 蛋白偶联受体；激动剂结合后胞内环暴露，偶联 Gs 异源三聚体。",
        domains: ["7 次跨膜螺旋", "胞外 N 端", "胞内 C 端（PKA/GRK 磷酸化位点）"],
        sites: ["Ser261/262/345/346：PKA 磷酸化 → 受体脱敏", "Tyr141：GRK 磷酸化 → β-arrestin 内吞"],
        interactions: ["与 Gsα 偶联（GDP/GTP 交换）", "被 β-arrestin 脱敏"] },
      { id: "gnas", symbol: "GNAS", name: "Gsα（刺激性 G 蛋白 α 亚基）", type: "G 蛋白",
        function: "受体激活后 GDP→GTP 交换，Gsα-GTP 结合并激活腺苷酸环化酶；GTP 水解后失活。",
        domains: ["GTPase 域", "螺旋域"],
        sites: ["Arg201：霍乱毒素 ADP-核糖基化 → GTPase 失活 → 组成性激活（致癌突变热点）"],
        interactions: ["被 ADRB2 激活", "激活腺苷酸环化酶", "GTP 酶活由 RGS 蛋白加速"] },
      { id: "adcy", symbol: "ADCY1/5/6", name: "腺苷酸环化酶", type: "酶",
        function: "催化 ATP → cAMP + PPi；是 cAMP 生成的限速酶。",
        domains: ["12 次跨膜", "两个胞内催化域 C1/C2"],
        sites: ["Ser：PKA 反馈磷酸化"],
        interactions: ["被 Gsα-GTP 激活", "被 Gi 抑制", "Forskolin 直接结合 C1/C2 催化域激活"] },
      { id: "camp", symbol: "cAMP", name: "cAMP（环磷酸腺苷）", type: "第二信使",
        function: "由腺苷酸环化酶生成、磷酸二酯酶（PDE）水解为 AMP 降解；浓度由合成/降解动态平衡决定。",
        domains: ["核苷酸环结构"],
        sites: ["无（小分子第二信使）"],
        interactions: ["结合 PKA 调节亚基 CNB 域 → 激活 PKA", "结合 EPAC → 激活 Rap1", "结合 HCN 通道"] },
      { id: "prkar", symbol: "PRKAR2A", name: "PKA 调节亚基 RIIα", type: "激酶调节亚基",
        function: "无 cAMP 时与催化亚基结合并抑制其活性；cAMP 结合后构象打开释放催化亚基。",
        domains: ["N 端二聚化域（D/D 域）", "CNB-A（环核苷酸结合域 A）", "CNB-B"],
        sites: ["Ser99：自磷酸化"],
        interactions: ["cAMP 结合 CNB-A/B → 亲和力骤降 → 与 C 亚基解离"] },
      { id: "prkaca", symbol: "PRKACA", name: "PKA 催化亚基 Cα", type: "丝/苏氨酸激酶",
        function: "被调节亚基抑制；cAMP 使调节亚基解离后激活，磷酸化下游底物（CREB、糖原合酶激酶等）。",
        domains: ["N 端豆蔻酰化", "ATP 结合口袋", "催化环（含 DFG motif）"],
        sites: ["Thr197：自磷酸化（激活必需）", "Ser338：激活环磷酸化"],
        interactions: ["磷酸化 CREB Ser133", "磷酸化糖原合酶激酶-3", "磷酸化 IP3 受体"] },
      { id: "creb1", symbol: "CREB1", name: "CREB（cAMP 反应元件结合蛋白）", type: "转录因子",
        function: "结合 DNA 的 CRE 元件（TGACGTCA），磷酸化后招募 CBP/p300 乙酰化组蛋白，激活转录。",
        domains: ["bZIP DNA 结合域", "KID（激酶诱导域）", "Q2 谷氨酰胺富集域"],
        sites: ["Ser133：PKA 磷酸化（关键激活位点，'激活开关'）", "Ser142/143：抑制性磷酸化"],
        interactions: ["pSer133 招募 CBP/p300 → 组蛋白 H3 乙酰化 → 染色质开放"] },
      { id: "target_genes", symbol: "FOS · JUN · CCND1", name: "cAMP 反应基因", type: "基因表达",
        function: "CREB 结合 CRE 后激活的靶基因：早期反应基因（FOS、JUN）与细胞周期基因（CCND1），驱动增殖/代谢。",
        domains: ["CRE 启动子元件"],
        sites: ["—"],
        interactions: ["FOS/JUN 上调", "CCND1 上调 → 细胞周期推进"] }
    ],
    edges: [
      { from: "adrb2", to: "gnas", type: "激活", label: "偶联 / GDP→GTP",
        mech: "激动剂结合 → 受体胞内环构象变化 → Gs 三聚体结合，GDP 释放、GTP 结合 Gsα" },
      { from: "gnas", to: "adcy", type: "激活", label: "GTP 依赖",
        mech: "Gsα-GTP 结合腺苷酸环化酶 C1/C2 催化域 → 稳定催化构象 → 酶激活" },
      { from: "adcy", to: "camp", type: "催化", label: "ATP → cAMP",
        mech: "腺苷酸环化酶催化 ATP 5'-3' 环化生成 cAMP + PPi" },
      { from: "camp", to: "prkar", type: "结合", label: "CNB-A/B 域",
        mech: "两个 cAMP 分子分别结合 RIIα 的 CNB-A 与 CNB-B 域 → 调节亚基'打开'，对催化亚基亲和力骤降 → 解离" },
      { from: "prkar", to: "prkaca", type: "抑制解除", label: "亚基解离 → 激活",
        mech: "无 cAMP：R 亚基假底物序列占据 C 亚基活性位点（抑制）；cAMP 结合后 R 亚基解离 → Cα 活性位点暴露并自磷酸化 Thr197 激活" },
      { from: "prkaca", to: "creb1", type: "磷酸化", label: "Ser133",
        mech: "Cα 转位入核，磷酸化 CREB KID 域的 Ser133，产生 CBP/p300 结合位点（pKID）" },
      { from: "creb1", to: "target_genes", type: "激活", label: "CRE 元件 + 乙酰化",
        mech: "pSer133-CREB 二聚化结合 CRE，招募 CBP/p300（组蛋白乙酰转移酶）→ 组蛋白 H3/H4 乙酰化 → 染色质开放 → 转录激活" }
    ],
    drugs: [
      { cls: "腺苷酸环化酶激活剂", ex: "Forskolin", mech: "直接结合 AC 催化域 C1/C2，绕过受体强制升高 cAMP → PKA 持续激活" },
      { cls: "磷酸二酯酶抑制剂", ex: "咖啡因 / 西地那非 / 茶碱", mech: "抑制 PDE 水解 → cAMP 降解减慢、蓄积 → PKA 信号增强（西地那非主要用于 PDE5→cGMP）" },
      { cls: "β 受体激动剂 / 拮抗剂", ex: "沙丁胺醇 / 普萘洛尔", mech: "作用于 ADRB2 增强或阻断 Gs 偶联，从而调控 cAMP 生成（沙丁胺醇：支气管舒张；普萘洛尔：负性肌力）" }
    ],
    phenotype: "代谢增强（糖原分解、脂肪动员）、心肌正性肌力/正性频率、神经元可塑性；FOS/JUN/CCND1 上调驱动增殖；长期过度激活与心力衰竭、肿瘤相关。"
  },

  mapk: {
    name: "EGFR-RAS-MAPK 通路",
    summary: "生长因子 → 受体酪氨酸激酶 → RAS → RAF → MEK → ERK → 转录，是细胞增殖/分化的主通路，肿瘤药核心靶区。",
    layers: ["受体酪氨酸激酶", "接头蛋白", "小 G 蛋白", "激酶级联", "转录因子"],
    nodes: [
      { id: "egfr", symbol: "EGFR/ERBB1", name: "表皮生长因子受体", type: "受体酪氨酸激酶",
        function: "EGF 结合诱导二聚化与酪氨酸自磷酸化，招募下游接头蛋白；过表达/突变驱动多种肿瘤。",
        domains: ["胞外配体结合域（I-IV）", "跨膜域", "胞内酪氨酸激酶域", "C 端调节域"],
        sites: ["Tyr1068：GRB2 结合位点", "Tyr1173：Shc 结合位点", "Tyr992：PLCγ 结合位点"],
        interactions: ["配体诱导同/异二聚化", "自磷酸化后招募 GRB2/SHC"] },
      { id: "grb2", symbol: "GRB2", name: "GRB2（生长因子受体结合蛋白 2）", type: "接头蛋白",
        function: "SH2 域结合受体磷酸化酪氨酸，SH3 域招募 SOS 至膜。",
        domains: ["SH2", "SH3-SH3"],
        interactions: ["SH2 结合 EGFR pTyr", "SH3 结合 SOS 脯氨酸富集区"] },
      { id: "sos1", symbol: "SOS1", name: "SOS（鸟苷酸交换因子）", type: "GEF",
        function: "催化 RAS 的 GDP→GTP 交换，激活 RAS。",
        interactions: ["被 GRB2 招募至膜 → 与膜上 RAS 相遇 → 促 GDP/GTP 交换"] },
      { id: "kras", symbol: "KRAS", name: "KRAS", type: "小 G 蛋白",
        function: "GDP/GTP 分子开关；GTP 结合时激活 RAF。G12/G13/Q61 突变是胰腺癌/肺癌最常见致癌驱动。",
        domains: ["GTPase 域", "开关 I/II 区"],
        sites: ["Gly12/Gly13/Gln61：突变热点（组成性激活，失去 GTP 水解）"],
        interactions: ["被 SOS 激活", "激活 RAF（结合 RAF 的 RBD 域）", "被 GAP（NF1）失活"] },
      { id: "braf", symbol: "BRAF", name: "BRAF", type: "丝/苏氨酸激酶",
        function: "RAS 下游第一级激酶；V600E 突变（黑色素瘤）持续激活。",
        domains: ["RBD（RAS 结合域）", "C 端激酶域"],
        sites: ["Ser338：磷酸化激活（同源 RAF）", "V600E：激活环突变（持续激活）"],
        interactions: ["被 RAS-GTP 招募并二聚化激活", "磷酸化 MEK1/2 激活环"] },
      { id: "mek1", symbol: "MAP2K1/MEK1", name: "MEK1/2", type: "双特异性激酶",
        function: "唯一天然底物为 ERK1/2；磷酸化其 TXY 基序。",
        domains: ["N 端负调控域", "催化域"],
        sites: ["Ser218/Ser222：被 RAF 磷酸化激活"],
        interactions: ["被 RAF 激活", "磷酸化 ERK1/2 的 Thr-Glu-Tyr"] },
      { id: "erk1", symbol: "MAPK1/ERK2", name: "ERK1/2", type: "丝/苏氨酸激酶",
        function: "入核磷酸化转录因子（ELK1、FOS），也磷酸化胞质底物（RSK）。",
        domains: ["催化域", "MAPK 插入域"],
        sites: ["Thr202/Tyr204（ERK1）/ Thr185/Tyr187（ERK2）：MEK 双磷酸化（TXY motif）——激活标志"],
        interactions: ["磷酸化 ELK1（Ser383）", "磷酸化 RSK", "磷酸化 FOS（Thr232）"] },
      { id: "elk1", symbol: "ELK1", name: "ELK1（三元复合因子）", type: "转录因子",
        function: "与 SRF 协同结合 SRE 元件，ERK 磷酸化后激活 FOS 转录。",
        domains: ["ETS DNA 结合域", "C 端转录激活域"],
        sites: ["Ser383/389：ERK 磷酸化 → 转录激活"],
        interactions: ["与 SRF 形成三元复合物结合 SRE", "激活 FOS 启动子"] },
      { id: "target_genes", symbol: "FOS · MYC · CCND1", name: "增殖基因", type: "基因表达",
        function: "FOS、MYC、CCND1 上调 → G1/S 期推进、增殖；持续激活与癌变相关。",
        interactions: ["FOS/MYC/CCND1 转录上调"] }
    ],
    edges: [
      { from: "egfr", to: "grb2", type: "结合", label: "SH2-pTyr",
        mech: "EGF 诱导 EGFR 二聚化 → C 端 Tyr1068 自磷酸化 → GRB2 SH2 域结合" },
      { from: "grb2", to: "sos1", type: "招募", label: "SH3 结合",
        mech: "GRB2 的 SH3 域结合 SOS 的脯氨酸富集区，把 SOS 带到质膜" },
      { from: "sos1", to: "kras", type: "激活", label: "GDP→GTP",
        mech: "SOS 催化 KRAS 释放 GDP、结合 GTP（GEF 活性）" },
      { from: "kras", to: "braf", type: "激活", label: "RBD 结合",
        mech: "KRAS-GTP 结合 BRAF 的 RBD 域，招募 BRAF 至膜并诱导二聚化激活" },
      { from: "braf", to: "mek1", type: "磷酸化", label: "Ser218/222",
        mech: "BRAF 磷酸化 MEK1/2 激活环的 Ser218/Ser222" },
      { from: "mek1", to: "erk1", type: "磷酸化", label: "TXY 双磷酸化",
        mech: "MEK 磷酸化 ERK 的 Thr-Glu-Tyr 基序（Thr202/Tyr204），ERK 激活" },
      { from: "erk1", to: "elk1", type: "磷酸化", label: "Ser383",
        mech: "激活的 ERK 入核，磷酸化 ELK1 C 端 Ser383/389" },
      { from: "elk1", to: "target_genes", type: "激活", label: "SRE 元件",
        mech: "pELK1 与 SRF 结合 SRE → 启动 FOS 等基因转录" }
    ],
    drugs: [
      { cls: "EGFR 酪氨酸激酶抑制剂", ex: "厄洛替尼 / 吉非替尼 / 奥希替尼", mech: "竞争结合 EGFR ATP 口袋（T790M 突变逃逸 → 三代药奥希替尼共价结合 Cys797）→ 阻断自磷酸化与下游级联" },
      { cls: "BRAF 抑制剂", ex: "维莫非尼 / 达拉非尼", mech: "选择性抑制 BRAF-V600E ATP 口袋（注意：对野生型 BRAF 反而可反常激活 MEK）" },
      { cls: "MEK 抑制剂", ex: "曲美替尼 / 司美替尼", mech: "结合 MEK1/2 别构口袋（区别于 ATP 竞争）→ 阻断 ERK 磷酸化" },
      { cls: "RAS 抑制剂", ex: "Sotorasib（KRAS-G12C）", mech: "共价结合 KRAS-G12C 的半胱氨酸（Cys12），锁定在 GDP 失活构象" }
    ],
    phenotype: "增殖、分化、存活增强；异常激活 → 肿瘤（EGFR/KRAS/BRAF 突变型肺癌、结直肠癌、黑色素瘤）；被靶向药阻断后增殖停滞、凋亡。"
  },

  pi3k: {
    name: "PI3K-AKT-mTOR 通路",
    summary: "RTK → PI3K → PIP3 → AKT → mTORC1，调控生长、存活、代谢，肿瘤与免疫治疗核心通路。",
    layers: ["受体酪氨酸激酶", "脂质激酶", "第二信使脂质", "丝/苏氨酸激酶", "代谢/翻译"],
    nodes: [
      { id: "rtk", symbol: "IGF1R/INSR", name: "RTK（IGF-1R / 胰岛素受体）", type: "受体酪氨酸激酶",
        function: "配体结合后自身磷酸化，为 PI3K 的 p85 调节亚基提供对接位点。",
        sites: ["Tyr 磷酸化（pYXXM 基序）：p85 结合位点"],
        interactions: ["p85 SH2 结合受体 pYXXM"] },
      { id: "pi3k", symbol: "PIK3CA/PIK3R1", name: "PI3K（p110α/p85）", type: "脂质激酶",
        function: "催化 PIP2 → PIP3（磷脂酰肌醇 3 位磷酸化）。PIK3CA 突变是肿瘤最常见激酶突变。",
        domains: ["p110：催化域/螺旋域", "p85：SH2 调节域"],
        sites: ["p110α E542K/E545K/H1047R：激活突变热点"],
        interactions: ["被 RTK 招募激活", "被 RAS 直接激活", "被 PTEN 反向去磷酸化"] },
      { id: "pip3", symbol: "PIP3", name: "PIP3（磷脂酰肌醇-3,4,5-三磷酸）", type: "膜脂质第二信使",
        function: "富集于质膜内侧，为 PH 域蛋白（AKT/PDK1）提供锚定位点。",
        interactions: ["AKT 的 PH 域结合 PIP3 → 转位至膜", "被 PTEN 去磷酸化为 PIP2（负调控）"] },
      { id: "akt", symbol: "AKT1/PKB", name: "AKT", type: "丝/苏氨酸激酶",
        function: "细胞存活主激酶：抑制凋亡（BAD、FOXO）、激活 mTORC1。",
        domains: ["PH 域", "催化域", "C 端调节域"],
        sites: ["Thr308：PDK1 磷酸化（膜上）", "Ser473：mTORC2 磷酸化（完全激活）"],
        interactions: ["被 PDK1（Thr308）与 mTORC2（Ser473）双磷酸化", "磷酸化 BAD/FOXO/GSK3β 抑制凋亡"] },
      { id: "mtorc1", symbol: "MTOR", name: "mTORC1", type: "激酶复合物",
        function: "营养/生长信号主传感器，磷酸化 S6K1 与 4E-BP1 促进翻译；被雷帕霉素抑制。",
        domains: ["FRB 域（雷帕霉素-FKBP12 结合位点）", "激酶域"],
        sites: ["Ser2448：AKT 底物位点"],
        interactions: ["被 AKT 经 TSC1/2-Rheb 激活", "磷酸化 S6K1（Thr389）与 4E-BP1"] },
      { id: "target_genes", symbol: "MYC · CCND1 · VEGFA", name: "生长/代谢程序", type: "基因表达",
        function: "AKT/mTOR 驱动蛋白合成、糖酵解、血管生成（HIF1α/VEGFA）、细胞周期。",
        interactions: ["S6K1 → 核糖体蛋白 S6 磷酸化 → 翻译增强", "4E-BP1 磷酸化 → eIF4E 释放 → 帽依赖翻译"] }
    ],
    edges: [
      { from: "rtk", to: "pi3k", type: "激活", label: "p85 SH2 结合",
        mech: "受体 pYXXM 基序被 p85 SH2 域结合 → p110 催化亚基转位至膜" },
      { from: "pi3k", to: "pip3", type: "催化", label: "PIP2→PIP3",
        mech: "p110 催化 PIP2 的肌醇环 3 位磷酸化生成 PIP3" },
      { from: "pip3", to: "akt", type: "招募", label: "PH 域结合",
        mech: "PIP3 富集处 AKT 经 PH 域结合 → 转位至质膜 → Thr308/Ser473 双磷酸化" },
      { from: "akt", to: "mtorc1", type: "激活", label: "TSC2 磷酸化",
        mech: "AKT 磷酸化 TSC2（抑制其 GAP 活性）→ Rheb-GTP 蓄积 → 激活 mTORC1" },
      { from: "mtorc1", to: "target_genes", type: "激活", label: "S6K1/4E-BP1",
        mech: "mTORC1 磷酸化 S6K1（Thr389）增强翻译；磷酸化 4E-BP1 释放 eIF4E → 帽依赖蛋白合成" }
    ],
    drugs: [
      { cls: "PI3K 抑制剂", ex: "Alpelisib（BYL719）", mech: "选择性抑制 p110α ATP 口袋，用于 PIK3CA 突变乳腺癌" },
      { cls: "AKT 抑制剂", ex: "Capivasertib", mech: "ATP 竞争性抑制 AKT 催化域，阻断下游存活信号" },
      { cls: "mTOR 抑制剂", ex: "依维莫司 / 雷帕霉素", mech: "与 FKBP12 结合后作用于 mTORC1 的 FRB 域 → 别构抑制；肿瘤/器官移植抗排斥" },
      { cls: "PTEN 缺失合成致死", ex: "PARP 抑制剂", mech: "PTEN 缺失肿瘤依赖碱基切除修复，PARP 抑制 → 合成致死（概念延伸）" }
    ],
    phenotype: "细胞生长、存活、代谢增强；过度激活 → 肿瘤（PIK3CA 突变、PTEN 缺失）；抑制后凋亡/自噬、增殖停滞。"
  },

  jakstat: {
    name: "JAK-STAT 信号通路",
    summary: "细胞因子/干扰素 → 受体 → JAK → STAT → 入核转录，免疫与炎症的核心通路。",
    layers: ["细胞因子受体", "酪氨酸激酶", "转录因子", "免疫基因"],
    nodes: [
      { id: "ifnar", symbol: "IFNAR1/2", name: "I 型干扰素受体", type: "细胞因子受体",
        function: "IFNα/β 结合诱导受体二聚化，激活偶联的 JAK1/TYK2。",
        interactions: ["配体诱导异二聚化 → JAK 靠近反式磷酸化"] },
      { id: "jak", symbol: "JAK1/JAK2", name: "JAK（Janus 激酶）", type: "酪氨酸激酶",
        function: "与受体胞内域组成性结合；受体聚集后 JAK 相互反式磷酸化激活，磷酸化受体胞内酪氨酸。",
        domains: ["FERM 域（结合受体）", "假激酶域（调节）", "激酶域"],
        sites: ["JAK2 Tyr1007/Tyr1008：激活环磷酸化", "JAK1 Tyr1022/1023"],
        interactions: ["反式磷酸化激活", "磷酸化受体 pTyr 供 STAT SH2 对接", "JAK2 V617F 突变 → 组成性激活（骨髓增殖性肿瘤）"] },
      { id: "stat", symbol: "STAT1/STAT3", name: "STAT（信号转导与转录激活因子）", type: "转录因子",
        function: "SH2 域结合受体 pTyr → JAK 磷酸化 STAT 的 C 端 Tyr → 二聚化（SH2-pTyr 互锁）→ 入核。",
        domains: ["SH2 域", "DNA 结合域", "C 端转录激活域"],
        sites: ["STAT1 Tyr701 / STAT3 Tyr705：激活磷酸化", "STAT3 Ser727：MAPK 磷酸化（增强激活）"],
        interactions: ["Tyr 磷酸化后 SH2-pTyr 反向互锁形成同/异二聚体", "二聚体入核结合 GAS 元件"] },
      { id: "target_genes", symbol: "ISG · SOCS · BCL2L1", name: "干扰素/炎症基因", type: "基因表达",
        function: "STAT 激活 ISG（抗病毒）、SOCS（负反馈抑制 JAK）、存活基因（BCL2L1）。",
        interactions: ["ISG15/ISG54 抗病毒蛋白上调", "SOCS1/3 反馈抑制 JAK 激酶活性"] }
    ],
    edges: [
      { from: "ifnar", to: "jak", type: "激活", label: "反式磷酸化",
        mech: "配体诱导受体二聚化 → 偶联的 JAK1/TYK2 相互磷酸化激活环（Tyr1007/1008）" },
      { from: "jak", to: "stat", type: "磷酸化", label: "Tyr701/705",
        mech: "JAK 先磷酸化受体胞内酪氨酸，STAT 经 SH2 域对接后被 JAK 磷酸化 C 端 Tyr" },
      { from: "stat", to: "target_genes", type: "激活", label: "GAS 元件",
        mech: "pSTAT 二聚化（SH2-pTyr 互锁）→ 入核结合 GAS 启动子元件 → 转录" }
    ],
    drugs: [
      { cls: "JAK 抑制剂", ex: "鲁索替尼 / 托法替尼", mech: "ATP 竞争性抑制 JAK1/JAK2 激酶域，用于骨髓纤维化、类风湿关节炎；对 JAK2-V617F 有效" },
      { cls: "STAT3 抑制剂", ex: "Napabucasin 类（研究性）", mech: "干扰 STAT3 二聚化/DNA 结合，阻断肿瘤存活程序（临床研究中）" },
      { cls: "细胞因子中和抗体", ex: "Tocilizumab（IL-6R）", mech: "阻断 IL-6 受体 → JAK1/STAT3 信号减弱（类风湿、细胞因子风暴）" }
    ],
    phenotype: "抗病毒/抗炎基因上调、免疫激活；异常激活（JAK2 V617F、STAT3 持续磷酸化）→ 骨髓增殖性肿瘤、自身免疫、肿瘤免疫逃逸。"
  },

  hdac: {
    name: "HDAC 表观遗传调控",
    summary: "组蛋白乙酰化状态决定染色质开放度；HDAC 去乙酰化 → 染色质压缩 → 转录抑制，是表观遗传药物的靶区。",
    layers: ["组蛋白修饰酶", "染色质结构", "转录状态"],
    nodes: [
      { id: "hat", symbol: "EP300/CBP", name: "组蛋白乙酰转移酶（p300/CBP）", type: "表观酶",
        function: "把乙酰辅酶 A 的乙酰基转移到组蛋白赖氨酸 ε-氨基，中和正电荷 → 染色质松解。",
        domains: ["HAT 催化域", "溴结构域（读乙酰赖氨酸）", "KIX 域（结合 CREB）"],
        sites: ["催化 H3K9ac/H3K27ac/H4K16ac"],
        interactions: ["被 CREB pSer133 招募", "溴结构域识别已有乙酰化 → 协同扩散"] },
      { id: "hdac", symbol: "HDAC1/2", name: "组蛋白去乙酰化酶（HDAC1/2）", type: "表观酶",
        function: "去除赖氨酸乙酰基 → 组蛋白正电荷恢复 → 与 DNA 结合更紧 → 染色质压缩、转录抑制。",
        domains: ["Zn²⁺ 依赖催化口袋（I 类）"],
        sites: ["催化 Zn²⁺：乙酰赖氨酸水解（HDAC 抑制剂竞争该位点）"],
        interactions: ["在 NuRD/CoREST 复合物中靶向基因启动子", "被 SAHA/伏立诺他抑制"] },
      { id: "chromatin", symbol: "H3/H4 组蛋白", name: "组蛋白 H3/H4 乙酰化状态", type: "染色质",
        function: "乙酰化 → euchromatin（开放，可转录）；去乙酰化 → heterochromatin（压缩，沉默）。",
        domains: ["H3 尾部 K9/K14/K27", "H4 尾部 K5/K8/K12/K16"],
        sites: ["H3K9ac/H3K27ac：活性启动子/增强子标记", "H3K9me3：异染色质标记（拮抗）"],
        interactions: ["乙酰化招募溴结构域蛋白（BET）", "去乙酰化后压缩抑制转录"] },
      { id: "target_genes", symbol: "CDKN1A · BCL2 · NOTCH", name: "表观沉默/激活程序", type: "基因表达",
        function: "HDAC 抑制 → 数百基因去抑制，包括 p21（CDKN1A，细胞周期阻滞）、促凋亡基因。",
        interactions: ["CDKN1A（p21）上调 → G1 阻滞", "BCL2 下调 → 促凋亡", "肿瘤中异常 HDAC 高表达 → 抑癌基因沉默"] }
    ],
    edges: [
      { from: "hat", to: "chromatin", type: "乙酰化", label: "H3/H4 赖氨酸",
        mech: "p300/CBP 催化组蛋白尾部赖氨酸乙酰化 → 中和正电荷 → 染色质松解为开放状态" },
      { from: "hdac", to: "chromatin", type: "去乙酰化", label: "Zn²⁺ 催化",
        mech: "HDAC 的 Zn²⁺ 催化口袋水解乙酰赖氨酸 → 恢复正电荷 → 组蛋白-DNA 结合增强 → 染色质压缩、转录沉默" },
      { from: "chromatin", to: "target_genes", type: "调控", label: "可及性开关",
        mech: "开放染色质（乙酰化高）允许转录机器结合 → 基因表达；压缩染色质（去乙酰化）沉默基因" }
    ],
    drugs: [
      { cls: "HDAC 抑制剂（pan）", ex: "伏立诺他（SAHA）/ 西达本胺", mech: "模拟乙酰赖氨酸竞争 Zn²⁺ 催化口袋 → 抑制去乙酰化 → 组蛋白超乙酰化 → 抑癌基因去抑制（血液瘤获批）" },
      { cls: "选择性 HDAC 抑制剂", ex: "Entinostat（I 类选择性）", mech: "优先抑制 HDAC1/2/3，减少非靶毒性" },
      { cls: "BET 抑制剂（下游）", ex: "JQ1（研究性）", mech: "竞争溴结构域（读乙酰赖氨酸）→ 阻断增强子活性（与 HDAC 抑制协同）" }
    ],
    phenotype: "数百基因表达重编程：p21 上调（周期阻滞）、凋亡、分化；肿瘤细胞对 HDAC 抑制剂敏感（表观遗传脆弱性）；健康细胞影响有限。"
  },
  general: {
    name: "细胞信号与基因表达调控（泛通路）",
    summary: "该输入未精确命中预设通路，按通用“刺激 → 受体 → 信号级联 → 转录因子 → 基因表达”框架展示；具体基因请参考右侧最相似蛋白与表达变化列表。",
    layers: ["刺激 / 输入", "受体", "信号转导", "转录因子", "基因表达"],
    nodes: [
      { id: "stim", symbol: "扰动输入", name: "扰动输入（基因/药物/蛋白）", type: "GPCR",
        function: "本次查询的扰动输入；模型在统一空间中检索最相似已知蛋白/药物，以其功能锚点推断效应。",
        domains: [], sites: [], interactions: [] },
      { id: "recep", symbol: "相似蛋白", name: "最相似已知蛋白", type: "受体酪氨酸激酶",
        function: "统一空间中与该输入最相似的已知蛋白（见右侧列表），是效应推断的功能参照。",
        domains: [], sites: [], interactions: [] },
      { id: "sig", symbol: "信号级联", name: "胞内信号转导", type: "激酶",
        function: "受体下游的激酶级联与第二信使系统，把胞外刺激传递到核内。",
        domains: [], sites: [], interactions: [] },
      { id: "tf", symbol: "转录因子", name: "转录因子激活", type: "转录因子",
        function: "信号传导最终激活或抑制转录因子，改变靶基因的转录速率。",
        domains: [], sites: [], interactions: [] },
      { id: "expr", symbol: "基因表达", name: "靶基因表达改变", type: "基因表达",
        function: "上调/下调的靶基因（见右侧预测列表），最终影响细胞增殖、凋亡、代谢等表型。",
        domains: [], sites: [], interactions: [] }
    ],
    edges: [
      { from: "stim", to: "recep", type: "结合", label: "识别/结合" },
      { from: "recep", to: "sig", type: "激活", label: "激活级联" },
      { from: "sig", to: "tf", type: "激活", label: "入核激活" },
      { from: "tf", to: "expr", type: "调控", label: "转录调控" }
    ],
    phenotype: "泛通路展示：具体效应取决于与输入最相似的已知蛋白/药物（见右侧列表）。"
  }
};

/* 药物 → 通路映射（136 药里按 target/pathway 注释归入可展示的通路） */
const DRUG_PATHWAY_HINT = {
  "Ruxolitinib (INCB018424)": "jakstat", "Tofacitinib": "jakstat",
  "Erlotinib (OSI-744)": "mapk", "Gefitinib (Iressa)": "mapk",
  "Lapatinib (GW-572016)": "mapk", "Afatinib (BIBW-2992)": "mapk",
  "Trametinib (GSK1120212)": "mapk", "Selumetinib (AZD6244)": "mapk",
  "Vemurafenib (PLX4032)": "mapk", "Dabrafenib (GSK2118436)": "mapk",
  "Sotorasib (AMG-510)": "mapk",
  "Alpelisib (BYL719)": "pi3k", "Idelalisib (CAL-101)": "pi3k",
  "Everolimus (RAD001)": "pi3k", "Rapamycin (Sirolimus)": "pi3k",
  "Temsirolimus (CCI-779)": "pi3k", "Capivasertib (AZD5363)": "pi3k",
  "Vorinostat (SAHA)": "hdac", "Panobinostat (LBH589)": "hdac",
  "Entinostat (MS-275)": "hdac", "Romidepsin (FK228)": "hdac",
  "Belinostat (PXD101)": "hdac", "Trichostatin A": "hdac",
  "Forskolin": "pka", "Caffeine": "pka", "Theophylline": "pka",
  "Salbutamol": "pka", "Propranolol": "pka", "Rolipram": "pka"
};
