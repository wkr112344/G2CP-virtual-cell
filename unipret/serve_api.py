"""
虚拟细胞工作台 · 实时计算 API（P6）
=====================================
Flask 本地服务：前端选择细胞系+药物/基因 → POST /predict（或 /gene）
→ 后端模型（stageB 预测 + unipert_pretrain CPI 检索）当场计算 → 返回 JSON。

启动：python unipret/serve_api.py --port 8765
前端：http://localhost:8765/gui/workbench.html
"""
import os
import sys
import json
import math
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
GUI = os.path.join(BASE, "data", "gui")
CHEM_MAP = os.path.join(BASE, "unipret", "sciplex3_chems.json")
GENE_NAMES = os.path.join(GUI, "hvg_gene_names.json")
STRUCT = os.path.join(GUI, "structures")
CELL_NAMES = ["K562", "A549", "MCF7"]

# 基因/复合物 → PDB 结构映射（与 fetch_structures.py 一致）
PROTEIN_PDB = {
    "ADRB2": "2RH1", "GNAS": "1AZT", "ADCY": "1CJK", "PRKAR2A": "1CX4",
    "PRKACA": "1ATP", "CREB1": "1DH3", "EGFR": "1M17", "GRB2": "1GRI",
    "SOS1": "1XDW", "KRAS": "1CRR", "BRAF": "4MNF", "MAP2K1": "3SLS",
    "MAPK1": "1TVO", "ELK1": "1DUX", "IGF1R": "1K3A", "PIK3CA": "4JPS",
    "AKT1": "3CQW", "MTOR": "4JSN", "IFNAR": "IFNAR1" if False else "3SE4",
    "JAK1": "4L00", "JAK2": "2B7A", "STAT1": "1BF5", "STAT3": "1BG1",
    "HDAC1": "4BKX", "HDAC2": "4LXZ", "EP300": "4PZR",
    "COMPLEX_PKA": "2QCS", "COMPLEX_RASRAF": "4G0N",
    "COMPLEX_EGFREXTRA": "1IVO", "COMPLEX_CREBCBP": "1KDX", "COMPLEX_JAK2": "4FVP",
}


def create_app(port=8765, device=None):
    """构建 Flask app（桌面壳可 import；命令行入口 main() 兼容原用法）。"""
    if device is None:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    import torch
    from flask import Flask, request, jsonify, send_from_directory
    from unipret.config import DEVICE
    from unipret.data_bridge import CompoundGraphCache
    from unipret.train_stages import _make_unipert
    from unipret.compound_encoder import smiles_to_ecfp4
    from unipret.effect_model import PerturbationEffectModel

    print(">>> 加载模型（stageB 预测 + unipert_pretrain CPI）...", flush=True)
    sb = torch.load(os.path.join(BASE, "stageB.pt"), map_location=DEVICE)
    gv_b = sb["gene_vocab"]
    hvg_n = sb.get("hvg_dim", 2000)
    unipert = _make_unipert(len(gv_b) + 1, gene_vocab=gv_b, gene_mode="esm").to(DEVICE)
    unipert.load_state_dict(sb["unipert"])
    ea = PerturbationEffectModel.build_stage_a(unipert, hvg_n).to(DEVICE)
    effect = PerturbationEffectModel.build_stage_b(ea).to(DEVICE)
    effect.load_state_dict(sb["effect"])
    effect.eval()

    up = torch.load(os.path.join(BASE, "unipert_pretrain.pt"), map_location=DEVICE)
    unipert_p = _make_unipert(len(up["gene_vocab"]), gene_vocab=up["gene_vocab"],
                              gene_mode="esm").to(DEVICE)
    unipert_p.load_state_dict(up["unipert"])
    unipert_p.eval()
    gv_p = {str(k).strip().upper(): v for k, v in up["gene_vocab"].items()}
    pool_genes = sorted(gv_p.keys())
    pool_idx = [gv_p[g] for g in pool_genes]
    with torch.no_grad():
        P = unipert_p.encode_gene(None, torch.tensor(pool_idx, device=DEVICE),
                                  torch.zeros(len(pool_idx), dtype=torch.long, device=DEVICE))
        P = P.cpu().numpy()
    import numpy as np
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    SIM = P @ P.T   # 池内蛋白相似度矩阵 [905,905]：实时算"这个蛋白下一步最可能传给谁"

    gene_names = json.load(open(GENE_NAMES, encoding="utf-8"))
    chem_map = json.load(open(CHEM_MAP, encoding="utf-8"))
    gcache = CompoundGraphCache()
    print(f"    模型就绪：stageB（hvg={hvg_n}）+ CPI 池 {len(pool_genes)}", flush=True)

    # ---- 未知蛋白编码：ESM 实时（CPU，避免与训练争 GPU）----
    from unipret.esm_encoder import load_esm, esm_mean_embedding
    esm_model, esm_alphabet = load_esm()
    esm_model.eval()
    print("    ESM2-8M 已加载（CPU，未知蛋白实时编码）", flush=True)

    def encode_unknown_protein(seq):
        feat = esm_mean_embedding(esm_model, esm_alphabet, [seq], device="cpu")
        ge = unipert_p.gene_encoder
        x = ge.act(ge.esm_proj(feat.to(DEVICE)))
        gid0 = torch.zeros(1, ge.gene_id_emb.embedding_dim, device=DEVICE)
        return ge.ln(ge.fuse(torch.cat([x, gid0], dim=1)))     # [1,256] 统一空间

    # 通路 → 基因（用于未知输入的通路归属；扩展版覆盖常见信号通路核心基因）
    PW_GENES = {
        "pka": ["ADRB2", "GNAS", "ADCY", "PRKAR2A", "PRKACA", "CREB1", "PRKACB"],
        "mapk": ["EGFR", "GRB2", "SOS1", "KRAS", "BRAF", "MAP2K1", "MAPK1", "MAPK3",
                 "ELK1", "NRAS", "RAF1", "MAPK14", "MAP2K3", "MAP2K6", "MAP4K3", "MAP4K5", "DUSP9"],
        "pi3k": ["IGF1R", "PIK3CA", "AKT1", "AKT2", "MTOR", "PTEN", "TSC1", "TSC2",
                 "RPTOR", "RICTOR", "PDPK1", "SGK1"],
        "jakstat": ["IFNAR", "JAK1", "JAK2", "STAT1", "STAT3", "STAT5A", "IL6R"],
        "hdac": ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC6", "EP300", "CREBBP"],
        "wnt": ["CTNNB1", "GSK3B", "APC", "AXIN1", "LEF1", "TCF7L2", "WNT1", "CTNNBIP1"],
        "notch": ["NOTCH1", "NOTCH2", "NOTCH3", "RBPJ", "MAML1", "DLL1"],
        "tgfb": ["TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4", "TGFB1", "TGFB2"],
        "nfkb": ["NFKB1", "NFKB2", "RELA", "NFKBIA", "IKBKB", "CHUK", "TNF"],
        "p53": ["TP53", "MDM2", "CDKN1A", "BAX", "TP63", "ATM", "TP73", "BCL2L11", "BAK1"],
        "cellcyc": ["CDK1", "CDK2", "CDK4", "CDK6", "CCND1", "CCNE1", "RB1", "CCNA2",
                    "CDKN2A", "CDKN1B"],
        "apoptosis": ["BCL2", "BCL2L1", "BAD", "CASP3", "CASP8", "CASP9", "BID", "MCL1", "BAK1", "BCL2L11"],
        "er": ["ESR1", "ESR2", "PGR", "AR", "NR3C1"],
        "hif": ["HIF1A", "VHL", "EPAS1", "EGLN1", "ARNT"],
        "egfr": ["ERBB2", "ERBB3", "ERBB4", "EGFR", "CBL", "PTPN1", "PTPN12", "PTPN13", "PTPN9"],
        "dnarepair": ["BRCA1", "BRCA2", "ATR", "PARP1", "RAD51", "CHEK1", "CHEK2"],
        "rhogtp": ["RHOA", "RAC1", "CDC42", "PAK1", "DIAPH1"],
        "ampk": ["PRKAA1", "PRKAA2", "STK11", "CAMKK2", "SIRT1"],
        # 扩展：覆盖 102 个可扰动词表基因（按功能归入具体通路）
        "epigen": ["KMT2A", "SET", "ARID1A", "BCORL1", "ELMSAN1", "CBFA2T3", "CITED1"],
        "tfdev": ["CEBPA", "CEBPB", "CEBPE", "FOXA1", "FOXA3", "FOXF1", "FOXL2", "FOXO4",
                  "ETS2", "IRF1", "KLF1", "RUNX1T1", "TBX2", "TBX3", "MEIS1", "SPI1",
                  "IKZF3", "SNAI1", "HNF4A", "AHR", "EGR1", "JUN", "FOSB", "TP73",
                  "DLX2", "HOXA13", "HOXB9", "HOXC13", "LHX1", "ISL2", "LYL1", "PRDM1",
                  "FEV", "POU3F2", "RREB1", "ZBTB1", "ZBTB10", "ZBTB25", "ZC3HAV1",
                  "ZNF318", "TSC22D1", "CSRNP1", "MIDN", "SAMD1"],
        "gpcrsig": ["S1PR2", "ARRDC3", "PRTG"],
        "glycolysis": ["HK2", "BPGM", "SLC2A1"],
    }
    PW_NAME = {
        "pka": "cAMP-PKA 信号通路", "mapk": "EGFR-RAS-MAPK 通路",
        "pi3k": "PI3K-AKT-mTOR 通路", "jakstat": "JAK-STAT 通路",
        "hdac": "HDAC 表观遗传调控", "wnt": "Wnt/β-catenin 通路",
        "notch": "Notch 信号通路", "tgfb": "TGF-β-SMAD 通路",
        "nfkb": "NF-κB 炎症信号通路", "p53": "p53 肿瘤抑制通路",
        "cellcyc": "细胞周期调控", "apoptosis": "凋亡信号通路",
        "er": "激素受体信号通路", "hif": "HIF 缺氧响应通路",
        "egfr": "EGFR 家族受体通路", "dnarepair": "DNA 损伤修复通路",
        "rhogtp": "Rho/Rac 细胞骨架通路", "ampk": "AMPK 能量代谢通路",
        "epigen": "表观遗传与染色质重塑", "tfdev": "转录因子与细胞分化",
        "gpcrsig": "GPCR 信号通路", "glycolysis": "糖酵解与能量代谢",
        "general": "细胞信号与基因表达调控（泛通路）",
    }
    # 药物自带通路注释 → 通路 key（v2：drug 链的通路归属优先用真实注释）
    PATHWAY_MAP = {
        "JAK/STAT": "jakstat", "MAPK": "mapk", "PI3K/Akt/mTOR": "pi3k",
        "DNA Damage": "dnarepair", "Epigenetics": "epigen", "Cell Cycle": "cellcyc",
        "Apoptosis": "apoptosis", "TGF-beta/Smad": "tgfb", "Angiogenesis": "hif",
        "Endocrinology & Hormones": "er", "Cytoskeletal Signaling": "rhogtp",
        "Metabolism": "glycolysis", "GPCR & G Protein": "gpcrsig",
        "Protein Tyrosine Kinase": "egfr", "Neuronal Signaling": "gpcrsig",
        "Microbiology": "general", "Ubiquitin": "general", "Proteases": "general",
        "Others": "general",
    }
    # 药物真实靶点 → 结合机制 + 结构域示意（v2：药物链第一跳用真实靶点，非相似度检索）
    TARGET_LIB = {
        "HDAC": {"name": "HDAC1/2/3（组蛋白去乙酰化酶）", "struct": "催化锌离子口袋 + 底物通道",
                 "bind": "抑制剂螯合催化域锌离子、占据底物通道，阻断组蛋白去乙酰化",
                 "mech": "组蛋白乙酰化水平升高 → 染色质解旋开放 → 数百基因转录重编程（抑癌基因去抑制）"},
        "JAK": {"name": "JAK1/JAK2（Janus 激酶）", "struct": "激酶催化域（ATP 口袋）",
                "bind": "竞争 ATP 结合位点，抑制 JAK 激酶催化活性",
                "mech": "细胞因子受体无法启动 JAK 磷酸化 → STAT 不被磷酸化、无法二聚化入核 → JAK-STAT 信号中断，免疫/炎症相关基因沉默"},
        "Aurora Kinase": {"name": "Aurora A/B（有丝分裂激酶）", "struct": "激酶催化域（ATP 口袋）",
                          "bind": "竞争 ATP，抑制 Aurora 激酶催化活性",
                          "mech": "中心体成熟/着丝粒组装失败 → 纺锤体检查点持续激活 → 有丝分裂阻滞、多极纺锤体 → 细胞分裂失败/凋亡"},
        "MEK": {"name": "MEK1/2（MAPK/ERK 激酶）", "struct": "激酶催化域 + 变构口袋",
                "bind": "结合变构口袋（非 ATP 竞争），锁定 MEK 非活性构象",
                "mech": "MEK 无法磷酸化 ERK → ERK 不转位入核 → MAPK 级联中断 → 增殖基因（MYC/CCND1）表达下降"},
        "EGFR": {"name": "EGFR（表皮生长因子受体）", "struct": "胞内酪氨酸激酶域（ATP 口袋）",
                 "bind": "竞争 ATP 结合位点，抑制 EGFR 自磷酸化",
                 "mech": "受体无法自身磷酸化 → 下游 GRB2/SOS/RAS 无法被招募 → MAPK/PI3K 两条增殖通路同时受抑"},
        "EGFR,HER2": {"name": "EGFR / HER2（ErbB 受体家族）", "struct": "胞内酪氨酸激酶域",
                      "bind": "竞争 ATP，抑制 EGFR/HER2 同源与异源二聚化信号",
                      "mech": "ErbB 二聚化信号中断 → RAS-MAPK 与 PI3K-AKT 增殖通路受抑 → 肿瘤增殖停滞"},
        "PARP": {"name": "PARP1/2（聚 ADP 核糖聚合酶）", "struct": "NAD+ 结合口袋（催化域）",
                 "bind": "竞争 NAD+ 结合位点，抑制 PAR 化修饰",
                 "mech": "DNA 单链断裂无法被 PAR 化标记招募修复 → 断裂累积为双链 → 复制叉崩溃 → 同源重组缺陷细胞凋亡（合成致死）"},
        "Bcl-2": {"name": "BCL2（抗凋亡蛋白）", "struct": "BH3 结合沟槽（变构位点）",
                  "bind": "占据 BH3 沟槽，释放被扣留的促凋亡蛋白（BAX/BAK）",
                  "mech": "BAX/BAK 在线粒体外膜成孔 → 释放细胞色素 c → 半胱天冬酶级联激活 → 凋亡"},
        "CDK": {"name": "CDK2/4/6（细胞周期蛋白依赖性激酶）", "struct": "激酶催化域（ATP 口袋）",
                "bind": "竞争 ATP，抑制 CDK-cyclin 复合物催化活性",
                "mech": "RB 蛋白无法磷酸化 → E2F 转录因子被扣留 → G1/S 检查点阻滞 → 细胞周期停滞"},
        "Topoisomerase": {"name": "拓扑异构酶 I/II", "struct": "DNA 切割-再连接催化域",
                          "bind": "稳定拓扑异构酶-DNA 共价复合物（切割中间体）",
                          "mech": "DNA 断裂无法再连接 → 复制叉遇断裂处崩溃 → 双链断裂积累 → DNA 损伤应答 + 凋亡"},
        "DNA/RNA Synthesis": {"name": "DNA/RNA 聚合酶", "struct": "核苷酸掺入催化位点",
                              "bind": "作为核苷酸类似物掺入新生链，终止链延伸",
                              "mech": "复制/转录链提前终止 → 增殖细胞 DNA 合成失败 → S 期阻滞 → 凋亡"},
        "DNA alkylator": {"name": "DNA 烷化剂（鸟嘌呤 N7）", "struct": "DNA 碱基（烷化位点）",
                          "bind": "烷化鸟嘌呤，形成 DNA 交联/加合物",
                          "mech": "DNA 交联阻断解旋与复制 → 双链断裂 → 错配修复耗竭 → 细胞死亡"},
        "HSP (e.g. HSP90)": {"name": "HSP90（分子伴侣）", "struct": "N 端 ATP 结合口袋",
                             "bind": "竞争 N 端 ATP 位点，抑制伴侣循环",
                             "mech": "客户蛋白（激酶/转录因子）无法正确折叠 → 被泛素化降解 → 多条致癌通路同时塌陷"},
        "Histone Methyltransferase": {"name": "组蛋白甲基转移酶（SET 结构域）", "struct": "SET 催化结构域",
                                      "bind": "竞争 SAM 甲基供体或底物通道，抑制组蛋白甲基化",
                                      "mech": "H3K27me3/H3K4me3 标记改变 → 靶基因表观沉默/激活 → 分化与增殖程序紊乱"},
        "DNA Methyltransferase": {"name": "DNMT（DNA 甲基转移酶）", "struct": "催化域（胞嘧啶口袋）",
                                  "bind": "掺入核苷类似物或竞争 SAM，抑制 CpG 甲基化",
                                  "mech": "启动子高甲基化沉默的基因（抑癌基因）重新表达 → 表观遗传重编程"},
        "VEGFR": {"name": "VEGFR1/2（血管内皮生长因子受体）", "struct": "胞内酪氨酸激酶域",
                  "bind": "竞争 ATP，抑制 VEGFR 激酶活性",
                  "mech": "内皮细胞增殖/迁移信号中断 → 血管新生受阻 → 肿瘤缺血"},
        "HIF": {"name": "HIF-1α / HIF-2α（缺氧诱导因子）", "struct": "bHLH-PAS 结构域",
                "bind": "抑制 HIF-α 羟化酶或 HIF 转录活性",
                "mech": "缺氧应答基因（VEGF/GLUT1/EPO）不再上调 → 代谢与血管新生适应受损"},
        "PKC": {"name": "PKC（蛋白激酶 C）", "struct": "激酶催化域（ATP 口袋）",
                "bind": "竞争 ATP，抑制 PKC 催化活性",
                "mech": "下游磷酸化级联中断 → 细胞骨架重排、增殖与分泌信号改变"},
        "Sirtuin": {"name": "SIRT1/2（NAD+ 依赖去乙酰化酶）", "struct": "NAD+ 结合催化域",
                    "bind": "竞争 NAD+ 辅因子，抑制去乙酰化",
                    "mech": "p53/FOXO 等乙酰化水平升高 → 代谢、衰老与应激相关基因表达改变"},
        "Glucocorticoid Receptor": {"name": "糖皮质激素受体（GR）", "struct": "配体结合域（LBD）",
                                    "bind": "作为配体占据 LBD，诱导受体构象改变与核转位",
                                    "mech": "GR 入核结合 GRE → 抗炎基因上调、促炎转录因子（AP-1/NF-κB）被拮抗 → 免疫抑制"},
        "Estrogen/progestogen Receptor": {"name": "雌激素/孕激素受体（ER/PR）", "struct": "配体结合域（LBD）",
                                          "bind": "作为配体结合 LBD，改变共调节因子招募",
                                          "mech": "受体二聚化入核 → 雌激素应答基因（增殖/分泌）转录改变 → 激素依赖性细胞命运改变"},
        "Beta Amyloid,Gamma-secretase": {"name": "γ-分泌酶复合物", "struct": "跨膜蛋白酶活性位点",
                                         "bind": "抑制 γ-分泌酶切割，减少 Aβ 生成",
                                         "mech": "APP 加工改变 → 淀粉样肽生成减少 → 神经毒性降低（同时 Notch 切割也受影响）"},
        "Microtubule Associated": {"name": "微管蛋白（α/β 异二聚体）", "struct": "紫杉醇/长春碱结合位点",
                                   "bind": "结合微管蛋白，稳定（紫杉烷）或去稳定（长春碱）微管",
                                   "mech": "微管动力学破坏 → 纺锤体无法正确组装/解聚 → 有丝分裂阻滞 → 凋亡"},
        "IGF-1R": {"name": "IGF-1 受体（胰岛素样生长因子受体）", "struct": "胞内酪氨酸激酶域",
                   "bind": "竞争 ATP，抑制 IGF-1R 激酶活性",
                   "mech": "IGF-1 生长信号中断 → 抗增殖、促凋亡"},
        "Others": {"name": "多靶点/未分类", "struct": "—", "bind": "—",
                   "mech": "作用机制多样，见下方模型预测的受影响蛋白与通路"},
    }
    # 通路级联机制（v3：每一站都是具体蛋白/分子名 + 该站机制）
    TRANSFER_LIB = {
        "jakstat": [
            {"n": "JAK（激酶）", "m": "受体二聚化启动 JAK 交叉磷酸化 → STAT 被磷酸化"},
            {"n": "pSTAT 二聚体", "m": "磷酸化 STAT 二聚化 → 转位入核"},
            {"n": "STAT-靶基因启动子", "m": "结合 GAS 启动子 → 上调免疫/增殖靶基因"}],
        "mapk": [
            {"n": "RAS-GTP", "m": "EGFR 活化 → 招募 GRB2-SOS → 催化 RAS 装载 GTP"},
            {"n": "RAF → MEK", "m": "RAS-GTP 招募 RAF → RAF 磷酸化激活 MEK"},
            {"n": "ERK（入核）", "m": "MEK 磷酸化 ERK → ERK 转位入核 → 激活 MYC/ELK1 增殖转录因子"}],
        "pi3k": [
            {"n": "PI3K / 膜上 PIP3", "m": "受体激活 PI3K → 质膜生成 PIP3"},
            {"n": "AKT", "m": "PIP3 招募 AKT → PDK1 磷酸化激活 AKT"},
            {"n": "mTORC1 / FOXO", "m": "AKT 磷酸化激活 mTORC1、抑制 FOXO → 促存活、促增殖、抑凋亡"}],
        "p53": [
            {"n": "p53（ATM/ATR 磷酸化）", "m": "DNA 损伤/应激 → ATM/ATR 磷酸化 p53 → p53 蛋白稳定"},
            {"n": "p53 四聚体", "m": "p53 四聚化 → 转录激活 p21/BAX"},
            {"n": "p21 → CDK；BAX → 线粒体", "m": "p21 抑制 CDK → 周期阻滞；BAX 促线粒体外膜透化 → 凋亡"}],
        "hdac": [
            {"n": "HDAC（组蛋白去乙酰化酶）", "m": "抑制 HDAC → 组蛋白乙酰化水平升高"},
            {"n": "染色质（开放状态）", "m": "乙酰化中和组蛋白正电荷 → 染色质解旋、转录因子可及性增加"},
            {"n": "抑癌基因启动子", "m": "抑癌基因重新表达 → 转录重编程"}],
        "cellcyc": [
            {"n": "CDK-cyclin 复合物", "m": "CDK 活性受抑 → RB 保持低磷酸化"},
            {"n": "E2F（被 RB 扣留）", "m": "低磷酸化 RB 结合并扣留 E2F → G1/S 基因无法转录"},
            {"n": "G1/S 检查点", "m": "周期阻滞于 G1 → 增殖停止"}],
        "apoptosis": [
            {"n": "BAX/BAK（线粒体孔）", "m": "BH3 信号激活 BAX/BAK → 线粒体外膜透化"},
            {"n": "细胞色素 c（释放）", "m": "细胞色素 c 释放到胞质 → 组装凋亡小体"},
            {"n": "半胱天冬酶级联", "m": "Caspase-9 激活 → Caspase-3 执行 → 细胞凋亡"}],
        "egfr": [
            {"n": "EGFR 二聚体（自磷酸化）", "m": "配体诱导受体二聚化 → 酪氨酸自磷酸化"},
            {"n": "GRB2-SOS → RAS", "m": "招募 GRB2/SOS → 激活 RAS"},
            {"n": "ERK + AKT", "m": "RAS→RAF→MEK→ERK 增殖级联 + PI3K-AKT 存活级联"}],
        "nfkb": [
            {"n": "IKK → IκB", "m": "IKK 磷酸化 IκB → IκB 被泛素化降解"},
            {"n": "NF-κB（入核）", "m": "NF-κB 释放并转位入核"},
            {"n": "NF-κB-炎症基因", "m": "转录促炎/抗凋亡基因（IL6、BCL2L1）"}],
        "dnarepair": [
            {"n": "ATM/ATR（损伤感应）", "m": "DNA 损伤被 ATM/ATR 感应激活"},
            {"n": "CHK1/2（检查点激酶）", "m": "检查点激酶被磷酸化激活"},
            {"n": "周期阻滞 + 修复复合物", "m": "阻滞细胞周期并招募修复复合物"}],
        "wnt": [
            {"n": "β-catenin（稳定）", "m": "Wnt 配体结合 Frizzled → β-catenin 不被降解而稳定"},
            {"n": "β-catenin（入核）", "m": "β-catenin 转位入核"},
            {"n": "TCF/LEF-靶基因", "m": "结合 TCF/LEF → 增殖/干性基因转录"}],
        "notch": [
            {"n": "γ-分泌酶切割 → NICD", "m": "Notch 被 γ-分泌酶切割释放 NICD"},
            {"n": "NICD（入核）", "m": "NICD 转位入核"},
            {"n": "CSL-靶基因", "m": "结合 CSL → 分化命运决定基因转录"}],
        "tgfb": [
            {"n": "SMAD2/3（磷酸化）", "m": "TGF-β 结合受体 → SMAD2/3 被受体磷酸化"},
            {"n": "SMAD 复合物（入核）", "m": "pSMAD 与 SMAD4 组装复合物入核"},
            {"n": "SBE-靶基因", "m": "结合 SBE → 生长抑制/EMT 基因转录"}],
        "er": [
            {"n": "ER（配体结合变构）", "m": "雌激素结合 ER 配体结合域 → 受体构象改变、热丌克蛋白解离"},
            {"n": "ER 二聚体（入核）", "m": "ER 二聚化并转位入核"},
            {"n": "ERE-靶基因", "m": "结合雌激素应答元件 → 增殖/分泌基因转录"}],
        "hif": [
            {"n": "HIF-α（稳定）", "m": "缺氧 → HIF-α 脯氨酸炾化受抑 → 蛋白稳定"},
            {"n": "HIF-α/β 二聚体（入核）", "m": "HIF-α 与 HIF-β 二聚化入核"},
            {"n": "HRE-靶基因", "m": "结合缺氧应答元件 → 血管新生/糖酸解基因转录"}],
        "rhogtp": [
            {"n": "RhoA/Rac（装载 GTP）", "m": "受体激活 GEF → RhoA/Rac 交换 GDP 为 GTP"},
            {"n": "PAK/ROCK", "m": "GTP 型 Rho 激活下游 PAK/ROCK"},
            {"n": "肌动蛋白骨架", "m": "肌动蛋白重排 → 迁移/收缩/分裂"}],
        "ampk": [
            {"n": "AMPK（激活）", "m": "AMP/ATP 升高 → AMPK 被 LKB1/CaMKK2 磷酸化激活"},
            {"n": "AMPK-代谢酶", "m": "AMPK 磷酸化下游代谢酶（ACC、ULK1）"},
            {"n": "分解代谢程序", "m": "抑制合成代谢、促进分解产能"}],
        "epigen": [
            {"n": "表观酶（修饰改变）", "m": "表观遗传酶活性改变 → 组蛋白/DNA 修饰重编程"},
            {"n": "染色质（状态改变）", "m": "染色质开放/关闭状态重编程"},
            {"n": "基因表达谱", "m": "基因表达谱大规模改变"}],
        "tfdev": [
            {"n": "转录因子（活性改变）", "m": "转录因子活性/丰度改变"},
            {"n": "顺式元件（结合改变）", "m": "靶基因启动子结合谱改变"},
            {"n": "分化程序", "m": "分化/应激程序重编程"}],
        "gpcrsig": [
            {"n": "G 蛋白（激活）", "m": "配体结合 GPCR → G 蛋白 α 亚基交换 GTP 激活"},
            {"n": "第二信使（cAMP/Ca²⁺）", "m": "腾苷酸环化酶/磷脂酶 C 活化 → 第二信使升高"},
            {"n": "下游激酶/离子通道", "m": "PKA/PKC/离子通道响应 → 细胞功能改变"}],
        "glycolysis": [
            {"n": "糖酸解酶（活性改变）", "m": "糖酸解酶活性/表达改变"},
            {"n": "代谢通量（重排）", "m": "葡萄糖代谢通量重排"},
            {"n": "产能/合成适应", "m": "细胞能量与生物合成适应"}],
        "general": [
            {"n": "扰动蛋白信号", "m": "扰动蛋白信号进入细胞"},
            {"n": "相似蛋白网络（放大）", "m": "通过相似蛋白网络逐级放大"},
            {"n": "转录组（重编程）", "m": "转录组重编程 → 细胞表型改变"}],
    }
    # 表型终点具体描述（v2：基因表达调控后细胞最终变成什么样）
    PW_PHENO = {
        "p53": "细胞周期阻滞于 G1/S（p21 抑制 CDK）、促凋亡、诱导衰老与 DNA 修复——肿瘤被抑制，正常细胞进入静息",
        "jakstat": "免疫/炎症相关基因下调 → 免疫应答与炎症减弱（骨髓增殖/自身免疫缓解）；长期抑制可致免疫抑制",
        "mapk": "增殖信号中断 → 细胞增殖停滞、趋向凋亡；肿瘤生长受抑",
        "pi3k": "存活信号减弱 → 细胞生长减缓、自噬/凋亡倾向增加",
        "hdac": "染色质开放 + 抑癌基因去抑制 → 分化/凋亡倾向增强、增殖受抑（对增殖快的肿瘤细胞杀伤更明显）",
        "cellcyc": "细胞周期阻滞于 G1 → 增殖停止，进入静息或衰老",
        "apoptosis": "线粒体凋亡通路激活 → 细胞程序性死亡",
        "egfr": "EGFR 驱动增殖停滞 → 肿瘤退缩；正常上皮更新减慢",
        "nfkb": "炎症基因下调 → 炎症反应减轻；抗凋亡能力下降",
        "dnarepair": "DNA 修复受损 → 损伤累积 → 突变增多、细胞趋向凋亡或衰老",
        "wnt": "干性/增殖程序下调 → 分化倾向增强",
        "notch": "分化命运决定改变 → 谱系分化异常或增强",
        "tgfb": "生长抑制恢复 → 增殖减慢；EMT 程序受影响",
        "er": "雌激素驱动增殖减弱 → 激素依赖性生长受抑",
        "hif": "缺氧适应减弱 → 血管新生与糖酵解代谢下降",
        "rhogtp": "细胞骨架重排受阻 → 迁移、侵袭能力下降",
        "ampk": "代谢转向分解产能 → 生长减缓、自噬增强",
        "epigen": "表观重编程 → 分化/衰老/凋亡倾向改变，增殖受抑",
        "tfdev": "分化与应激程序重排 → 细胞命运改变",
        "gpcrsig": "GPCR 下游信号改变 → 分泌、收缩、迁移等功能变化",
        "glycolysis": "糖酵解通量改变 → 能量代谢与生物合成适应",
        "general": "转录组整体重编程 → 细胞状态（增殖/分化/应激/存活）向扰动方向偏移",
    }
    PPI_MECH = {"2QCS": "cAMP 结合调节亚基 CNB 域 → 与催化亚基解离（抑制解除）",
                "4G0N": "KRAS-GTP 结合 BRAF 的 RBD 域 → RAF 二聚化激活",
                "1KDX": "pSer133-CREB 的 KID 域结合 CBP 的 KIX 域 → 招募共激活因子",
                "1IVO": "EGF 配体桥接诱导 EGFR 胞外域二聚化"}

    # 基因 → 敲除机制（覆盖 102 可扰动词表重点基因；功能+敲除后果）
    GENE_MECH = {
        "CDKN1A": "p21（细胞周期抑制因子）：抑制 CDK2/4-cyclin 复合物，执行 p53 的周期阻滞功能。敲除 → 失去对 CDK 的抑制 → G1/S 检查点失效 → 细胞增殖加快、DNA 损伤后不能及时停滞。",
        "CDKN1B": "p27（CIP/KIP 家族）：抑制 CDK2-cyclinE。敲除 → CDK2 过度激活 → 加速进入 S 期 → 增殖增强（肿瘤中常见失活）。",
        "CDKN1C": "p57：抑制多种 CDK，维持分化/静止状态。敲除 → 分化缺陷、增殖失控。",
        "BAK1": "促凋亡蛋白（Bcl-2 家族效应分子）：线粒体外膜成孔 → 释放细胞色素 c → 启动凋亡。敲除 → 凋亡执行受阻 → 应激细胞存活增加（抗凋亡表型）。",
        "BCL2L11": "BIM（BH3-only 促凋亡蛋白）：中和抗凋亡 Bcl-2 成员、激活 BAX/BAK。敲除 → 凋亡阈值升高 → 细胞对损伤/药物诱导凋亡不敏感。",
        "MAPK1": "ERK2（MAPK 通路末端激酶）：磷酸化核转录因子与胞质底物，驱动增殖/分化。敲除 → ERK 信号减弱 → 生长因子驱动的增殖受损。",
        "MAP2K3": "MKK3（p38 通路激酶）：激活 p38 → 应激/炎症响应。敲除 → p38 应激响应减弱。",
        "MAP2K6": "MKK6（p38 通路激酶）：同上，激活 p38。敲除 → 应激响应受损。",
        "MAP4K3": "MAP4K 家族激酶：上游激酶级联。敲除 → 下游 JNK/p38 信号减弱。",
        "MAP4K5": "MAP4K5：调控 JNK/p38 与免疫信号。敲除 → 应激信号减弱。",
        "DUSP9": "MAPK 磷酸酶：去磷酸化失活 ERK（负调控）。敲除 → ERK 去抑制 → MAPK 信号增强（类似抑癌基因失活）。",
        "TGFBR2": "TGF-β 受体 II：配体结合启动 SMAD 通路 → 抑制增殖/促凋亡。敲除 → TGF-β 生长抑制解除 → 增殖失控（多种癌中失活）。",
        "TP73": "p73（p53 家族转录因子）：转录 p21/BAX 等促凋亡基因。敲除 → p53 样肿瘤抑制功能部分丧失。",
        "JUN": "AP-1 转录因子 c-Jun：响应应激/生长信号，调控增殖与炎症基因。敲除 → AP-1 靶基因表达改变 → 增殖/应激响应受损。",
        "FOSB": "AP-1 亚基：与 Jun 形成二聚体。敲除 → AP-1 活性降低。",
        "EGR1": "早期生长响应转录因子：损伤/应激即刻早期基因。敲除 → 应激转录响应延迟。",
        "CBL": "E3 泛素连接酶：泛素化降解活化 RTK（负调控）。敲除 → RTK（EGFR 等）降解减少 → 受体信号增强。",
        "PTPN1": "PTP1B（蛋白酪氨酸磷酸酶）：去磷酸化胰岛素受体/RTK（负调控）。敲除 → 胰岛素/RTK 信号增强（代谢与肿瘤相关）。",
        "PTPN12": "PTP-PEST：负调控 RTK/整合素信号。敲除 → 迁移/增殖信号增强。",
        "PTPN13": "PTP-BAS：磷酸酶，调控凋亡/细胞骨架。敲除 → 相关信号改变。",
        "PTPN9": "磷酸酶：去磷酸化 STAT3 等。敲除 → JAK-STAT 信号增强。",
        "SGK1": "血清/糖皮质激素激酶（PI3K 下游）：调控离子通道与增殖。敲除 → PI3K 下游效应部分减弱。",
        "CEBPA": "髓系分化主控转录因子：促粒细胞分化、抑制增殖。敲除 → 髓系分化阻滞、增殖增强（AML 相关）。",
        "CEBPB": "CEBP 家族：炎症/急性期响应。敲除 → 炎症基因响应受损。",
        "CEBPE": "嗜酸性/中性粒分化转录因子。敲除 → 粒细胞分化缺陷。",
        "SPI1": "PU.1（造血分化主控）：髓系/淋系分化。敲除 → 造血分化阻滞。",
        "IRF1": "干扰素响应转录因子：激活 IFN 靶基因与免疫监视。敲除 → 抗病毒/抗肿瘤免疫减弱。",
        "IKZF3": "Aiolos（淋巴分化转录因子）。敲除 → 淋巴细胞分化/功能异常。",
        "KMT2A": "MLL1（组蛋白 H3K4 甲基转移酶）：写入转录激活标记。敲除 → H3K4me3 减少 → 靶基因（HOX 等）沉默 → 发育/分化缺陷。",
        "SET": "SET 蛋白：抑制组蛋白乙酰化/PP2A。敲除 → 表观调控失衡。",
        "ARID1A": "SWI/SNF 染色质重塑复合物亚基：开放染色质促转录。敲除 → 染色质重塑缺陷 → 基因表达异常（多种癌失活）。",
        "AHR": "芳香烃受体：外源物（二噁英等）代谢与免疫调控转录因子。敲除 → 外源物解毒响应受损。",
        "HK2": "己糖激酶 2：糖酵解第一步限速酶。敲除 → 糖酵解受损 → 能量供应下降（癌细胞对 HK2 依赖高）。",
        "S1PR2": "鞘氨醇-1-磷酸受体 2（GPCR）：调控迁移/血管生成。敲除 → 相关信号改变。",
        "FOXO4": "叉头转录因子：应激/代谢/凋亡基因调控。敲除 → 抗氧化与凋亡响应减弱。",
        "HNF4A": "肝/肠/胰分化主控转录因子。敲除 → 内胚层分化缺陷。",
        "KLF1": "红细胞分化主控转录因子。敲除 → 红系分化障碍。",
        "TBX2": "T-box 转录因子：发育与增殖调控。敲除 → 发育/增殖基因改变。",
        "TBX3": "T-box 转录因子：维持干性/抑制衰老。敲除 → 干性/分化改变。",
        "SNAI1": "Snail（EMT 主控）：抑制 E-钙黏蛋白 → 诱导上皮-间质转化。敲除 → EMT 受阻 → 迁移/侵袭能力下降。",
        "RUNX1T1": "ETO（RUNX1 共抑制因子）：招募表观抑制复合物。敲除 → RUNX1 靶基因去抑制（白血病相关）。",
        "ETS2": "ETS 家族转录因子：增殖/肿瘤相关。敲除 → 靶基因表达改变。",
        "FOXA1": "FoxA1（先锋转录因子）：打开染色质启动激素受体（ER/AR）靶基因。敲除 → 激素响应基因表达受损。",
        "FOXA3": "FoxA3：肝/代谢转录因子。敲除 → 代谢基因改变。",
        "FOXF1": "FoxF1：间质/血管发育。敲除 → 发育缺陷。",
        "FOXL2": "FoxL2：卵巢分化/性腺发育。敲除 → 卵巢分化异常。",
        "MEIS1": "HOX 共因子：造血/心脏发育。敲除 → 发育基因改变。",
        "LYL1": "碱性螺旋-环-螺旋转录因子：造血。敲除 → 造血分化异常。",
        "PRDM1": "Blimp-1：浆细胞分化主控。敲除 → B 细胞终末分化缺陷。",
        "BPGM": "2,3-BPG 变位酶：合成 2,3-BPG 调节血红蛋白氧亲和力。敲除 → 红细胞氧释放受损。",
        "NCL": "核仁蛋白：rRNA 加工/核糖体生成。敲除 → 蛋白质合成能力下降。",
        "COL1A1": "I 型胶原 α1：细胞外基质主要成分。敲除 → 基质结构缺陷。",
        "COL2A1": "II 型胶原：软骨主要成分。敲除 → 软骨发育缺陷。",
        "CLDN6": "紧密连接蛋白 claudin-6。敲除 → 上皮屏障功能改变。",
        "CNN1": "calponin-1：平滑肌收缩调节。敲除 → 平滑肌收缩调控改变。",
        "SLC38A2": "氨基酸转运体 SNAT2。敲除 → 氨基酸摄取/代谢信号改变（mTORC1 相关）。",
        "SLC4A1": "带 3 蛋白（Cl-/HCO3- 交换，红细胞）。敲除 → 红细胞离子交换缺陷。",
        "SLC6A9": "甘氨酸转运体 GLYT1。敲除 → 甘氨酸稳态改变。",
        "TMSB4X": "胸腺素 β4：肌动蛋白结合/细胞迁移。敲除 → 细胞骨架动力学改变。",
    }

    def gene_mech(gname):
        """基因敲除机制说明：命中 GENE_MECH 用精确条目；否则按通路生成简要机制。"""
        if gname in GENE_MECH:
            return GENE_MECH[gname]
        for k, gs in PW_GENES.items():
            if gname in gs:
                return (f"{gname} 参与「{PW_NAME.get(k, k)}」：该通路成员被敲除后，"
                        f"通路的信号传递随之改变（详见上方通路图与下方表达变化预测）。")
        return (f"{gname}：转录因子/调控因子类基因（当前词表多为分化与表观调控相关）。"
                f"敲除后其下游靶基因表达改变，具体方向见下方预测列表。")

    def pathway_of(gene_names_hit):
        for k, gs in PW_GENES.items():
            if any(g in gs for g in gene_names_hit):
                return k
        return "general"

    app = Flask(__name__, static_folder=None)

    def top_genes(pred, k=12):
        order = np.argsort(-pred)
        up = [{"g": gene_names[i], "v": float(pred[i])}
              for i in order[:k] if not gene_names[i].startswith("Gene_")]
        dn = [{"g": gene_names[i], "v": float(pred[i])}
              for i in order[-k:] if not gene_names[i].startswith("Gene_")]
        return up, dn

    @app.route("/health")
    def health():
        return jsonify({"ok": True, "drugs": len(chem_map)})

    @app.route("/genes")
    def genes():
        """返回基因扰动词表（stageA 训练过的基因）。"""
        return jsonify({"genes": sorted(gv_b.keys())})

    @app.route("/predict", methods=["POST"])
    def predict():
        d = request.get_json(force=True)
        drug, cell = d.get("drug"), int(d.get("cell", 0))
        info = chem_map.get(drug, {})
        sm = info.get("smiles", "")
        g = gcache.get(sm)
        if g is None:
            g = smiles_to_ecfp4(sm)
        if g is None or not g.any():
            return jsonify({"error": "药物构图失败"})
        with torch.no_grad():
            pred = effect.forward_compound(
                [g], torch.tensor([cell], device=DEVICE)).cpu().numpy()[0]
            z = unipert_p.encode_compound(
                [g], torch.zeros(1, dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
        z = z / (np.linalg.norm(z) + 1e-8)
        sim = P @ z
        targets = [{"g": pool_genes[i], "s": float(sim[i])}
                   for i in np.argsort(-sim)[:6]]
        up, dn = top_genes(pred)
        return jsonify({"drug": drug, "cell": CELL_NAMES[cell],
                        "smiles": sm, "target_anno": info.get("target", ""),
                        "pathway_anno": info.get("pathway", ""),
                        "targets": targets, "up": up, "down": dn})

    @app.route("/gene", methods=["POST"])
    def gene():
        d = request.get_json(force=True)
        gname, cell = d.get("gene"), int(d.get("cell", 0))
        gnorm = str(gname).strip().upper()
        gid = gv_b.get(gnorm)
        if gid is None:
            return jsonify({"error": f"基因 {gname} 不在词表"})
        with torch.no_grad():
            pred = effect.forward_gene(
                torch.tensor([gid], device=DEVICE),
                seqs=None, cell_line_idx=torch.tensor([cell], device=DEVICE)).cpu().numpy()[0]
        up, dn = top_genes(pred)
        pw = pathway_of([gnorm])   # 基因扰动 → 通路归属（18 通路 + general 兜底）
        return jsonify({"gene": gname, "cell": CELL_NAMES[cell],
                        "up": up, "down": dn,
                        "pathway": pw, "pathway_name": PW_NAME.get(pw, "多通路/未归类"),
                        "mech": gene_mech(gnorm),
                        "perturbation": "knockout",
                        "perturbation_note": "当前模型为 CRISPR 敲除（基因功能丧失）的预测效应"})

    @app.route("/gui/<path:fn>")
    def gui(fn):
        return send_from_directory(GUI, fn)

    @app.route("/custom_drug", methods=["POST"])
    def custom_drug():
        """未知药物：任意 SMILES → 实时构图 → 表达预测 + 靶点检索 + 通路归属。"""
        d = request.get_json(force=True)
        sm = (d.get("smiles") or "").strip()
        cell = int(d.get("cell", 0))
        if not sm:
            return jsonify({"error": "请输入 SMILES"})
        g = gcache.get(sm)
        if g is None:
            g = smiles_to_ecfp4(sm)
        if g is None or not g.any():
            return jsonify({"error": "SMILES 无法解析为分子图（请检查格式）"})
        with torch.no_grad():
            pred = effect.forward_compound(
                [g], torch.tensor([cell], device=DEVICE)).cpu().numpy()[0]
            z = unipert_p.encode_compound(
                [g], torch.zeros(1, dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
        z = z / (np.linalg.norm(z) + 1e-8)
        sim = P @ z
        targets = [{"g": pool_genes[i], "s": float(sim[i])}
                   for i in np.argsort(-sim)[:8]]
        pw = pathway_of([t["g"] for t in targets[:6]])
        up, dn = top_genes(pred)
        return jsonify({"smiles": sm, "cell": CELL_NAMES[cell],
                        "targets": targets, "up": up, "down": dn,
                        "pathway": pw, "pathway_name": PW_NAME.get(pw, "多通路/未归类")})

    # 三字母氨基酸缩写 → 单字母（肽类药物序列自动转换）
    AA3TO1 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
              "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
              "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
              "Tyr": "Y", "Val": "V", "Sec": "U", "Pyl": "O"}

    def _peptide_to_1letter(seq_raw):
        """识别肽序列（三字母缩写，如 Lys-Cys-Asn-NH₂ / {修饰}-Ala-...）并转单字母。
        返回 (单字母序列, 修饰信息列表) 或 (None, []) 表示不是肽格式。"""
        import re as _re
        mods = _re.findall(r"\{([^}]*)\}", seq_raw)
        core = _re.sub(r"\{[^}]*\}", "", seq_raw)
        core = _re.sub(r"-NH2$|-NH₂$|-OH$", "", core.strip())
        parts = [p for p in _re.split(r"[-_\s]+", core) if p]
        if not parts or not all(p in AA3TO1 for p in parts):
            return None, []
        return "".join(AA3TO1[p] for p in parts), mods

    @app.route("/custom_protein", methods=["POST"])
    def custom_protein():
        """未知蛋白/肽：任意序列 → ESM 实时编码 → 统一空间相似蛋白 + 表达预测 + 通路归属。
        自动识别三字母肽序列（如 Lys-Cys-Asn-…-NH₂）转单字母；N 端 {修饰} 提取但暂不参与编码。"""
        import re as _re
        d = request.get_json(force=True)
        raw = d.get("sequence") or ""
        seq1, mods = _peptide_to_1letter(raw)
        pep_note = ""
        if seq1:
            seq = seq1
            if mods:
                pep_note = f"识别为肽序列（{len(seq)} aa）；N 端修饰 {mods} 未参与编码（模型不含修饰基团表征）"
            else:
                pep_note = f"识别为肽序列（{len(seq)} aa，三字母缩写已转单字母）"
        else:
            seq = _re.sub(r"[^A-Za-z]", "", raw).strip().upper()
        cell = int(d.get("cell", 0))
        if len(seq) < 20:
            return jsonify({"error": f"序列太短（清洗后 {len(seq)} aa，至少 20）——请检查是否包含有效氨基酸字母"})
        with torch.no_grad():
            z = encode_unknown_protein(seq).cpu().numpy()[0]
        z = z / (np.linalg.norm(z) + 1e-8)
        sim = P @ z
        hits = [{"g": pool_genes[i], "s": float(sim[i])}
                for i in np.argsort(-sim)[:8]]
        pw = pathway_of([h["g"] for h in hits[:6]])
        with torch.no_grad():
            zt = torch.as_tensor(z, device=DEVICE).unsqueeze(0)
            pred = effect.head_gene(
                zt + effect.unipert.cell_line(torch.tensor([cell], device=DEVICE))
            ).cpu().numpy()[0]
        up, dn = top_genes(pred)
        return jsonify({"seq_len": len(seq), "cell": CELL_NAMES[cell],
                        "similar": hits, "up": up, "down": dn,
                        "pathway": pw, "pathway_name": PW_NAME.get(pw, "多通路/未归类"),
                        "pep_note": pep_note or None})

    @app.route("/interact", methods=["POST"])
    def interact():
        """未知互作：药物-蛋白 / 蛋白-蛋白 的统一空间互作打分。"""
        import re as _re
        d = request.get_json(force=True)
        mode = d.get("mode", "drug_protein")
        with torch.no_grad():
            if mode == "drug_protein":
                sm = (d.get("smiles") or "").strip()
                seq1, _m = _peptide_to_1letter(d.get("sequence") or "")
                seq = seq1 if seq1 else _re.sub(r"[^A-Za-z]", "", d.get("sequence") or "").strip().upper()
                g = gcache.get(sm)
                if g is None:
                    g = smiles_to_ecfp4(sm)
                if g is None or not g.any():
                    return jsonify({"error": "药物 SMILES 无法解析"})
                if len(seq) < 20:
                    return jsonify({"error": "蛋白序列太短"})
                zd = unipert_p.encode_compound(
                    [g], torch.zeros(1, dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
                zp = encode_unknown_protein(seq).cpu().numpy()[0]
            else:  # protein_protein
                sa1, _m1 = _peptide_to_1letter(d.get("seqA") or "")
                sb1, _m2 = _peptide_to_1letter(d.get("seqB") or "")
                sa = sa1 if sa1 else _re.sub(r"[^A-Za-z]", "", d.get("seqA") or "").strip().upper()
                sb = sb1 if sb1 else _re.sub(r"[^A-Za-z]", "", d.get("seqB") or "").strip().upper()
                if len(sa) < 20 or len(sb) < 20:
                    return jsonify({"error": "序列太短"})
                zd = encode_unknown_protein(sa).cpu().numpy()[0]
                zp = encode_unknown_protein(sb).cpu().numpy()[0]
        zd = zd / (np.linalg.norm(zd) + 1e-8)
        zp = zp / (np.linalg.norm(zp) + 1e-8)
        cos = float(np.dot(zd, zp))
        # 简单互作强度（与 905 池分布比，z 检验近似）
        pool_cos = P @ zp
        z_score = (cos - float(np.mean(pool_cos))) / max(float(np.std(pool_cos)), 1e-8)
        p_val = float(np.clip(1 - 0.5 * (1 + math.erf(z_score / np.sqrt(2))), 1e-10, 1.0))

        def _top_sim(z, k=3):
            s = P @ z
            return [{"g": pool_genes[i], "s": float(s[i])}
                    for i in np.argsort(-s)[:k]]

        return jsonify({"mode": mode, "similarity": cos, "z_score": z_score,
                        "p_value": p_val,
                        "similarA": _top_sim(zd), "similarB": _top_sim(zp),
                        "note": "统一空间余弦相似度；P 值基于 905 蛋白池分布的单侧 z 检验"})

    @app.route("/cascade", methods=["POST"])
    def cascade():
        """扰动跟踪链 v4：箭头方向全部由模型实时计算——
        起点 → 预测受影响蛋白(↑/↓ 带预测值) → 各蛋白的统一空间相似下游(→) → 表型。
        不再使用预制级联模板。"""
        import re as _re
        d = request.get_json(force=True)
        ctype = d.get("type", "drug")
        cell = int(d.get("cell", 0))
        val = (d.get("value") or "").strip()
        pool_set = set(pool_genes)

        def _struct_of(name):
            for tok in _re.findall(r"[A-Z][A-Z0-9]{2,}", str(name)):
                if tok in PROTEIN_PDB:
                    return True, PROTEIN_PDB[tok]
            return False, None

        def _target_info(t):
            if not t:
                return TARGET_LIB.get("Others", {})
            if t in TARGET_LIB:
                return TARGET_LIB[t]
            tl = t.lower()
            for k, v in TARGET_LIB.items():
                kw = k.split()[0].lower()
                if kw in tl or tl in k.lower():
                    return v
            return TARGET_LIB.get("Others", {})

        def _mk(stage, kind, name, mech, score=None, struct=None, has_struct=None, pdb_id=None):
            return {"stage": stage, "kind": kind, "name": name, "mech": mech,
                    "score": score, "struct": struct, "has_struct": has_struct,
                    "pdb_id": pdb_id, "dir": None, "v": None, "next": None}

        chain = []
        pw = "general"
        pw_name = "多通路/未归类"

        def _add(kind, name, mech, dir=None, v=None, score=None, struct=None, has_struct=None, pdb_id=None):
            nd = _mk(len(chain), kind, name, mech, score, struct, has_struct, pdb_id)
            nd["dir"] = dir
            nd["v"] = v
            if chain:
                chain[-1]["next"] = name
            chain.append(nd)
            return nd

        def _sim_next(gname, used):
            """统一空间实时算：池中该蛋白的相似下游（排除自身与已用节点）"""
            if gname not in pool_set:
                return None
            gi = pool_genes.index(gname)
            for si in np.argsort(-SIM[gi]):
                cand = pool_genes[si]
                if cand != gname and cand not in used:
                    return cand, float(SIM[gi][si])
            return None

        pred = None
        if ctype == "drug":
            known = val in chem_map
            info = chem_map.get(val, {}) if known else {}
            sm = info.get("smiles", "") if known else val
            tgt = info.get("target", "")
            ti = _target_info(tgt)
            g = gcache.get(sm)
            if g is None:
                g = smiles_to_ecfp4(sm)
            if g is None or not g.any():
                return jsonify({"error": "药物/SMILES 无法解析（已知药请从列表选，未知药请输入 SMILES）"})
            with torch.no_grad():
                pred = effect.forward_compound(
                    [g], torch.tensor([cell], device=DEVICE)).cpu().numpy()[0]
                z = unipert_p.encode_compound(
                    [g], torch.zeros(1, dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
            z = z / (np.linalg.norm(z) + 1e-8)
            sim = P @ z
            tgts = [pool_genes[i] for i in np.argsort(-sim)[:3]]
            pw = PATHWAY_MAP.get(info.get("pathway", ""), "general")
            if pw == "general":
                pw = pathway_of(tgts[:6] + [ti.get("name", "")])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            if known:
                _add("start", "药物 " + val, "真实靶点：" + ti.get("name", "—"))
                _add("target", ti.get("name", tgt or "未知靶点"),
                     "结合方式：" + ti.get("bind", "—") + "。" + ti.get("mech", ""),
                     struct=ti.get("struct", "—"))
            else:
                _add("start", "未知药物（SMILES）", "无实验靶点注释，实时检索预测靶点")
                _add("target", "预测靶点 " + tgts[0],
                     "实时计算：与 " + tgts[0] + " 统一空间最接近（相似度 " + format(float(sim[pool_genes.index(tgts[0])]), ".3f") + "），预测结合该类蛋白（线索级）",
                     struct="预测结合（无实验结构注释）")

        elif ctype == "gene":
            gnorm = val.upper()
            gid = gv_b.get(gnorm)
            if gid is None:
                return jsonify({"error": "基因 " + gnorm + " 不在词表"})
            with torch.no_grad():
                pred = effect.forward_gene(
                    torch.tensor([gid], device=DEVICE), seqs=None,
                    cell_line_idx=torch.tensor([cell], device=DEVICE)).cpu().numpy()[0]
            pw = pathway_of([gnorm])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            mech0 = gene_mech(gnorm)
            _add("start", gnorm + " 敲除（CRISPR）", mech0 or "基因功能丧失")

        else:  # protein
            seq1, mods = _peptide_to_1letter(val)
            seq = seq1 if seq1 else _re.sub(r"[^A-Za-z]", "", val).strip().upper()
            if len(seq) < 20:
                return jsonify({"error": "序列太短（清洗后 " + str(len(seq)) + " aa，至少 20）"})
            with torch.no_grad():
                z = encode_unknown_protein(seq).cpu().numpy()[0]
            z = z / (np.linalg.norm(z) + 1e-8)
            sim = P @ z
            hits = [pool_genes[i] for i in np.argsort(-sim)[:3]]
            pw = pathway_of(hits[:6])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            with torch.no_grad():
                zt = torch.as_tensor(z, device=DEVICE).unsqueeze(0)
                pred = effect.head_gene(
                    zt + effect.unipert.cell_line(torch.tensor([cell], device=DEVICE))
                ).cpu().numpy()[0]
            _add("start", "输入蛋白/肽（" + str(len(seq)) + " aa）", "序列经 ESM 编码，实时检索最相似蛋白")
            _add("target", "最相似蛋白 " + hits[0],
                 "与 " + hits[0] + " 统一空间最接近（相似度 " + format(float(sim[pool_genes.index(hits[0])]), ".3f") + "），可能行使相似功能")

        # ---- 模型实时算链：预测受影响蛋白（↑/↓）→ 相似下游（→）----
        if pred is not None:
            used = {nd["name"] for nd in chain}
            up, dn = top_genes(pred, k=20)
            ups = [u for u in up if u["g"] in pool_set and u["g"] not in used][:2]
            dns = [u for u in dn if u["g"] in pool_set and u["g"] not in used][:1]
            for u in ups:
                hs, pdb = _struct_of(u["g"])
                _add("transcript", "↑ " + u["g"], "模型预测：表达上调（变化值 " + format(u["v"], "+.3f") + "）——扰动信号的第一个落点",
                     dir="up", v=u["v"], score=0.7, has_struct=hs, pdb_id=pdb)
            for u in dns:
                hs, pdb = _struct_of(u["g"])
                _add("transcript", "↓ " + u["g"], "模型预测：表达下调（变化值 " + format(u["v"], "+.3f") + "）",
                     dir="dn", v=u["v"], score=0.6, has_struct=hs, pdb_id=pdb)
            if ups:
                used = {nd["name"] for nd in chain}
                nxt = _sim_next(ups[0]["g"].lstrip("↑ "), used)
                if nxt:
                    hs, pdb = _struct_of(nxt[0])
                    _add("protein", "→ " + nxt[0],
                         "模型实时算的下一步：与 " + ups[0]["g"].lstrip("↑ ") + " 在统一空间最相似（相似度 " + format(nxt[1], ".3f") + "），扰动信号可能沿此传递",
                         dir="sim", v=nxt[1], score=0.5, has_struct=hs, pdb_id=pdb)

        _add("phenotype", pw_name,
             "由以上预测转录变化推断的表型：" + PW_PHENO.get(pw, "转录组整体重编程 → 细胞表型改变"))

        return jsonify({"type": ctype, "cell": CELL_NAMES[cell], "chain": chain,
                        "pathway": pw, "pathway_name": pw_name})

    @app.route("/structure/<name>")
    def structure(name):
        """返回蛋白的 PDB 结构文本（3Dmol 加载用）。name=基因符号或 PDB ID。"""
        pdb_id = PROTEIN_PDB.get(str(name).upper(), str(name).upper())
        path = os.path.join(STRUCT, pdb_id + ".pdb")
        if not os.path.isfile(path):
            return jsonify({"error": f"无 {name} 的结构"}), 404
        return send_from_directory(STRUCT, pdb_id + ".pdb", mimetype="text/plain")

    @app.route("/interface/<name>")
    def interface(name):
        """蛋白-蛋白复合物 3D 数据：链分组 + 界面残基（互作位点高亮）。"""
        name_u = str(name).strip().upper()
        ipath = os.path.join(GUI, "interface_residues.json")
        if not os.path.isfile(ipath):
            return jsonify({"error": "界面数据未生成"}), 404
        iface = json.load(open(ipath, encoding="utf-8"))
        for pdb, v in iface.items():
            if v["proteinA"].upper() == name_u or v["proteinB"].upper() == name_u:
                return jsonify({"pdb": pdb, "proteinA": v["proteinA"],
                                "proteinB": v["proteinB"],
                                "chainA": v["chainA"], "chainB": v["chainB"],
                                "ifaceA": v["ifaceA"], "ifaceB": v["ifaceB"],
                                "mech": PPI_MECH.get(pdb, "")})
        return jsonify({"error": "该蛋白暂无复合物结构"}), 404

    @app.route("/params")
    def params():
        """论文级互作参数（ΔG/亲和力/界面面积/SASA/氢键/P 值）。"""
        p = os.path.join(GUI, "interaction_params.json")
        if not os.path.isfile(p):
            return jsonify({"error": "参数未生成"})
        return send_from_directory(GUI, "interaction_params.json")

    print(f"[OK] API 就绪：http://localhost:{port}/gui/workbench.html", flush=True)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()
    app = create_app(args.port, args.device)
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
