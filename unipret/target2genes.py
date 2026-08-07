"""
sciPlex3 target 类别 → 基因符号 映射（P3：扩 UniPert 对比预训练 CPI 正对）
=====================================================================
sciPlex3 obs.target 是靶点类别名（如 "Aurora Kinase"、"EGFR,HER2"），
不是标准基因符号。本映射把类别解析成基因列表，用于构造"药物→靶点基因"
正样本对（InfoNCE 的正对）。

规则：优先精确实体（EGFR→EGFR），类别展开（CDK→CDK1/2/4/6）。
"""
import re

TARGET2GENES = {
    "EGFR": ["EGFR"], "HER2": ["ERBB2"], "ERBB2": ["ERBB2"], "EGFR,HER2": ["EGFR", "ERBB2"],
    "c-RET": ["RET"], "RET": ["RET"], "c-Met": ["MET"], "IGF-1R": ["IGF1R"],
    "Trk receptor": ["NTRK1"], "FAAH": ["FAAH"], "IDH1": ["IDH1"],
    "Aurora Kinase": ["AURKA", "AURKB"], "Bcr-Abl": ["BCR", "ABL1"],
    "FLT3": ["FLT3"], "JAK": ["JAK1", "JAK2", "JAK3"], "EGFR,JAK": ["EGFR", "JAK1", "JAK2"],
    "CDK": ["CDK1", "CDK2", "CDK4", "CDK6"], "Aurora Kinase,CDK": ["AURKA", "AURKB", "CDK1", "CDK2"],
    "HDAC": ["HDAC1", "HDAC2", "HDAC3", "HDAC6"], "EGFR,HDAC,HER2": ["EGFR", "HDAC1", "HDAC2", "ERBB2"],
    "HDAC,PI3K": ["HDAC1", "HDAC2", "PIK3CA", "PIK3CB"],
    "PARP": ["PARP1", "PARP2"], "MEK": ["MAP2K1", "MAP2K2"],
    "VEGFR": ["KDR", "FLT1"], "FGFR": ["FGFR1", "FGFR2"], "FGFR,VEGFR": ["FGFR1", "FGFR2", "KDR", "FLT1"],
    "c-RET,VEGFR": ["RET", "KDR", "FLT1"], "VEGFR,PDGFR,c-Kit": ["KDR", "PDGFRA", "KIT"],
    "Aurora Kinase,VEGFR": ["AURKA", "KDR", "FLT1"], "Aurora Kinase,Bcr-Abl,FLT3": ["AURKA", "BCR", "ABL1", "FLT3"],
    "Aurora Kinase,Bcr-Abl,c-RET,FGFR": ["AURKA", "ABL1", "RET", "FGFR1", "FGFR2"],
    "Aurora Kinase,Bcr-Abl,JAK": ["AURKA", "ABL1", "JAK1", "JAK2"],
    "PDGFR,Raf,VEGFR": ["PDGFRA", "BRAF", "KDR", "FLT1"], "Raf": ["BRAF", "RAF1"],
    "c-Kit": ["KIT"], "PI3K": ["PIK3CA", "PIK3CB", "PIK3CD"], "Src,Sirtuin,PKC,PI3K": ["SRC", "SIRT1", "PRKCA", "PIK3CA"],
    "PKA,EGFR,PKC": ["PRKACA", "EGFR", "PRKCA"], "mTOR": ["MTOR"], "Src": ["SRC"],
    "HSP (e.g. HSP90)": ["HSP90AA1"], "HSP90": ["HSP90AA1"],
    "Topoisomerase": ["TOP1", "TOP2A"], "Sirtuin": ["SIRT1", "SIRT2"], "Autophagy,Sirtuin": ["ATG7", "SIRT1"],
    "Beta Amyloid,Gamma-secretase": ["APP", "PSEN1"], "Bcl-2": ["BCL2"],
    "Estrogen/progestogen Receptor": ["ESR1", "ESR2", "PGR"], "Aromatase": ["CYP19A1"],
    "TGF-beta/Smad": ["TGFBR1", "SMAD2", "SMAD3"], "Pim": ["PIM1"], "PLK": ["PLK1"],
    "AMPK": ["PRKAA1"], "Telomerase": ["TERT"], "Histamine Receptor": ["HRH1"],
    "Dopamine Receptor": ["DRD2"], "MT Receptor": ["MTNR1A", "MTNR1B"], "Lipoxygenase": ["ALOX5"],
    "Histone Methyltransferase": ["EZH2", "DOT1L"], "Histone Demethylase": ["KDM1A"],
    "DNA alkylator": ["MGMT"], "DNA/RNA Synthesis": ["TUBB", "TUBB4B"], "Microtubule": ["TUBB", "TUBA1A"],
    "Autophagy": ["ATG7"], "Proteasome": ["PSMB1", "PSMB5"], "Others": [],
}


def target_to_genes(target_str):
    """解析 target 注释 → 基因符号列表（去重，保持顺序）。"""
    if not target_str:
        return []
    target_str = target_str.strip()
    if target_str in TARGET2GENES:
        return list(TARGET2GENES[target_str])
    out = []
    for part in re.split(r"[,\+/]", target_str):
        part = part.strip()
        genes = TARGET2GENES.get(part, [])
        if not genes and part and part[0].isupper() and not any(
                c in part for c in " -()."):
            # 兜底：看起来像单个基因符号（如 'AURKA'）
            if re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", part):
                genes = [part]
        out.extend(genes)
    seen, res = set(), []
    for g in out:
        if g and g not in seen:
            seen.add(g); res.append(g)
    return res


def build_sciplex3_cpi(chem_map, gene2idx, drug_index_by_name):
    """从 sciplex3_chems.json 构造 CPI 正对。
    返回 (pairs[(gene_idx, drug_idx)], meta[(gene, drug_name, smiles)])。
    只保留 gene2idx 里已有的基因；drug_index_by_name: {name.lower(): drug_idx}。
    """
    pairs, meta = [], []
    for name, v in chem_map.items():
        di = drug_index_by_name.get(name.lower())
        if di is None:
            continue
        for g in target_to_genes(v.get("target", "")):
            gi = gene2idx.get(g)
            if gi is not None:
                pairs.append((gi, di))
                meta.append((g, name, v.get("smiles", "")))
    return pairs, meta
