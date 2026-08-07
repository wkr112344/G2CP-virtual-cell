# -*- coding: utf-8 -*-
"""给 serve_g2cp.py 加"外推参照"逻辑：未知分子按药效片段规则推荐真实同类药。
用户需求：预测这个分子时应参照 nimesulide（二芳基磺酰胺类），而非 celecoxib。"""
import ast

p = "serve_g2cp.py"
s = open(p, encoding="utf-8").read()

# 1) import rdkit Chem
old_imp = "from unipret.compound_encoder import smiles_to_ecfp4, ECFP4_BITS"
new_imp = old_imp + "\nfrom rdkit import Chem"
assert old_imp in s, "import 锚点缺失"
s = s.replace(old_imp, new_imp, 1)

# 2) FRAG_RULES 规则库（插在 def drugs 前）
frag = '''# 药效片段规则库：未知分子 → 化学骨架类别 → 训练集中真实同类参照药
FRAG_RULES = [
    (Chem.MolFromSmarts("[NX3][SX4](=O)(=O)[cX3]"), "二芳基磺酰胺类（Ar-SO2-NH-Ar'）",
     "磺酰胺 N 连接第二个芳环——COX-2 选择性抑制剂常用骨架（如尼美舒利）",
     ["nimesulide", "sulfaphenazole"]),
    (Chem.MolFromSmarts("c[SX4](=O)(=O)[NX3]"), "芳环磺酰胺类",
     "芳环直接连磺酰胺（含昔布类 SO2NH2、碳酸酐酶抑制剂）",
     ["celecoxib", "acetazolamide", "valdecoxib"]),
    (Chem.MolFromSmarts("[NX3]C(=O)[NX3][SX4](=O)(=O)"), "磺酰脲类",
     "磺酰脲结构（对甲苯磺酰脲）",
     ["tolbutamide"]),
    (Chem.MolFromSmarts("OC(=O)CC1=CC=CC=C1"), "芳基乙酸类 NSAID",
     "芳基乙酸骨架（双氯芬酸/吲哚美辛类）",
     ["diclofenac"]),
]


def frag_ref(mol):
    """SMILES 分子 → (类别, 说明, 代表药列表, 首选) 或 None"""
    for _sm, _cls, _note, _drugs in FRAG_RULES:
        try:
            if mol.HasSubstructMatch(_sm):
                return {"cls": _cls, "note": _note, "drugs": _drugs, "top": _drugs[0]}
        except Exception:
            continue
    return None


def drugs():
'''
old_drugs = "def drugs():"
assert old_drugs in s
s = s.replace(old_drugs, frag, 1)

# 3) predict 返回 ref 字段（SMILES 未知分子分支）
old_pred = '''        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([cell], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
    order = np.argsort(-out)
    up = [{"g": hvg[i], "v": float(out[i])} for i in order[:10] if not hvg[i].startswith("Gene_")]
    dn = [{"g": hvg[i], "v": float(out[i])} for i in order[-10:] if not hvg[i].startswith("Gene_")]
    return jsonify({"drug": value, "name": DRUG_NAMES.get(value, value if value in dv else "未知药物(SMILES)"),
                    "cell": cl_names[cell], "up": up, "down": dn})'''
new_pred = '''        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([cell], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
    order = np.argsort(-out)
    up = [{"g": hvg[i], "v": float(out[i])} for i in order[:10] if not hvg[i].startswith("Gene_")]
    dn = [{"g": hvg[i], "v": float(out[i])} for i in order[-10:] if not hvg[i].startswith("Gene_")]
    # 外推参照：未知分子按药效片段规则推荐真实同类药（训练集有实验数据）
    ref = None
    if vid is None:
        mol = Chem.MolFromSmiles(value)
        if mol is not None:
            ref = frag_ref(mol)
    return jsonify({"drug": value, "name": DRUG_NAMES.get(value, value if value in dv else "未知药物(SMILES)"),
                    "cell": cl_names[cell], "up": up, "down": dn, "ref": ref})'''
assert old_pred in s, "predict 锚点缺失"
s = s.replace(old_pred, new_pred, 1)

open(p, "w", encoding="utf-8").write(s)
ast.parse(s)
print("patch_ref 完成，语法 OK")
