"""
UniPert 官方接口 —— 输入适配器（IO adapters）
============================================
把官方 UniPert 包支持的输入格式，解析成本骨架能吃的格式：
  - 基因：FASTA 文件 / 基因名列表 / UniProt id 列表 / 本地 proteins 字典
  - 化合物：SMILES 字典 / SMILES 列表 / .csv（多列名兼容）/ .txt（name\\tSMILES）/ 化合物名列表
  - 本地名↔序列、名↔SMILES 解析器（来自本项目的 dataset.json）

注意：官方包会联网去 UniProt / PubChem 取序列和 SMILES；本机网络受限，
所以本地解析器优先用 dataset.json 里已经抓好的真实数据（218 药 + 20 基因序列），
查不到的就标记为 invalid（与官方返回 invalid_inputs 的语义一致）。
"""
import csv
import os
import json
import warnings


# ----------------------------------------------------------- 本地数据加载
def _default_dataset_json():
    """找到项目里的 dataset.json（真实药 + 真实基因序列）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "dataset.json"),
        os.path.join(os.getcwd(), "dataset.json"),
        os.path.join(os.getcwd(), "virtual-cell", "dataset.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def load_local_maps(dataset_json=None):
    """
    返回 (gene_to_seq, name_to_smiles, name_to_genes)
      gene_to_seq : 基因符号 -> AA 序列（来自 proteins）
      name_to_smiles : 药物名(大写) -> SMILES（来自 drugs）
      name_to_genes : 药物名(大写) -> [靶基因符号]（来自 drugs.targets）
    """
    gene_to_seq, name_to_smiles, name_to_genes = {}, {}, {}
    path = dataset_json or _default_dataset_json()
    if path is None or not os.path.isfile(path):
        warnings.warn("未找到 dataset.json，本地名→序列/SMILES 解析器为空（仅 FASTA/SMILES 输入可用）。")
        return gene_to_seq, name_to_smiles, name_to_genes
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    for sym, meta in (d.get("proteins") or {}).items():
        if isinstance(meta, dict) and meta.get("seq"):
            gene_to_seq[sym] = meta["seq"]
    for dr in (d.get("drugs") or []):
        nm = (dr.get("name") or "").strip().upper()
        sm = dr.get("smiles")
        if nm and sm:
            name_to_smiles[nm] = sm
            tg = dr.get("targets") or []
            name_to_genes[nm] = tg
    return gene_to_seq, name_to_smiles, name_to_genes


# ----------------------------------------------------------- FASTA 解析
def parse_fasta(path):
    """返回 dict: 标签 -> 序列字符串（与官方 FastaBatchedDataset 行为一致，去重保序）。"""
    seqs = {}
    cur_label, buf = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_label is not None:
                    seqs.setdefault(cur_label, "".join(buf))
                cur_label = line[1:].strip().split()[0]  # 取 > 后第一个词当 id
                buf = []
            else:
                buf.append(line)
    if cur_label is not None:
        seqs.setdefault(cur_label, "".join(buf))
    return seqs


# ----------------------------------------------------------- SMILES 文件解析
def _guess_delimiter(sample_line):
    for dl in [",", ";", "\t", "|"]:
        if dl in sample_line:
            return dl
    return ","


def _find_smiles_col(cols):
    for c in cols:
        if "smil" in c.lower():
            return c
    return None


def _find_name_col(cols, smiles_col):
    prefs = ["cmpd", "compound", "drug", "name", "id", "perturb"]
    for c in cols:
        cl = c.lower()
        if c == smiles_col:
            continue
        for p in prefs:
            if p in cl:
                return c
    # 退路：取非 smiles 的第一列
    for c in cols:
        if c != smiles_col:
            return c
    return None


def parse_smiles_csv(path):
    """
    兼容官方常见列名：Compound/SMILES、Name/Smiles、cmpdname/canonicalsmiles 等。
    返回 dict: 化合物名 -> SMILES。
    """
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        first = f.readline()
        f.seek(0)
        dl = _guess_delimiter(first)
        reader = csv.DictReader(f, delimiter=dl)
        cols = reader.fieldnames or []
        sc = _find_smiles_col(cols)
        nc = _find_name_col(cols, sc)
        if sc is None or nc is None:
            raise ValueError(f"无法在 CSV 列 {cols} 中识别 (name, SMILES) 两列。")
        for row in reader:
            nm = (row.get(nc) or "").strip()
            sm = (row.get(sc) or "").strip()
            if nm and sm:
                out[nm] = sm
    return out


def parse_smiles_txt(path):
    """官方 .txt：两列 化合物名\\tSMILES（tab 分隔）。返回 dict。"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()  # 退路：空格分隔
            if len(parts) >= 2:
                nm, sm = parts[0].strip(), parts[1].strip()
                if nm and sm:
                    out[nm] = sm
    return out


# ----------------------------------------------------------- 统一入口
def resolve_compound_inputs(compound_names=None, compound_dict=None,
                             csv_file=None, smiles_list=None,
                             name_to_smiles=None):
    """
    把官方 encode_compounds 的多种入参，统一成 dict: 名 -> SMILES。
    返回 (resolved_dict, invalid_names)
    """
    resolved, invalid = {}, []
    if compound_dict:
        for k, v in compound_dict.items():
            if v:
                resolved[str(k)] = v
            else:
                invalid.append(str(k))
    if csv_file:
        resolved.update(parse_smiles_csv(csv_file))
    if smiles_list:
        for i, sm in enumerate(smiles_list):
            if sm:
                resolved[f"compound_{i}"] = sm
    if compound_names:
        n2s = name_to_smiles or {}
        for nm in compound_names:
            key = str(nm).strip().upper()
            if key in n2s:
                resolved[str(nm)] = n2s[key]
            else:
                invalid.append(str(nm))
    return resolved, invalid


def resolve_gene_inputs(gene_names=None, uniprot_ids=None, fasta_file=None,
                        gene_to_seq=None):
    """
    把官方 encode_genes 的多种入参，统一成 dict: 标签 -> 序列。
    返回 (resolved_dict, invalid_labels)
    """
    resolved, invalid = {}, []
    local = dict(gene_to_seq or {})
    if fasta_file:
        local.update(parse_fasta(fasta_file))
    def _add(label, seq):
        if seq:
            resolved[label] = seq
        else:
            invalid.append(label)
    if gene_names:
        for g in gene_names:
            _add(str(g), local.get(str(g).strip().upper()) or local.get(str(g)))
    if uniprot_ids:
        # 本地没有 UniProt->序列映射（需联网），查不到即 invalid
        for u in uniprot_ids:
            _add(str(u), local.get(str(u)))
    return resolved, invalid
