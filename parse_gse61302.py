# -*- coding: utf-8 -*-
"""解析 GSE61302(ASC 脂肪分化) → 探针表达 → 基因符号 → 验证脂肪标志基因。

输出: data/geo_asc/gse61302_gene_expr.json
- {'samples': [...], 'expr': {gene_symbol: [15个值]}, 'labels': [未分化/7天/21天]}
验证: FABP4/PPARG/LPL/ADIPOQ 在分化样本中是否上调(数据质量检查)
"""
import sys, os, gzip, json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(BASE, "data", "geo_asc")

# 1) 读 Series Matrix, 提取表达矩阵
lines = None
with gzip.open(os.path.join(GEO, "GSE61302_series_matrix.txt.gz"), "rt", errors="ignore") as f:
    lines = f.readlines()

# 样本顺序
sample_ids = []
start = None
for i, l in enumerate(lines):
    if l.startswith("!Sample_geo_accession"):
        sample_ids = l.rstrip().split("\t")[1:]
        sample_ids = [s.strip('"') for s in sample_ids]
    if l.startswith("!series_matrix_table_begin"):
        start = i + 1
        break
print(f"样本 {len(sample_ids)}: {sample_ids}")

# 样本分组(从 Sample_title 拿)
titles = []
for l in lines:
    if l.startswith("!Sample_title"):
        titles = l.rstrip().split("\t")[1:]
        titles = [t.strip('"') for t in titles]
        break
print(f"样本标题: {titles}")

# 表达矩阵
probes, matrix = [], []
for l in lines[start:]:
    if l.startswith("!series_matrix_table_end"):
        break
    parts = l.rstrip().split("\t")
    if parts[0].strip('"') == "ID_REF":
        continue
    probes.append(parts[0].strip('"'))
    matrix.append([float(x) for x in parts[1:]])
matrix = np.array(matrix)
print(f"探针 {len(probes)} × 样本 {matrix.shape[1]}")
print(f"表达值范围: {matrix.min():.1f} ~ {matrix.max():.1f}")

# 2) 探针 → 基因符号 (mygene)
import mygene
mg = mygene.MyGeneInfo()
out = mg.querymany(probes, scopes="reporter", fields="symbol", verbose=False, species="human")
probe2sym = {}
for r in out:
    q = r.get("query")
    sym = r.get("symbol")
    if q and sym:
        if isinstance(sym, list):
            sym = sym[0]
        probe2sym[q] = sym
print(f"探针→基因映射: {len(probe2sym)}/{len(probes)}")

# 3) 基因表达(多探针取均值)
gene_expr = {}
for i, p in enumerate(probes):
    sym = probe2sym.get(p)
    if not sym:
        continue
    if sym in gene_expr:
        gene_expr[sym].append(matrix[i])
    else:
        gene_expr[sym] = [matrix[i]]
gene_mean = {g: np.mean(v, axis=0) for g, v in gene_expr.items()}
print(f"唯一基因 {len(gene_mean)}")

# 4) 验证脂肪标志基因
print("\n=== 脂肪标志基因在 ASC 分化中的表达(未分化 vs 7天 vs 21天) ===")
targets = ["PPARG", "FABP4", "LPL", "ADIPOQ", "CEBPA", "CD36", "SREBF1", "PLIN1", "LEP", "DGAT1"]
# 样本分组: 标题含 undifferentiated = 未分化; 7day / 21day
grp = []
for t in titles:
    tl = t.lower()
    if "undiff" in tl:
        grp.append("undiff")
    elif "7day" in tl or "7 day" in tl:
        grp.append("d7")
    else:
        grp.append("d21")
for g in targets:
    if g in gene_mean:
        v = gene_mean[g]
        u = np.mean(v[[i for i, x in enumerate(grp) if x == "undiff"]])
        d7 = np.mean(v[[i for i, x in enumerate(grp) if x == "d7"]])
        d21 = np.mean(v[[i for i, x in enumerate(grp) if x == "d21"]])
        print(f"  {g}: 未分化 {u:.0f} | 7天 {d7:.0f} | 21天 {d21:.0f} | 21天/未分化 {d21/u:.2f}x")
    else:
        print(f"  {g}: ❌ 无数据")

# 5) 保存
json.dump({"samples": sample_ids, "labels": grp,
           "expr": {g: v.tolist() for g, v in gene_mean.items()}},
          open(os.path.join(GEO, "gse61302_gene_expr.json"), "w"))
print(f"\n已存 {os.path.join(GEO, 'gse61302_gene_expr.json')}")
