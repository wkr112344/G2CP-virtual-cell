"""
拉取蛋白/复合物三维结构（P7 · 论文级 3D 互作）
=====================================================================
对通路蛋白建「代表性 PDB 结构」映射表（真实实验结构，含抑制剂/配体的
复合物结构优先），从 RCSB 拉取存 data/gui/structures/{id}.pdb。
另含蛋白-蛋白复合物（互作 3D 展示用）。

用法：python unipret/fetch_structures.py
"""
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "gui", "structures")
RCSB = "https://files.rcsb.org/download/{}.pdb"

# 通路蛋白 → 代表性 PDB（真实实验结构；含配体/抑制剂的优先）
PROTEIN_PDB = {
    # PKA/cAMP
    "ADRB2": "2RH1",        # β2-AR 与激动剂复合物
    "GNAS": "1AZT",         # Gαs 三聚体
    "ADCY": "1CJK",         # 腺苷酸环化酶催化域（含 Gsα 与 forskolin）
    "PRKAR2A": "1CX4",      # PKA 调节亚基（RIIα）二聚体
    "PRKACA": "1ATP",       # PKA 催化亚基 Cα（经典激酶结构）
    "CREB1": "1DH3",        # CREB bZIP 结合 DNA（CRE 元件）
    # EGFR-MAPK
    "EGFR": "1M17",         # EGFR 激酶域 + 厄洛替尼（药物结合位点！）
    "GRB2": "1GRI",         # GRB2 SH2 域
    "SOS1": "1XDW",         # SOS1
    "KRAS": "1CRR",         # KRAS-GDP（开关 I/II）
    "BRAF": "4MNF",         # BRAF 激酶域（V600E）
    "MAP2K1": "3SLS",       # MEK1 激酶域
    "MAPK1": "1TVO",        # ERK2 激酶域
    "ELK1": "1DUX",         # ELK1 ETS 域结合 DNA
    # PI3K-AKT
    "IGF1R": "1K3A",        # IGF-1R 激酶域
    "PIK3CA": "4JPS",       # p110α/p85α 复合物
    "AKT1": "3CQW",         # AKT1 激酶域
    "MTOR": "4JSN",         # mTORC1（FRB + 激酶域，含雷帕霉素）
    # JAK-STAT
    "IFNAR": "3SE4",        # IFNAR1 胞外域
    "JAK1": "4L00",         # JAK1 激酶域
    "JAK2": "2B7A",         # JAK2 激酶域
    "STAT1": "1BF5",        # STAT1 核心（同二聚体）
    "STAT3": "1BG1",        # STAT3 核心（同二聚体）
    # HDAC 表观
    "HDAC1": "4BKX",        # HDAC1（与抑制剂复合物）
    "HDAC2": "4LXZ",        # HDAC2（与 SAHA 复合物）
    "EP300": "4PZR",        # p300 HAT 催化域
    # 互作复合物（蛋白-蛋白 3D）
    "_COMPLEX_PKA": "2QCS",     # PKA 全酶 RIIβ-Cα 复合物（cAMP 结合）
    "_COMPLEX_RASRAF": "4G0N",  # KRAS-RAF1 RBD 复合物
    "_COMPLEX_EGFREXTRA": "1IVO",  # EGF-EGFR 胞外二聚体
    "_COMPLEX_CREBCBP": "1KDX", # pKID(CREB)-KIX(CBP) 复合物（Ser133 识别）
    "_COMPLEX_JAK2": "4FVP",    # JAK2 激酶域-抑制肽
}

GENES = {"ADRB2": "ADRB2", "GNAS": "GNAS", "ADCY": "ADCY1", "PRKAR2A": "PRKAR2A",
         "PRKACA": "PRKACA", "CREB1": "CREB1", "EGFR": "EGFR", "GRB2": "GRB2",
         "SOS1": "SOS1", "KRAS": "KRAS", "BRAF": "BRAF", "MAP2K1": "MAP2K1",
         "MAPK1": "MAPK1", "ELK1": "ELK1", "IGF1R": "IGF1R", "PIK3CA": "PIK3CA",
         "AKT1": "AKT1", "MTOR": "MTOR", "IFNAR": "IFNAR1", "JAK1": "JAK1",
         "JAK2": "JAK2", "STAT1": "STAT1", "STAT3": "STAT3", "HDAC1": "HDAC1",
         "HDAC2": "HDAC2", "EP300": "EP300"}


def fetch(pdb_id):
    dest = os.path.join(OUT, pdb_id + ".pdb")
    if os.path.isfile(dest) and os.path.getsize(dest) > 5000:
        return "cached"
    url = RCSB.format(pdb_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "virtual-cell/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) > 5000 and b"ATOM" in data:
            with open(dest, "wb") as f:
                f.write(data)
            return f"ok({len(data)//1024}KB)"
        return "empty"
    except Exception as e:
        return f"fail:{e}"


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f">>> 拉取 {len(PROTEIN_PDB)} 个结构 -> {OUT}", flush=True)
    ok = 0
    for name, pdb in PROTEIN_PDB.items():
        r = fetch(pdb)
        if r.startswith("ok") or r == "cached":
            ok += 1
        print(f"    {name:16s} {pdb:6s} {r}", flush=True)
        time.sleep(0.3)
    print(f"✅ 完成：{ok}/{len(PROTEIN_PDB)} 结构可用", flush=True)


if __name__ == "__main__":
    main()
