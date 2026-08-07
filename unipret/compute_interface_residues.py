"""
计算蛋白-蛋白复合物的界面残基（P7b：双蛋白 3D 互作位点）
=====================================================================
对复合物 PDB：链分组（A 组/B 组）+ 链间距离<5Å 的残基（界面位点）
产物：data/gui/interface_residues.json
  { pdb: {proteinA, proteinB, chainA:[...], chainB:[...], ifaceA:[resn..], ifaceB:[resn..]} }
"""
import os
import sys
import json

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
STRUCT = os.path.join(BASE, "data", "gui", "structures")
OUT = os.path.join(BASE, "data", "gui", "interface_residues.json")

# 复合物 → 蛋白对（与 interaction_params 一致）
PPI = [
    ("PRKAR2A", "PRKACA", "2QCS"),
    ("KRAS", "BRAF", "4G0N"),
    ("CREB1", "EP300", "1KDX"),
    ("EGFR", "EGFR", "1IVO"),
]


def compute(pdb_path):
    from Bio.PDB import PDBParser
    p = PDBParser(QUIET=True)
    s = p.get_structure("x", pdb_path)
    chains = sorted({r.get_parent().id for r in s[0].get_residues()})
    if len(chains) < 2:
        return None
    half = len(chains) // 2
    A, B = chains[:half], chains[half:]
    # 收集每链残基的原子
    atoms = {}
    for ch in chains:
        for r in s[0][ch].get_residues():
            if r.id[0] != " ":
                continue
            for a in r:
                atoms.setdefault((ch, r.id[1]), []).append(a.coord)
    ifaceA, ifaceB = set(), set()
    for ca, ra in atoms:
        if ca not in A:
            continue
        for cb, rb in atoms:
            if cb not in B:
                continue
            close = False
            for c1 in atoms[(ca, ra)][::3]:
                for c2 in atoms[(cb, rb)][::3]:
                    if np.linalg.norm(c1 - c2) < 5.0:
                        close = True
                        break
                if close:
                    break
            if close:
                ifaceA.add(ra)
                ifaceB.add(rb)
    return {"chainA": A, "chainB": B,
            "ifaceA": sorted(ifaceA), "ifaceB": sorted(ifaceB)}


def main():
    out = {}
    for pa, pb, pdb in PPI:
        path = os.path.join(STRUCT, pdb + ".pdb")
        if not os.path.isfile(path):
            continue
        r = compute(path)
        if not r:
            continue
        out[pdb] = {"proteinA": pa, "proteinB": pb, **r}
        print(f"    {pdb} {pa}-{pb}: 链A={r['chainA']} 界面残基 {len(r['ifaceA'])} 个"
              f" | 链B={r['chainB']} 界面残基 {len(r['ifaceB'])} 个", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"✅ interface_residues.json（{len(out)} 复合物）", flush=True)


if __name__ == "__main__":
    main()
