"""
P2：把 sciPlex3 全部化合物补上 SMILES（用自带 chembl-ID 直查 ChEMBL）
=====================================================================
背景：阶段 B 之前只用 51 药，因为靠"药名 token 匹配"本地 218 药。
sciPlex3 obs 自带 chembl-ID 列（153 个独特 id）——直接用 ChEMBL REST 查
canonical_smiles，绕开名字匹配，目标 ~150 化合物。

产物：sciplex3_chems.json = {perturbation: {chembl_id, smiles, target, pathway}}
用法：python unipret/fetch_sciplex3_chems.py
"""
import os
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import anndata as ad
import pandas as pd

H5AD = "C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sciplex3_chems.json")
TMP = OUT + ".part"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/molecule/{}.json"


def _get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "virtual-cell/1.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                return r.read().decode()
        except Exception as e:
            wait = 1.5 ** (i + 1)
            time.sleep(wait)
    return None


def _fetch_one(cid):
    txt = _get(CHEMBL_API.format(cid))
    if not txt:
        return cid, None
    try:
        j = json.loads(txt)
        sm = (j.get("molecule_structures") or {}).get("canonical_smiles")
        return cid, sm
    except Exception:
        return cid, None


def main():
    print(">>> 提取 sciPlex3 perturbation -> chembl-id 映射", flush=True)
    ds = ad.read_h5ad(H5AD, backed="r")
    obs = ds.obs
    pmap, tmap, pamap = {}, {}, {}
    for p, c, t, pa in zip(obs["perturbation"], obs["chembl-ID"],
                           obs["target"], obs["pathway"]):
        if p is None or pd.isna(p):
            continue
        p = str(p).strip()
        if pd.notna(c):
            pmap.setdefault(p, str(c).strip())
        if pd.notna(t):
            tmap.setdefault(p, str(t).strip())
        if pd.notna(pa):
            pamap.setdefault(p, str(pa).strip())
    chembls = sorted({v for v in pmap.values() if v and v.startswith("CHEMBL")})
    print(f"    perturbation={len(pmap)} 独特 chembl={len(chembls)}", flush=True)

    # 断点续传：已抓的存 TMP
    smiles_map = {}
    if os.path.isfile(TMP):
        try:
            smiles_map = json.load(open(TMP, encoding="utf-8"))
            print(f"    续传：已缓存 {len(smiles_map)} 个", flush=True)
        except Exception:
            smiles_map = {}
    todo = [c for c in chembls if c not in smiles_map]
    print(f">>> 并行查 ChEMBL（8 workers，剩 {len(todo)} 个）...", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for cid, sm in ex.map(_fetch_one, todo):
            if sm:
                smiles_map[cid] = sm
            done += 1
            if done % 40 == 0:
                json.dump(smiles_map, open(TMP, "w", encoding="utf-8"))
                print(f"    {done}/{len(todo)} ...", flush=True)
    json.dump(smiles_map, open(TMP, "w", encoding="utf-8"))
    print(f"    smiles 获取成功 {len(smiles_map)}/{len(chembls)}", flush=True)

    out = {}
    for p, cid in pmap.items():
        if cid in smiles_map:
            out[p] = {"chembl_id": cid, "smiles": smiles_map[cid],
                      "target": tmap.get(p, ""), "pathway": pamap.get(p, "")}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if os.path.isfile(TMP):
        os.remove(TMP)
    print(f"✅ sciplex3_chems.json 已存：{len(out)} 化合物有 SMILES "
          f"({os.path.getsize(OUT)//1024}KB)", flush=True)
    return len(out)


if __name__ == "__main__":
    main()
