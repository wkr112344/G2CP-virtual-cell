"""
生成 HVG 基因符号表（供实时 API 使用）
=====================================
把 sciPlex3 HVG 2000 的 ENSG → mygene → 基因符号，存 data/gui/hvg_gene_names.json
"""
import os
import sys
import json
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
SCIPLEX3 = "C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad"
OUT = os.path.join(BASE, "data", "gui", "hvg_gene_names.json")


def ensg_to_symbol(ensg_list, batch=200):
    import urllib.request
    out = {}
    for i in range(0, len(ensg_list), batch):
        chunk = [e for e in ensg_list[i:i + batch] if e and e.startswith("ENSG")]
        if not chunk:
            continue
        payload = json.dumps({"q": chunk, "scopes": "ensembl.gene",
                              "fields": "symbol", "species": "human"}).encode()
        try:
            req = urllib.request.Request(
                "https://mygene.info/v3/query", data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "virtual-cell/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                hits = json.loads(r.read().decode())
            if isinstance(hits, dict):
                hits = hits.get("hits", [])
            for h in hits:
                if h.get("query") and h.get("symbol"):
                    out[h["query"]] = str(h["symbol"])
        except Exception:
            pass
        time.sleep(0.15)
    return out


def main():
    from unipret.data_bridge import PerturbationReader, select_hvg
    r = PerturbationReader(SCIPLEX3, backed=True)
    hvg = select_hvg(r, n=2000, max_cells=50000)
    ensg_real = [str(r.ad.var["ensembl_id"].iloc[i]) for i in hvg]
    r.close()
    print(f">>> 映射 {len(hvg)} 个 HVG ...", flush=True)
    sym_map = ensg_to_symbol([e for e in ensg_real if e.startswith("ENSG")])
    gene_names = [sym_map.get(e, "Gene_" + str(i)) for i, e in enumerate(ensg_real)]
    ok = sum(1 for g in gene_names if not g.startswith("Gene_"))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(gene_names, f)
    print(f"✅ hvg_gene_names.json：{ok}/{len(gene_names)} 有效基因名", flush=True)


if __name__ == "__main__":
    main()
