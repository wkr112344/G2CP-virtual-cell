"""
从 UniProt 批量拉 LINCS 靶点基因的蛋白序列（P5-A：扩蛋白池）
=====================================================================
LINCS 靶点基因 ~890 个 → UniProt REST 分批查询（gene_exact + organism 9606）
→ 存 uniprot_seqs.json {基因符号: 蛋白序列}

用法：python unipret/fetch_uniprot_seqs.py
"""
import os
import sys
import csv
import json
import time
import urllib.request
import urllib.parse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINCS_TSV = os.path.join(BASE, "LINCS_small_molecules.tsv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uniprot_seqs.json")
API = "https://rest.uniprot.org/uniprotkb/search"


def get_all_targets():
    genes = set()
    with open(LINCS_TSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tg = (row.get("target") or "").strip()
            if tg and tg != "-":
                for g in tg.replace(",", " ").split():
                    g = g.strip().upper()
                    if g and g != "-":
                        genes.add(g)
    return sorted(genes)


def query_batch(genes):
    q = " OR ".join(f"gene_exact:{g}" for g in genes)
    q += " AND organism_id:9606"
    url = API + "?" + urllib.parse.urlencode({
        "query": q, "fields": "gene_names,sequence",
        "format": "tsv", "size": "500"})
    req = urllib.request.Request(url, headers={"User-Agent": "virtual-cell/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode()
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return ""


def main():
    genes = get_all_targets()
    print(f">>> LINCS 靶点基因 {len(genes)} 个，分批查 UniProt", flush=True)
    seqs = {}
    B = 50
    for i in range(0, len(genes), B):
        batch = genes[i:i + B]
        txt = query_batch(batch)
        lines = [l for l in txt.strip().split("\n") if l and not l.startswith("Gene Names")]
        for line in lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            gnames, seq = parts[0], parts[1]
            if not seq or "Sequence" in gnames:
                continue
            for g in gnames.split():
                g = g.strip().upper()
                if g in set(batch) and g not in seqs and len(seq) > 20:
                    seqs[g] = seq
        if (i // B + 1) % 5 == 0:
            print(f"    {i + len(batch)}/{len(genes)}，已命中 {len(seqs)}", flush=True)
        time.sleep(0.3)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(seqs, f, ensure_ascii=False)
    print(f"✅ uniprot_seqs.json：{len(seqs)}/{len(genes)} 基因拉到序列", flush=True)


if __name__ == "__main__":
    main()
