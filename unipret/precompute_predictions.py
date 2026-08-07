"""
预计算 GUI 数据（P6）：136 药 × 3 系表达预测 + CPI 靶点检索
=====================================================================
1. stageB 对 408 个 (药×系) 样本预测表达变化（2000 HVG）
2. HVG 的 ENSG → mygene → 基因符号（真名展示）
3. 每样本 top20 上调/下调基因（真名 + 预测值）
4. unipert_pretrain（905 蛋白池）做 CPI 靶点检索 top5
产物：data/gui/drugs_predictions.json
"""
import os
import sys
import json
import time
import urllib.request
import urllib.parse

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
SCIPLEX3 = "C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad"
CHEM_MAP = os.path.join(BASE, "unipret", "sciplex3_chems.json")
OUT = os.path.join(BASE, "data", "gui", "drugs_predictions.json")
CELL_NAMES = ["K562", "A549", "MCF7"]


def ensg_to_symbol(ensg_list, batch=200):
    """mygene POST 批量 ENSG -> symbol。返回 {ensg: symbol}。"""
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
                hits = json.loads(r.read().decode())   # POST 返回 list
            if isinstance(hits, dict):
                hits = hits.get("hits", [])
            for h in hits:
                if h.get("query") and h.get("symbol"):
                    out[h["query"]] = str(h["symbol"])
        except Exception as e:
            print(f"    mygene 批 {i} 失败: {e}", flush=True)
        time.sleep(0.15)
    return out


def main():
    t0 = time.time()
    import torch
    from unipret.config import DEVICE
    from unipret.data_bridge import PerturbationReader, select_hvg, build_compound_samples_chembl
    from unipret.train_stages import _make_unipert
    from unipret.compound_encoder import smiles_to_ecfp4
    from unipret.effect_model import PerturbationEffectModel
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(">>> 预计算 GUI 数据", flush=True)

    # 预测模型 stageB
    sb = torch.load(os.path.join(BASE, "stageB.pt"), map_location=DEVICE)
    gv_b = sb["gene_vocab"]
    hvg_n = sb.get("hvg_dim", 2000)
    reader = PerturbationReader(SCIPLEX3, backed=True)
    hvg = select_hvg(reader, n=hvg_n, max_cells=50000)
    ensg_list = [str(reader.var_names[i]) for i in hvg]
    # 真实 ENSG（var.ensembl_id）
    ensg_col = []
    for i in hvg:
        e = reader.obs  # placeholder
        break
    ensg_real = [str(reader.ad.var["ensembl_id"].iloc[i]) for i in hvg]
    print(f"    映射 {len(hvg)} 个 HVG 的 ENSG → 基因符号 ...", flush=True)
    sym_map = ensg_to_symbol([e for e in ensg_real if e.startswith("ENSG")])
    gene_names = [sym_map.get(e, "ENSG" + e[4:8]) for e in ensg_real]
    ok = sum(1 for g in gene_names if g.startswith("ENSG") is False)
    print(f"    映射成功 {ok}/{len(gene_names)}", flush=True)

    chem_map = json.load(open(CHEM_MAP, encoding="utf-8"))
    samples = build_compound_samples_chembl(reader, chem_map, hvg)
    for s in samples:
        s["smiles"] = chem_map[s["name"]]["smiles"]
    samples = [s for s in samples if s.get("smiles")]
    reader.close()
    print(f"    样本 {len(samples)}", flush=True)

    unipert = _make_unipert(len(gv_b) + 1, gene_vocab=gv_b, gene_mode="esm").to(DEVICE)
    unipert.load_state_dict(sb["unipert"])
    ea = PerturbationEffectModel.build_stage_a(unipert, hvg_n).to(DEVICE)
    effect = PerturbationEffectModel.build_stage_b(ea).to(DEVICE)   # 含 head_comp
    effect.load_state_dict(sb["effect"])
    effect.eval()

    # CPI 检索模型（unipert_pretrain，905 池）
    up = torch.load(os.path.join(BASE, "unipert_pretrain.pt"), map_location=DEVICE)
    gv_p = {str(k).strip().upper(): v for k, v in up["gene_vocab"].items()}
    unipert_p = _make_unipert(len(up["gene_vocab"]), gene_vocab=up["gene_vocab"],
                              gene_mode="esm").to(DEVICE)
    unipert_p.load_state_dict(up["unipert"])
    unipert_p.eval()
    pool_genes = sorted(gv_p.keys())
    pool_idx = [gv_p[g] for g in pool_genes]
    with torch.no_grad():
        P = unipert_p.encode_gene(None, torch.tensor(pool_idx, device=DEVICE),
                                  torch.zeros(len(pool_idx), dtype=torch.long, device=DEVICE))
        P = P.cpu().numpy()
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    print(f"    CPI 蛋白池 {len(pool_genes)}", flush=True)

    cache = {}
    from unipret.data_bridge import CompoundGraphCache
    gcache = CompoundGraphCache()
    all_smiles = sorted({s["smiles"] for s in samples})
    with ThreadPoolExecutor(max_workers=8) as ex:
        for sm, g in zip(all_smiles, ex.map(smiles_to_ecfp4, all_smiles)):
            if g is not None and g.any():
                gcache.cache[sm] = g

    result = {}
    with torch.no_grad():
        for s in samples:
            nm = s["name"]
            cl = s["cell_line_idx"]
            g = gcache.get(s["smiles"])
            if g is None:
                continue
            pred = effect.forward_compound(
                [g], torch.tensor([cl], device=DEVICE)).cpu().numpy()[0]
            order = np.argsort(-pred)
            top_up = [{"g": gene_names[i], "v": float(pred[i])}
                      for i in order[:15] if not gene_names[i].startswith("ENSG")]
            top_dn = [{"g": gene_names[i], "v": float(pred[i])}
                      for i in order[-15:] if not gene_names[i].startswith("ENSG")]
            # CPI 靶点 top5
            z = unipert_p.encode_compound([g], torch.zeros(1, dtype=torch.long, device=DEVICE))
            z = z.cpu().numpy()[0]
            z = z / (np.linalg.norm(z) + 1e-8)
            sim = P @ z
            top_t = [{"g": pool_genes[i], "s": float(sim[i])}
                     for i in np.argsort(-sim)[:5]]
            cell = CELL_NAMES[cl] if cl < 3 else "K562"
            result.setdefault(nm, {}).setdefault(cell, {})
            result[nm][cell] = {"up": top_up, "down": top_dn,
                                "targets": top_t,
                                "pathway": chem_map.get(nm, {}).get("pathway", ""),
                                "target_anno": chem_map.get(nm, {}).get("target", "")}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"✅ drugs_predictions.json：{len(result)} 药 × 3 系（{time.time()-t0:.0f}s，"
          f"{os.path.getsize(OUT)//1024}KB）", flush=True)


if __name__ == "__main__":
    main()
