"""
扩展 ESM 缓存（P5-A：蛋白池从 20 → 910 基因）
=====================================================================
把 uniprot_seqs.json（890 个 LINCS 靶点基因序列）+ dataset.json（本地 20）
全部 ESM2-8M 编码，合并进 esm_cache.pt（{基因: 320维}）。

用法：python uniprot/extend_esm_cache.py
"""
import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
SEQ_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uniprot_seqs.json")
LOCAL_JSON = os.path.join(BASE, "dataset.json")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "esm_cache.pt")


def main():
    import torch
    from unipret.esm_encoder import load_esm, esm_mean_embedding, ESM_CACHE_PATH

    seqs = {}
    with open(SEQ_JSON, encoding="utf-8") as f:
        seqs.update({k.upper(): v for k, v in json.load(f).items()})
    with open(LOCAL_JSON, encoding="utf-8") as f:
        d = json.load(f)
        for k, v in (d.get("proteins") or {}).items():
            if isinstance(v, dict) and v.get("seq"):
                seqs.setdefault(k.upper(), v["seq"])
    print(f">>> 待编码序列：{len(seqs)} 条", flush=True)

    model, alphabet = load_esm()
    names = sorted(seqs.keys())
    embs = esm_mean_embedding(model, alphabet, [seqs[n] for n in names],
                              device="cuda" if torch.cuda.is_available() else "cpu")
    cache = {n: embs[i].numpy().astype("float32") for i, n in enumerate(names)}
    torch.save(cache, CACHE)
    print(f"✅ esm_cache.pt 已更新：{len(cache)} 基因（{os.path.getsize(CACHE)//2**20}MB）", flush=True)


if __name__ == "__main__":
    main()
