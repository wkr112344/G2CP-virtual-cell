# -*- coding: utf-8 -*-
"""一次性补丁：重写 serve_api.py 的 /cascade 接口为 v2。"""
src = open('unipret/serve_api.py', encoding='utf-8').read()
start = src.index('    @app.route("/cascade", methods=["POST"])')
end = src.index('    @app.route("/structure/<name>")')
new_fn = '''    @app.route("/cascade", methods=["POST"])
    def cascade():
        """扰动跟踪链（v2）：起点(真实靶点/最近蛋白) → 信号级联 → 转录变化 → 具体表型。
        每一级：名字 + 结合/变构机制(具体文字) + 结构可用性。"""
        import re as _re
        d = request.get_json(force=True)
        ctype = d.get("type", "drug")
        cell = int(d.get("cell", 0))
        val = (d.get("value") or "").strip()
        pool_set = set(pool_genes)

        def _struct_of(name):
            for tok in _re.findall(r"[A-Z][A-Z0-9]{2,}", str(name)):
                if tok in PROTEIN_PDB:
                    return True, PROTEIN_PDB[tok]
            return False, None

        def _target_info(t):
            if not t:
                return TARGET_LIB.get("Others", {})
            if t in TARGET_LIB:
                return TARGET_LIB[t]
            tl = t.lower()
            for k, v in TARGET_LIB.items():
                kw = k.split()[0].lower()
                if kw in tl or tl in k.lower():
                    return v
            return TARGET_LIB.get("Others", {})

        def _mk(stage, kind, name, mech, score=None, struct=None, has_struct=None, pdb_id=None):
            return {"stage": stage, "kind": kind, "name": name, "mech": mech,
                    "score": score, "struct": struct, "has_struct": has_struct,
                    "pdb_id": pdb_id, "next": None}

        chain = []
        pw = "general"
        pw_name = "多通路/未归类"

        if ctype == "drug":
            info = chem_map.get(val, {})
            sm = info.get("smiles", "")
            tgt = info.get("target", "")
            ti = _target_info(tgt)
            g = gcache.get(sm)
            if g is None:
                g = smiles_to_ecfp4(sm)
            if g is None or not g.any():
                return jsonify({"error": "药物构图失败（已知药请从列表选，未知药请用 SMILES）"})
            with torch.no_grad():
                pred = effect.forward_compound(
                    [g], torch.tensor([cell], device=DEVICE)).cpu().numpy()[0]
                z = unipert_p.encode_compound(
                    [g], torch.zeros(1, dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
            z = z / (np.linalg.norm(z) + 1e-8)
            sim = P @ z
            tgts = [{"g": pool_genes[i], "s": float(sim[i])}
                    for i in np.argsort(-sim)[:3]]
            pw = pathway_of([t["g"] for t in tgts[:6]] + [ti.get("name", "")])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            up, _ = top_genes(pred)
            extra = [u["g"] for u in up if u["g"] in pool_set][:2]
            chain.append(_mk(0, "start", "药物 " + val,
                             "真实靶点：" + ti.get("name", "—") + "。" + ti.get("bind", ""), 1.0))
            chain.append(_mk(1, "target", ti.get("name", tgt or "未知靶点"),
                             "结合方式：" + ti.get("bind", "—") + "。" + ti.get("mech", ""),
                             1.0, struct=ti.get("struct", "—")))
            steps = TRANSFER_LIB.get(pw, TRANSFER_LIB["general"])
            for i, s in enumerate(steps[:2]):
                chain.append(_mk(2 + i, "signal", "信号级联 · 第 " + str(i+1) + " 站",
                                 s, 0.9 - 0.1 * i, struct="通路级联（蛋白-蛋白传递）"))
            for i, gname in enumerate(extra[:1]):
                hs, pdb = _struct_of(gname)
                chain.append(_mk(3 + len(steps[:2]), "transcript", gname + "（模型预测上调）",
                                 "模型预测该基因表达上调（统一空间与药效信号相关）", 0.6,
                                 struct="转录水平变化", has_struct=hs, pdb_id=pdb))
            chain.append(_mk(99, "phenotype", pw_name, PW_PHENO.get(pw, "转录组整体重编程 → 细胞表型改变")))

        elif ctype == "gene":
            gnorm = val.upper()
            gid = gv_b.get(gnorm)
            if gid is None:
                return jsonify({"error": "基因 " + gnorm + " 不在词表"})
            with torch.no_grad():
                pred = effect.forward_gene(
                    torch.tensor([gid], device=DEVICE), seqs=None,
                    cell_line_idx=torch.tensor([cell], device=DEVICE)).cpu().numpy()[0]
                zg = unipert_p.encode_gene(
                    None, torch.tensor([gid], device=DEVICE),
                    torch.zeros(1, dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
            zg = zg / (np.linalg.norm(zg) + 1e-8)
            sim = P @ zg
            near = [pool_genes[i] for i in np.argsort(-sim)[:2]]
            pw = pathway_of([gnorm])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            up, _ = top_genes(pred)
            extra = [u["g"] for u in up if u["g"] in pool_set][:2]
            mech0 = gene_mech(gnorm)
            chain.append(_mk(0, "start", gnorm + " 敲除（CRISPR）", mech0 or "基因功能丧失", 1.0))
            hs, pdb = _struct_of(gnorm)
            chain.append(_mk(1, "target", "最近蛋白 " + (near[0] if near else "—"),
                             "敲除基因的蛋白产物在统一空间中与 " + (near[0] if near else "—") + " 最接近，扰动信号从基因传递到该蛋白",
                             0.9, has_struct=hs, pdb_id=pdb))
            steps = TRANSFER_LIB.get(pw, TRANSFER_LIB["general"])
            for i, s in enumerate(steps[:2]):
                chain.append(_mk(2 + i, "signal", "信号级联 · 第 " + str(i+1) + " 站", s, 0.8 - 0.1 * i,
                                 struct="通路级联（蛋白-蛋白传递）"))
            for i, gname in enumerate(extra[:1]):
                hs, pdb = _struct_of(gname)
                chain.append(_mk(4, "transcript", gname + "（模型预测上调）",
                                 "模型预测该基因表达上调（敲除的次级效应）", 0.5,
                                 struct="转录水平变化", has_struct=hs, pdb_id=pdb))
            chain.append(_mk(99, "phenotype", pw_name, PW_PHENO.get(pw, "转录组整体重编程 → 细胞表型改变")))

        else:  # protein
            seq1, mods = _peptide_to_1letter(val)
            seq = seq1 if seq1 else _re.sub(r"[^A-Za-z]", "", val).strip().upper()
            if len(seq) < 20:
                return jsonify({"error": "序列太短（清洗后 " + str(len(seq)) + " aa，至少 20）"})
            with torch.no_grad():
                z = encode_unknown_protein(seq).cpu().numpy()[0]
            z = z / (np.linalg.norm(z) + 1e-8)
            sim = P @ z
            hits = [pool_genes[i] for i in np.argsort(-sim)[:3]]
            pw = pathway_of(hits[:6])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            with torch.no_grad():
                zt = torch.as_tensor(z, device=DEVICE).unsqueeze(0)
                pred = effect.head_gene(
                    zt + effect.unipert.cell_line(torch.tensor([cell], device=DEVICE))
                ).cpu().numpy()[0]
            up, _ = top_genes(pred)
            extra = [u["g"] for u in up if u["g"] in pool_set][:2]
            chain.append(_mk(0, "start", "输入蛋白/肽（" + str(len(seq)) + " aa）",
                             "序列经 ESM 编码，在 905 蛋白池中检索最相似蛋白", 1.0))
            hs, pdb = _struct_of(hits[0] if hits else "")
            chain.append(_mk(1, "target", "最相似蛋白 " + (hits[0] if hits else "—"),
                             "该蛋白在统一空间中与 " + (hits[0] if hits else "—") + " 最接近，可能行使相似功能或参与同一通路",
                             0.9, has_struct=hs, pdb_id=pdb))
            steps = TRANSFER_LIB.get(pw, TRANSFER_LIB["general"])
            for i, s in enumerate(steps[:2]):
                chain.append(_mk(2 + i, "signal", "信号级联 · 第 " + str(i+1) + " 站", s, 0.8 - 0.1 * i,
                                 struct="通路级联（蛋白-蛋白传递）"))
            for i, gname in enumerate(extra[:1]):
                hs, pdb = _struct_of(gname)
                chain.append(_mk(4, "transcript", gname + "（模型预测上调）",
                                 "模型预测该基因表达上调（该蛋白入细胞后的转录组响应）", 0.5,
                                 struct="转录水平变化", has_struct=hs, pdb_id=pdb))
            chain.append(_mk(99, "phenotype", pw_name, PW_PHENO.get(pw, "转录组整体重编程 → 细胞表型改变")))

        for i, nd in enumerate(chain):
            nd["next"] = chain[i+1]["name"] if i+1 < len(chain) else None
        return jsonify({"type": ctype, "cell": CELL_NAMES[cell], "chain": chain,
                        "pathway": pw, "pathway_name": pw_name})

'''
src = src[:start] + new_fn + src[end:]
open('unipret/serve_api.py', 'w', encoding='utf-8').write(src)
import ast
ast.parse(src)
print('cascade v2 写入成功，语法 OK')
