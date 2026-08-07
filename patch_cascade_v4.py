# -*- coding: utf-8 -*-
"""一次性补丁：/cascade v4 —— 箭头方向全部由模型实时计算（预测受影响蛋白↑/↓ + 统一空间相似下游→）。"""
src = open('unipret/serve_api.py', encoding='utf-8').read()
start = src.index('    @app.route("/cascade", methods=["POST"])')
end = src.index('    @app.route("/structure/<name>")')
new_fn = '''    @app.route("/cascade", methods=["POST"])
    def cascade():
        """扰动跟踪链 v4：箭头方向全部由模型实时计算——
        起点 → 预测受影响蛋白(↑/↓ 带预测值) → 各蛋白的统一空间相似下游(→) → 表型。
        不再使用预制级联模板。"""
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
                    "pdb_id": pdb_id, "dir": None, "v": None, "next": None}

        chain = []
        pw = "general"
        pw_name = "多通路/未归类"

        def _add(kind, name, mech, dir=None, v=None, score=None, struct=None, has_struct=None, pdb_id=None):
            nd = _mk(len(chain), kind, name, mech, score, struct, has_struct, pdb_id)
            nd["dir"] = dir
            nd["v"] = v
            if chain:
                chain[-1]["next"] = name
            chain.append(nd)
            return nd

        def _sim_next(gname, used):
            """统一空间实时算：池中该蛋白的相似下游（排除自身与已用节点）"""
            if gname not in pool_set:
                return None
            gi = pool_genes.index(gname)
            for si in np.argsort(-SIM[gi]):
                cand = pool_genes[si]
                if cand != gname and cand not in used:
                    return cand, float(SIM[gi][si])
            return None

        pred = None
        if ctype == "drug":
            known = val in chem_map
            info = chem_map.get(val, {}) if known else {}
            sm = info.get("smiles", "") if known else val
            tgt = info.get("target", "")
            ti = _target_info(tgt)
            g = gcache.get(sm)
            if g is None:
                g = smiles_to_ecfp4(sm)
            if g is None or not g.any():
                return jsonify({"error": "药物/SMILES 无法解析（已知药请从列表选，未知药请输入 SMILES）"})
            with torch.no_grad():
                pred = effect.forward_compound(
                    [g], torch.tensor([cell], device=DEVICE)).cpu().numpy()[0]
                z = unipert_p.encode_compound(
                    [g], torch.zeros(1, dtype=torch.long, device=DEVICE)).cpu().numpy()[0]
            z = z / (np.linalg.norm(z) + 1e-8)
            sim = P @ z
            tgts = [pool_genes[i] for i in np.argsort(-sim)[:3]]
            pw = PATHWAY_MAP.get(info.get("pathway", ""), "general")
            if pw == "general":
                pw = pathway_of(tgts[:6] + [ti.get("name", "")])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            if known:
                _add("start", "药物 " + val, "真实靶点：" + ti.get("name", "—"))
                _add("target", ti.get("name", tgt or "未知靶点"),
                     "结合方式：" + ti.get("bind", "—") + "。" + ti.get("mech", ""),
                     struct=ti.get("struct", "—"))
            else:
                _add("start", "未知药物（SMILES）", "无实验靶点注释，实时检索预测靶点")
                _add("target", "预测靶点 " + tgts[0],
                     "实时计算：与 " + tgts[0] + " 统一空间最接近（相似度 " + format(float(sim[pool_genes.index(tgts[0])]), ".3f") + "），预测结合该类蛋白（线索级）",
                     struct="预测结合（无实验结构注释）")

        elif ctype == "gene":
            gnorm = val.upper()
            gid = gv_b.get(gnorm)
            if gid is None:
                return jsonify({"error": "基因 " + gnorm + " 不在词表"})
            with torch.no_grad():
                pred = effect.forward_gene(
                    torch.tensor([gid], device=DEVICE), seqs=None,
                    cell_line_idx=torch.tensor([cell], device=DEVICE)).cpu().numpy()[0]
            pw = pathway_of([gnorm])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            mech0 = gene_mech(gnorm)
            _add("start", gnorm + " 敲除（CRISPR）", mech0 or "基因功能丧失")

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
            _add("start", "输入蛋白/肽（" + str(len(seq)) + " aa）", "序列经 ESM 编码，实时检索最相似蛋白")
            _add("target", "最相似蛋白 " + hits[0],
                 "与 " + hits[0] + " 统一空间最接近（相似度 " + format(float(sim[pool_genes.index(hits[0])]), ".3f") + "），可能行使相似功能")

        # ---- 模型实时算链：预测受影响蛋白（↑/↓）→ 相似下游（→）----
        if pred is not None:
            used = {nd["name"] for nd in chain}
            up, dn = top_genes(pred, k=20)
            ups = [u for u in up if u["g"] in pool_set and u["g"] not in used][:2]
            dns = [u for u in dn if u["g"] in pool_set and u["g"] not in used][:1]
            for u in ups:
                hs, pdb = _struct_of(u["g"])
                _add("transcript", "↑ " + u["g"], "模型预测：表达上调（变化值 " + format(u["v"], "+.3f") + "）——扰动信号的第一个落点",
                     dir="up", v=u["v"], score=0.7, has_struct=hs, pdb_id=pdb)
            for u in dns:
                hs, pdb = _struct_of(u["g"])
                _add("transcript", "↓ " + u["g"], "模型预测：表达下调（变化值 " + format(u["v"], "+.3f") + "）",
                     dir="dn", v=u["v"], score=0.6, has_struct=hs, pdb_id=pdb)
            if ups:
                used = {nd["name"] for nd in chain}
                nxt = _sim_next(ups[0]["g"].lstrip("↑ "), used)
                if nxt:
                    hs, pdb = _struct_of(nxt[0])
                    _add("protein", "→ " + nxt[0],
                         "模型实时算的下一步：与 " + ups[0]["g"].lstrip("↑ ") + " 在统一空间最相似（相似度 " + format(nxt[1], ".3f") + "），扰动信号可能沿此传递",
                         dir="sim", v=nxt[1], score=0.5, has_struct=hs, pdb_id=pdb)

        _add("phenotype", pw_name,
             "由以上预测转录变化推断的表型：" + PW_PHENO.get(pw, "转录组整体重编程 → 细胞表型改变"))

        return jsonify({"type": ctype, "cell": CELL_NAMES[cell], "chain": chain,
                        "pathway": pw, "pathway_name": pw_name})

'''
src = src[:start] + new_fn + src[end:]
open('unipret/serve_api.py', 'w', encoding='utf-8').write(src)
import ast
ast.parse(src)
print("cascade v4 写入成功，语法 OK")
