# -*- coding: utf-8 -*-
"""一次性补丁：/cascade v3 —— 级联站用具体蛋白名 + 未知药物 SMILES 实时计算。"""
src = open('unipret/serve_api.py', encoding='utf-8').read()

# 1) 三处 signal 循环：用 step 的 n/m
old_sig = '''            steps = TRANSFER_LIB.get(pw, TRANSFER_LIB["general"])
            for i, s in enumerate(steps[:2]):
                chain.append(_mk(2 + i, "signal", "信号级联 · 第 " + str(i+1) + " 站",
                                 s, 0.9 - 0.1 * i, struct="通路级联（蛋白-蛋白传递）"))'''
new_sig = '''            steps = TRANSFER_LIB.get(pw, TRANSFER_LIB["general"])
            for i, s in enumerate(steps[:2]):
                chain.append(_mk(2 + i, "signal", s["n"], s["m"], 0.9 - 0.1 * i,
                                 struct="通路级联（蛋白-蛋白传递）"))'''
n1 = src.count(old_sig)
src = src.replace(old_sig, new_sig)

old_sig2 = '''            steps = TRANSFER_LIB.get(pw, TRANSFER_LIB["general"])
            for i, s in enumerate(steps[:2]):
                chain.append(_mk(2 + i, "signal", "信号级联 · 第 " + str(i+1) + " 站", s, 0.8 - 0.1 * i,
                                 struct="通路级联（蛋白-蛋白传递）"))'''
new_sig2 = '''            steps = TRANSFER_LIB.get(pw, TRANSFER_LIB["general"])
            for i, s in enumerate(steps[:2]):
                chain.append(_mk(2 + i, "signal", s["n"], s["m"], 0.8 - 0.1 * i,
                                 struct="通路级联（蛋白-蛋白传递）"))'''
n2 = src.count(old_sig2)
src = src.replace(old_sig2, new_sig2)
print(f"signal 循环替换: drug版 {n1} 处, gene/protein版 {n2} 处")

# 2) drug 分支：支持未知 SMILES
old_drug = '''        if ctype == "drug":
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
            pw = PATHWAY_MAP.get(info.get("pathway", ""), "general")
            if pw == "general":
                pw = pathway_of([t["g"] for t in tgts[:6]] + [ti.get("name", "")])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            up, _ = top_genes(pred)
            extra = [u["g"] for u in up if u["g"] in pool_set][:2]
            chain.append(_mk(0, "start", "药物 " + val,
                             "真实靶点：" + ti.get("name", "—") + "。" + ti.get("bind", ""), 1.0))
            chain.append(_mk(1, "target", ti.get("name", tgt or "未知靶点"),
                             "结合方式：" + ti.get("bind", "—") + "。" + ti.get("mech", ""),
                             1.0, struct=ti.get("struct", "—")))'''
new_drug = '''        if ctype == "drug":
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
            tgts = [{"g": pool_genes[i], "s": float(sim[i])}
                    for i in np.argsort(-sim)[:3]]
            pw = PATHWAY_MAP.get(info.get("pathway", ""), "general")
            if pw == "general":
                pw = pathway_of([t["g"] for t in tgts[:6]] + [ti.get("name", "")])
            pw_name = PW_NAME.get(pw, "多通路/未归类")
            up, _ = top_genes(pred)
            extra = [u["g"] for u in up if u["g"] in pool_set][:2]
            if known:
                chain.append(_mk(0, "start", "药物 " + val,
                                 "真实靶点：" + ti.get("name", "—") + "。" + ti.get("bind", ""), 1.0))
                chain.append(_mk(1, "target", ti.get("name", tgt or "未知靶点"),
                                 "结合方式：" + ti.get("bind", "—") + "。" + ti.get("mech", ""),
                                 1.0, struct=ti.get("struct", "—")))
            else:
                chain.append(_mk(0, "start", "未知药物（SMILES）",
                                 "无实验靶点注释，按统一空间实时检索预测靶点", 1.0))
                chain.append(_mk(1, "target", "预测靶点 " + tgts[0]["g"],
                                 "实时计算：该分子与 " + tgts[0]["g"] + " 在统一空间最接近（相似度 "
                                 + format(tgts[0]["s"], ".3f") + "），预测其结合该类蛋白（线索级，未实验验证）",
                                 1.0, struct="预测结合（未知药物无实验结构注释）"))'''
assert old_drug in src, "drug 分支未匹配"
src = src.replace(old_drug, new_drug, 1)

open('unipret/serve_api.py', 'w', encoding='utf-8').write(src)
import ast
ast.parse(src)
print("cascade v3 写入成功，语法 OK")
