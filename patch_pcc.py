# -*- coding: utf-8 -*-
"""PCC 提升补丁：
1. --pcc_w：PCC 感知损失权重参数化（默认 0.5 → 可加大到 1.5）
2. --headw 加宽：表型头 1024→2048（容量↑ 拟合↑）
3. load_from 只继承编码器（gene_emb/cp_lin/cell_emb），head 尺寸变化不冲突
"""
import ast

p = "train_g2cp_contrast.py"
s = open(p, encoding="utf-8").read()

old_arg = '''    ap.add_argument("--headw", type=int, default=1024)'''
new_arg = '''    ap.add_argument("--headw", type=int, default=1024)
    ap.add_argument("--pcc_w", type=float, default=0.5, help="PCC 感知损失权重")'''
assert old_arg in s
s = s.replace(old_arg, new_arg, 1)

old_load = '''    if args.load_from:
        ck0 = torch.load(args.load_from, map_location="cpu", weights_only=False)
        net.load_state_dict(ck0["net"])
        log(f"已加载预训练 {args.load_from}")'''
new_load = '''    if args.load_from:
        ck0 = torch.load(args.load_from, map_location="cpu", weights_only=False)
        # 只继承编码器（head 尺寸可能不同，重新学）
        sd = {k: v for k, v in ck0["net"].items() if not k.startswith("head")}
        net.load_state_dict(sd, strict=False)
        log(f"已加载预训练 {args.load_from}（编码器部分）")'''
assert old_load in s
s = s.replace(old_load, new_load, 1)

# 两处 0.5 * lp → args.pcc_w * lp
old_l1 = '''                loss = lm + 0.5 * lp + 0.2 * ln'''
new_l1 = '''                loss = lm + args.pcc_w * lp + 0.2 * ln'''
assert old_l1 in s
s = s.replace(old_l1, new_l1, 1)
old_l2 = '''                loss = lm + 0.5 * lp + args.lam_nce * ln'''
new_l2 = '''                loss = lm + args.pcc_w * lp + args.lam_nce * ln'''
assert old_l2 in s
s = s.replace(old_l2, new_l2, 1)

open(p, "w", encoding="utf-8").write(s)
ast.parse(s)
print("PCC 提升补丁完成，语法 OK")
