# -*- coding: utf-8 -*-
"""train_g2cp_contrast.py 两阶段升级补丁：
1. --cls_bal：类别平衡采样（MoA 同类药同 batch 密集共现 → InfoNCE 同类正对爆炸 → SMD↑）
2. --cls_per_batch：每 batch 采样的 MoA 类别数
3. --freeze_enc：冻结 gene_emb/cp_lin/cell_emb，只训表型 head（Phase B 微调，PCC↑ 且 SMD 不回退）
"""
import ast

p = "train_g2cp_contrast.py"
s = open(p, encoding="utf-8").read()

# 1) 参数
old_arg = '''    ap.add_argument("--lam_struct", type=float, default=0.0)'''
new_arg = '''    ap.add_argument("--lam_struct", type=float, default=0.0)
    ap.add_argument("--cls_bal", type=float, default=0.6, help="类别平衡采样比例（0=关闭）")
    ap.add_argument("--cls_per_batch", type=int, default=16)
    ap.add_argument("--freeze_enc", action="store_true", help="冻结编码器只训 head")'''
assert old_arg in s
s = s.replace(old_arg, new_arg, 1)

# 2) freeze_enc：net 定义后、opt 创建前
old_opt = '''    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)'''
new_opt = '''    if args.freeze_enc:
        for nm, p in net.named_parameters():
            if not nm.startswith("head"):
                p.requires_grad = False
        log("已冻结编码器（gene_emb/cp_lin/cell_emb），只训表型 head")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)'''
assert old_opt in s
s = s.replace(old_opt, new_opt, 1)

# 3) cls2drugs 构建（cls_t 定义后）
old_cls = '''    cls_t = torch.from_numpy(cls_arr).long().to(DEVICE)'''
new_cls = '''    cls_t = torch.from_numpy(cls_arr).long().to(DEVICE)
    # 类别 → 药物样本索引（用于类别平衡采样）
    cls2drugs = {}
    for i in range(n):
        if kind[i] == 1 and cls_arr[i] >= 0:
            cls2drugs.setdefault(int(cls_arr[i]), []).append(i)
    log(f"MoA 类别（可平衡采样）: {len(cls2drugs)} 类")'''
assert old_cls in s
s = s.replace(old_cls, new_cls, 1)

# 4) 采样循环改造：前 cls_bal 比例的 batch 走类别平衡
old_loop = '''        for bi in range(n_batches):
            pick = rng2.choice(tr_pert, size=min(args.bpert, len(tr_pert)), replace=False)
            idxs, groups = [], []
            for gi, p in enumerate(pick):
                ci = pert2idx[int(p)]
                k = min(len(ci), args.max_cell)
                sel = rng2.choice(ci, size=k, replace=False)
                idxs.extend(sel)
                groups.extend([gi] * k)'''
new_loop = '''        n_cls_b = max(1, int(n_batches * args.cls_bal)) if (args.cls_bal > 0 and len(cls2drugs) > 1 and args.phase != "finetune") else 0
        for bi in range(n_batches):
            idxs, groups = [], []
            if bi < n_cls_b:
                # 类别平衡：每 batch 从 ncls 个 MoA 类各取若干药，同类药密集共现
                ncls = min(args.cls_per_batch, len(cls2drugs))
                cls_keys = list(cls2drugs.keys())
                cls_pick = rng2.choice(cls_keys, size=ncls, replace=False)
                per_cls = max(2, args.bpert // ncls)
                gid = 0
                for cid in cls_pick:
                    pool = np.asarray(cls2drugs[cid], dtype=np.int64)
                    if len(pool) == 0:
                        continue
                    _, first = np.unique(key[pool], return_index=True)
                    rng2.shuffle(first)
                    for fi in first[:per_cls]:
                        si = int(pool[fi])
                        ci = pert2idx.get(int(pert_uid[si]), [si])
                        k = min(len(ci), args.max_cell)
                        sel = rng2.choice(ci, size=k, replace=False)
                        idxs.extend(sel); groups.extend([gid] * k); gid += 1
            else:
                pick = rng2.choice(tr_pert, size=min(args.bpert, len(tr_pert)), replace=False)
                for gi, p in enumerate(pick):
                    ci = pert2idx[int(p)]
                    k = min(len(ci), args.max_cell)
                    sel = rng2.choice(ci, size=k, replace=False)
                    idxs.extend(sel)
                    groups.extend([gi] * k)'''
assert old_loop in s
s = s.replace(old_loop, new_loop, 1)

open(p, "w", encoding="utf-8").write(s)
ast.parse(s)
print("两阶段补丁完成，语法 OK")
