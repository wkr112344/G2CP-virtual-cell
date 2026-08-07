# -*- coding: utf-8 -*-
"""补丁 v9：结构蒸馏——药物嵌入相似度显式对齐 ECFP4 Tanimoto（保住结构/MoA 语义）。"""
src = open('train_g2cp_contrast.py', encoding='utf-8').read()

src = src.replace('''    ap.add_argument("--lam_cls", type=float, default=0.0)
    args = ap.parse_args()''',
'''    ap.add_argument("--lam_cls", type=float, default=0.0)
    ap.add_argument("--lam_struct", type=float, default=0.0)
    args = ap.parse_args()''')

src = src.replace('''        secs = time.time() - t0
        eta = secs * (args.epochs - ep - 1)''',
'''        if args.lam_struct > 0:
            for _ in range(10):
                pick_d = rng2.choice(n_drug, size=64, replace=False)
                pd = torch.from_numpy(pick_d).long().to(DEVICE)
                zd = F.normalize(net.cp_lin(fps_t[pd]), dim=1)
                cosd = zd @ zd.T
                fa = fps_t[pd]
                inter = fa @ fa.T
                union = fa.sum(1)[:, None] + fa.sum(1)[None, :] - inter
                tgt = inter / (union + 1e-8)
                ls = F.mse_loss(cosd, tgt)
                (args.lam_struct * ls).backward()
                opt.step(); opt.zero_grad()
        secs = time.time() - t0
        eta = secs * (args.epochs - ep - 1)''')

open('train_g2cp_contrast.py', 'w', encoding='utf-8').write(src)
import ast
ast.parse(src)
print('结构蒸馏补丁写入，语法 OK')
