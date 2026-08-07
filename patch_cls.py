# -*- coding: utf-8 -*-
"""补丁 v8：药物 MoA 类别分类头（CE 损失强监督类别分离 → 提升 SMD）。"""
src = open('train_g2cp_contrast.py', encoding='utf-8').read()

# 1) G2CPNet 加分类头
src = src.replace('''        self.head = nn.Sequential(nn.LayerNorm(emb + 32), nn.Linear(emb + 32, head_w), nn.GELU(),
                                  nn.Linear(head_w, n_out))''',
'''        self.head = nn.Sequential(nn.LayerNorm(emb + 32), nn.Linear(emb + 32, head_w), nn.GELU(),
                                  nn.Linear(head_w, n_out))
        self.cls_head = nn.Linear(emb, 256)''')

src = src.replace('''        c = self.cell_emb(cell)
        return self.head(torch.cat([z, c], dim=1)), z''',
'''        c = self.cell_emb(cell)
        return self.head(torch.cat([z, c], dim=1)), z, self.cls_head(z)''')

# 2) 参数：分类权重
src = src.replace('''    ap.add_argument("--max_cell", type=int, default=20)
    args = ap.parse_args()''',
'''    ap.add_argument("--max_cell", type=int, default=20)
    ap.add_argument("--lam_cls", type=float, default=0.0)
    args = ap.parse_args()''')

# 3) n_pcl 传入模型（分类头输出尺寸）
src = src.replace('''    log(f"药物类别（MoA）: {len(pcl_ids)} 类，覆盖药物样本 {int((cls_arr >= 0).sum())}")
    cls_t = torch.from_numpy(cls_arr).long().to(DEVICE)''',
'''    n_pcl = len(pcl_ids)
    log(f"药物类别（MoA）: {n_pcl} 类，覆盖药物样本 {int((cls_arr >= 0).sum())}")
    cls_t = torch.from_numpy(cls_arr).long().to(DEVICE)
    net.cls_out = n_pcl''')

src = src.replace('''    net = G2CPNet(n_gene, ECFP4_BITS, args.emb, len(cl_names), yv.shape[1], args.headw).to(DEVICE)
    net._fps = torch.from_numpy(fps).to(DEVICE)''',
'''    net = G2CPNet(n_gene, ECFP4_BITS, args.emb, len(cl_names), yv.shape[1], args.headw).to(DEVICE)
    net.cls_out = 256
    net._fps = torch.from_numpy(fps).to(DEVICE)''')

# 4) 前向解包 + CE 损失
src = src.replace('''            out, z = net(kb, kv_t[b], cell_t[b])''',
'''            out, z, zc = net(kb, kv_t[b], cell_t[b])''')

src = src.replace('''            ln = nce_loss(z, g, tau=args.tau, groups2=gc)
            if args.phase == "contrast":
                loss = args.lam_nce * ln
            elif args.phase == "finetune":
                loss = lm + 0.5 * lp + 0.2 * ln
            else:
                loss = lm + 0.5 * lp + args.lam_nce * ln''',
'''            ln = nce_loss(z, g, tau=args.tau, groups2=gc)
            if args.phase == "contrast":
                loss = args.lam_nce * ln
            elif args.phase == "finetune":
                loss = lm + 0.5 * lp + 0.2 * ln
            else:
                loss = lm + 0.5 * lp + args.lam_nce * ln
            if args.lam_cls > 0:
                cm = gc >= 0
                if cm.any():
                    cl = nn.functional.cross_entropy(net.cls_head(zc[cm])[:, :net.cls_out], gc[cm])
                    loss = loss + args.lam_cls * cl''')

open('train_g2cp_contrast.py', 'w', encoding='utf-8').write(src)
import ast
ast.parse(src)
print('v8 分类头补丁写入，语法 OK')
