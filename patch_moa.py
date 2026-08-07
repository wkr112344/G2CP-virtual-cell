# -*- coding: utf-8 -*-
"""补丁：MoA 类别对比监督（CMAP PCL 同类药互为对比正对）。"""
src = open('train_g2cp_contrast.py', encoding='utf-8').read()

src = src.replace('''def nce_loss(z, groups, tau=0.15):
    """NT-Xent：同组（同扰动）互为正样本。"""
    sim = (z @ z.T) / tau
    N = z.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    mask = (groups[:, None] == groups[None, :]) & ~eye''',
'''def nce_loss(z, groups, tau=0.15, groups2=None):
    """NT-Xent：同扰动 或 同 MoA 类别 互为正样本。"""
    sim = (z @ z.T) / tau
    N = z.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    mask = (groups[:, None] == groups[None, :]) & ~eye
    if groups2 is not None:
        same_cls = (groups2[:, None] == groups2[None, :]) & ~eye & (groups2[:, None] >= 0)
        mask = mask | same_cls''')

src = src.replace('''    log(f"唯一扰动: {len(perts)}")''',
'''    # 药物 MoA 类别（CMAP_mmc1 PCL）：同类药互为对比正对
    pcl_path = os.path.join(BASE, "data", "g2cp", "data", "CMAP_mmc1.txt")
    drug_pcl = {}
    if os.path.isfile(pcl_path):
        with open(pcl_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\\n").split("\\t")
                if len(parts) < 6:
                    continue
                for d in parts[4].split("|"):
                    if d.startswith("BRD-"):
                        drug_pcl.setdefault(d, parts[0])
    pcl_ids = sorted(set(drug_pcl.values()))
    pcl2id = {p: i for i, p in enumerate(pcl_ids)}
    cls_arr = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        if kind[i] == 1:
            pid = drug_vocab[int(key[i])] if int(key[i]) < len(drug_vocab) else ""
            cls_arr[i] = pcl2id.get(drug_pcl.get(pid, ""), -1)
    log(f"药物类别（MoA）: {len(pcl_ids)} 类，覆盖药物样本 {int((cls_arr >= 0).sum())}")
    cls_t = torch.from_numpy(cls_arr).long().to(DEVICE)''')

src = src.replace('''            b = torch.from_numpy(np.array(idxs)).long().to(DEVICE)
            g = torch.from_numpy(np.array(groups)).long().to(DEVICE)
            kb = kind_t[b]''',
'''            b = torch.from_numpy(np.array(idxs)).long().to(DEVICE)
            g = torch.from_numpy(np.array(groups)).long().to(DEVICE)
            gc = cls_t[b]
            kb = kind_t[b]''')

src = src.replace('ln = nce_loss(z, g, tau=args.tau)', 'ln = nce_loss(z, g, tau=args.tau, groups2=gc)')

open('train_g2cp_contrast.py', 'w', encoding='utf-8').write(src)
import ast
ast.parse(src)
print('MoA 类别对比写入，语法 OK')
