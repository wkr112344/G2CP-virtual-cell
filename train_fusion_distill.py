# -*- coding: utf-8 -*-
"""train_fusion_distill.py —— V10(5系高精度) 蒸馏进融合模型(289系)。

目标：学生模型在 5系上输出逼近 V10(教师, PCC 0.463)，同时保持 289 系全覆盖 + SMD 超论文。

关键设计：
- 教师 = g2cp_v10.pt（5系专属 head，PCC 0.463）
- 学生 = g2cp_fusion_v3.pt 继续训练（289系，SMD 1.908）
- 教师输出离线预计算（5系样本的 978 维软标签），训练时直接查表 → 省显存
- 蒸馏损失只作用于 5系样本：MSE(学生输出, 教师软标签)
- 冻结学生编码器(cp_lin/gene_emb) → SMD 不动；只训 head + cell_emb
"""
import sys, os, json, time, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
FUSION = os.path.join(BASE, "data", "g2cp_cache_fusion")
CACHE5 = os.path.join(BASE, "data", "g2cp_cache_5cell")
from train_g2cp_contrast import G2CPNet, nce_loss
from unipret.compound_encoder import ECFP4_BITS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIVE = ["HT29", "A375", "A549", "MCF7", "PC3"]
TEACHER = os.path.join(BASE, "g2cp_v10.pt")
STUDENT = os.path.join(BASE, "g2cp_fusion_v3.pt")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_net(ckpt, cache_dir):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    gene_vocab = [str(x) for x in ck["gene_vocab"]]
    drug_vocab = [str(x) for x in ck["drug_vocab"]]
    cl_names = [str(x) for x in ck["cl_names"]]
    emb = ck["net"]["head.0.weight"].shape[0] - 32
    headw = ck["net"]["head.1.weight"].shape[0]
    net = G2CPNet(len(gene_vocab), ECFP4_BITS, emb, len(cl_names), len(ck["hvg"]), headw).to(DEVICE)
    net.load_state_dict(ck["net"], strict=False)
    net.eval()
    fps = np.load(os.path.join(cache_dir, "drug_fps.npy"))
    net._fps = torch.from_numpy(fps).to(DEVICE)
    return net, gene_vocab, drug_vocab, cl_names, ck["hvg"]


def teacher_embed(net, kind, key, fp):
    """教师嵌入（与训练统一公式一致）"""
    k = torch.tensor([key], device=DEVICE).long()
    k0 = torch.tensor([kind], device=DEVICE).float().unsqueeze(1)
    z = net.gene_emb(torch.clamp(k, 0, net.gene_emb.num_embeddings - 1)) * k0 + \
        net.cp_lin(torch.from_numpy(fp).float().unsqueeze(0).to(DEVICE)) * (1 - k0)
    return F.normalize(z, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bpert", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam_distill", type=float, default=0.5, help="蒸馏损失权重")
    ap.add_argument("--pcc_w", type=float, default=1.0)
    ap.add_argument("--lam_nce", type=float, default=0.1)
    ap.add_argument("--save", default="g2cp_fusion_distill.pt")
    ap.add_argument("--max_cell", type=int, default=20)
    args = ap.parse_args()

    # ---------- 学生数据 ----------
    m = np.load(os.path.join(FUSION, "meta.npz"), allow_pickle=True)
    kind, key, cell = m["kind"], m["key"], m["cell"]
    gene_vocab = [str(x) for x in m["gene_vocab"]]
    drug_vocab = [str(x) for x in m["drug_vocab"]]
    cl_names = [str(x) for x in m["cl_names"]]
    n_gene, n_drug = len(gene_vocab), len(drug_vocab)
    yv = np.load(os.path.join(FUSION, "y.npy"))
    fps = np.load(os.path.join(FUSION, "drug_fps.npy"))
    n = len(kind)
    log(f"学生数据: {n} 样本（基因 {n_gene} / 药物 {n_drug} / 系 {len(cl_names)}）")

    # 扰动唯一标签
    pert_uid = np.where(kind == 0, key, n_gene + key)
    pert2idx = defaultdict(list)
    for i in range(n):
        pert2idx[int(pert_uid[i])].append(i)
    perts = sorted(pert2idx.keys())

    # MoA 类别
    pcl_path = os.path.join(BASE, "data", "g2cp", "data", "CMAP_mmc1.txt")
    drug_pcl = {}
    if os.path.isfile(pcl_path):
        with open(pcl_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
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
    cls_t = torch.from_numpy(cls_arr).long().to(DEVICE)
    cls2drugs = {}
    for i in range(n):
        if kind[i] == 1 and cls_arr[i] >= 0:
            cls2drugs.setdefault(int(cls_arr[i]), []).append(i)

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[perm[:int(n * 0.9)]] = True
    tr_pert = sorted({int(pert_uid[i]) for i in range(n) if tr_mask[i]})
    log(f"训练扰动 {len(tr_pert)} / 留出扰动 {len(perts)-len(tr_pert)}")

    # ---------- 教师 ----------
    t_net, t_gv, t_dv, t_cl, t_hvg = load_net(TEACHER, CACHE5)
    t_gv_set, t_dv_set = set(t_gv), set(t_dv)
    t_gmap = {g: i for i, g in enumerate(t_gv)}
    t_dmap = {d: i for i, d in enumerate(t_dv)}
    t_cl_idx = {c: i for i, c in enumerate(t_cl)}
    log(f"教师就绪: 基因 {len(t_gv)} / 药 {len(t_dv)} / 系 {t_cl}")

    # 5系在融合缓存中的样本索引
    five_idx = [cl_names.index(c) for c in FIVE]
    is5 = np.isin(cell, five_idx)
    # 5系样本中教师词表可映射的
    distillable = np.zeros(n, dtype=bool)
    for i in range(n):
        if not is5[i]:
            continue
        if kind[i] == 0:
            g = gene_vocab[int(key[i])]
            distillable[i] = g in t_gv_set
        else:
            d = drug_vocab[int(key[i])]
            distillable[i] = d in t_dv_set
    log(f"5系样本 {int(is5.sum())} / 可蒸馏 {int(distillable.sum())}")

    # ---------- 离线预计算教师软标签 ----------
    soft_path = os.path.join(FUSION, "teacher_soft.npy")
    if os.path.isfile(soft_path):
        soft = np.load(soft_path)
        log(f"加载教师软标签缓存 {soft.shape}")
    else:
        log("预计算教师软标签（5系可蒸馏样本 978 维输出）...")
        di = np.where(distillable)[0]
        soft = np.zeros((n, yv.shape[1]), dtype=np.float32)
        t_fps = np.load(os.path.join(CACHE5, "drug_fps.npy"))
        t_fps_t = torch.from_numpy(t_fps).to(DEVICE)
        with torch.no_grad():
            for s in range(0, len(di), 256):
                bi = di[s:s + 256]
                outs = []
                for i in bi:
                    kd = int(kind[i])
                    if kd == 0:
                        g = gene_vocab[int(key[i])]
                        kt = t_gmap[g]
                        z = t_net.gene_emb(torch.tensor([kt], device=DEVICE).long())
                        z = F.normalize(z, dim=1)
                    else:
                        d = drug_vocab[int(key[i])]
                        kt = t_dmap[d]
                        z = t_net.cp_lin(t_fps_t[[kt]])
                        z = F.normalize(z, dim=1)
                    cname = cl_names[int(cell[i])]
                    ct = t_cl_idx[cname]
                    out = t_net.head(torch.cat([z, t_net.cell_emb(torch.tensor([ct], device=DEVICE).long())], dim=1))
                    outs.append(out[0])
                soft[bi] = torch.stack(outs).cpu().numpy()
                if (s // 256) % 10 == 0:
                    log(f"  教师软标签 {s+len(bi)}/{len(di)}")
        np.save(soft_path, soft)
        log(f"教师软标签已存 {soft_path}")
    soft_t = torch.from_numpy(soft).to(DEVICE)

    # ---------- 学生（加载融合 v3） ----------
    ck = torch.load(STUDENT, map_location="cpu", weights_only=False)
    emb = ck["net"]["head.0.weight"].shape[0] - 32
    headw = ck["net"]["head.1.weight"].shape[0]
    net = G2CPNet(n_gene, ECFP4_BITS, emb, len(cl_names), yv.shape[1], headw).to(DEVICE)
    net.load_state_dict(ck["net"], strict=False)
    net.cls_out = len(pcl_ids)
    net._fps = torch.from_numpy(fps).to(DEVICE)
    log(f"学生加载 {os.path.basename(STUDENT)}: head {headw}")

    # 冻结编码器 → 保 SMD；只训 head + cell_emb
    for nm, p in net.named_parameters():
        if not nm.startswith("head") and nm != "cell_emb.weight":
            p.requires_grad = False
    log("冻结 gene_emb/cp_lin（SMD 不动），只训 head + cell_emb")

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    kv_t = torch.from_numpy(key).long().to(DEVICE)
    kind_t = torch.from_numpy(kind).float().to(DEVICE)
    cell_t = torch.from_numpy(cell).long().to(DEVICE)
    yv_t = torch.from_numpy(yv).to(DEVICE)
    distillable_t = torch.from_numpy(distillable).to(DEVICE)

    log(f"可训参数 {sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.2f}M | 显存 {torch.cuda.memory_allocated()/1048576:.0f}MB")

    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        rng2 = np.random.RandomState(ep)
        total, tl, td = 0, 0.0, 0.0
        n_batches = max(1, len(tr_pert) // args.bpert)
        for bi in range(n_batches):
            pick = rng2.choice(tr_pert, size=min(args.bpert, len(tr_pert)), replace=False)
            idxs, groups = [], []
            for gi, p in enumerate(pick):
                ci = pert2idx[int(p)]
                k = min(len(ci), args.max_cell)
                sel = rng2.choice(ci, size=k, replace=False)
                idxs.extend(sel)
                groups.extend([gi] * k)
            b = torch.from_numpy(np.array(idxs)).long().to(DEVICE)
            g = torch.from_numpy(np.array(groups)).long().to(DEVICE)
            kb = kind_t[b]
            out, z, _ = net(kb, kv_t[b], cell_t[b])
            yb = yv_t[b]
            lm = F.mse_loss(out, yb)
            om = out - out.mean(1, keepdim=True)
            ym = yb - yb.mean(1, keepdim=True)
            pcc = (om * ym).sum(1) / (om.norm(dim=1) * ym.norm(dim=1) + 1e-8)
            lp = (1 - pcc).mean()
            ln = nce_loss(z, g, tau=0.15, groups2=cls_t[b])
            loss = lm + args.pcc_w * lp + args.lam_nce * ln
            # 蒸馏损失：只对可蒸馏(5系)样本，向教师软标签对齐
            dm = distillable_t[b]
            if dm.any():
                ld = F.mse_loss(out[dm], soft_t[b][dm])
                loss = loss + args.lam_distill * ld
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += len(b)
            tl += loss.item() * len(b)
        sched.step()
        secs = time.time() - t0
        eta = secs * (args.epochs - ep - 1)
        log(f"epoch {ep+1}/{args.epochs} | loss {tl/total:.4f} | {secs:.0f}s | 剩余 {eta/60:.1f}min")
        if (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            torch.save({"net": net.state_dict(), "gene_vocab": gene_vocab, "drug_vocab": drug_vocab,
                        "cl_names": cl_names, "hvg": json.load(open(os.path.join(FUSION, "hvg.json")))}, args.save)
            log(f"  保存 {args.save}")
    log(f"蒸馏训练完成 {args.save}")


if __name__ == "__main__":
    main()
