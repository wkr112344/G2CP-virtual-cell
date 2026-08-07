# -*- coding: utf-8 -*-
"""train_fusion.py —— V10(高PCC) × V5(高SMD) 融合训练。

底座 = V10：
- gene_emb（4,994 基因符号，与 fusion 缓存词表完全一致 → 直接继承）
- cp_lin（ECFP4→512，共享层，新药天然可用）
- head（1024 宽，5系高精度表型头）
- cell_emb 前 5 行（按细胞系名字映射继承）

扩展：
- cell_emb → 289 行（新 284 系从 V5 按名字拷贝，V5 已在 289 系上训过细胞系条件向量）
- drug_vocab → 39,343（新药走 cp_lin 共享层，无需新参数）

数据 = fusion 干净缓存（5系 pool 原始 + 284 新系 extra，不合并不平均）
loss = MSE + 0.5*(1-PCC) + 0.5*infoNCE（联合，V10 同款配方）
"""
import sys, os, json, time, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
FUSION = os.path.join(BASE, "data", "g2cp_cache_fusion")
from train_g2cp_contrast import G2CPNet, nce_loss
from unipret.compound_encoder import ECFP4_BITS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V10 = os.path.join(BASE, "g2cp_v10.pt")
V5 = os.path.join(BASE, "g2cp_v5.pt")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_fusion_net(ck10, ck5, n_gene, n_cell, n_out, headw):
    """V10(高PCC) × V5(高SMD) 融合 v4 —— "既要又要"正确初始化：
    - gene_emb ← V10（基因符号 4,994，干净词表）
    - cp_lin  ← V5（289系药物嵌入 → SMD 1.89+ 的功臣；ECFP4→512 与词表无关，天然兼容新药）
    - head    ← V10（1024 宽 5系高精度表型头 → PCC 0.463 的功臣）
    - cell_emb：5系 ← V10（干净，PCC 关键），284 新系 ← V5（289系覆盖）
    """
    net = G2CPNet(n_gene, ECFP4_BITS, 512, n_cell, n_out, headw).to(DEVICE)
    sd = {}
    # gene_emb + head 从 V10（PCC 功臣）
    for k in ["gene_emb.weight",
              "head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias",
              "head.2.weight", "head.2.bias", "head.3.weight", "head.3.bias"]:
        if k in ck10["net"]:
            sd[k] = ck10["net"][k]
    # cp_lin 从 V5（SMD 功臣）
    for k in ["cp_lin.weight", "cp_lin.bias"]:
        if k in ck5["net"]:
            sd[k] = ck5["net"][k]
    # cell_emb：5系从 V10（按名字），新系从 V5（按名字）
    ce = net.cell_emb.weight.data.clone()
    cl10 = [str(x) for x in ck10["cl_names"]]
    for i, c in enumerate(cl10):
        if c in FUSION_CL_IDX:
            ce[FUSION_CL_IDX[c]] = ck10["net"]["cell_emb.weight"][i]
    cl5 = [str(x) for x in ck5["cl_names"]]
    for i, c in enumerate(cl5):
        if c in FUSION_CL_IDX and c not in set(cl10):
            ce[FUSION_CL_IDX[c]] = ck5["net"]["cell_emb.weight"][i]
    sd["cell_emb.weight"] = ce
    net.load_state_dict(sd, strict=False)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bpert", type=int, default=64)
    ap.add_argument("--headw", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam_nce", type=float, default=0.5)
    ap.add_argument("--pcc_w", type=float, default=0.5)
    ap.add_argument("--save", default="g2cp_fusion.pt")
    ap.add_argument("--freeze_enc", action="store_true", help="冻结编码器只训 head")
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--max_cell", type=int, default=20)
    ap.add_argument("--cls_bal", type=float, default=0.6)
    ap.add_argument("--cls_per_batch", type=int, default=16)
    args = ap.parse_args()

    global FUSION_CL_IDX
    m = np.load(os.path.join(FUSION, "meta.npz"), allow_pickle=True)
    kind, key, cell = m["kind"], m["key"], m["cell"]
    gene_vocab = [str(x) for x in m["gene_vocab"]]
    drug_vocab = [str(x) for x in m["drug_vocab"]]
    cl_names = [str(x) for x in m["cl_names"]]
    FUSION_CL_IDX = {c: i for i, c in enumerate(cl_names)}
    n_gene, n_drug = len(gene_vocab), len(drug_vocab)
    yv = np.load(os.path.join(FUSION, "y.npy"))
    fps = np.load(os.path.join(FUSION, "drug_fps.npy"))
    n = len(kind)
    log(f"融合数据: {n} 样本（基因 {n_gene} / 药物 {n_drug} / 细胞系 {len(cl_names)}）→ {yv.shape[1]} HVG")

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
    n_pcl = len(pcl_ids)
    log(f"MoA 类别: {n_pcl} 类，覆盖药物样本 {int((cls_arr >= 0).sum())}")
    cls_t = torch.from_numpy(cls_arr).long().to(DEVICE)
    cls2drugs = {}
    for i in range(n):
        if kind[i] == 1 and cls_arr[i] >= 0:
            cls2drugs.setdefault(int(cls_arr[i]), []).append(i)
    log(f"MoA 类别（可平衡采样）: {len(cls2drugs)} 类")

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[perm[:int(n * 0.9)]] = True
    tr_pert = sorted({int(pert_uid[i]) for i in range(n) if tr_mask[i]})
    log(f"训练扰动 {len(tr_pert)} / 留出扰动 {len(perts)-len(tr_pert)}")

    # 构建融合网络
    ck10 = torch.load(V10, map_location="cpu", weights_only=False)
    ck5 = torch.load(V5, map_location="cpu", weights_only=False)
    net = build_fusion_net(ck10, ck5, n_gene, len(cl_names), yv.shape[1], args.headw)
    net.cls_out = n_pcl
    net._fps = torch.from_numpy(fps).to(DEVICE)
    log(f"融合网络就绪：编码器继承 V10，细胞系 289（5系=V10 + 284系=V5）")

    if args.freeze_enc:
        for nm, p in net.named_parameters():
            if not nm.startswith("head") and nm != "cell_emb.weight":
                p.requires_grad = False
        log("冻结 gene_emb/cp_lin（cell_emb 与 head 可训）")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    kv_t = torch.from_numpy(key).long().to(DEVICE)
    kind_t = torch.from_numpy(kind).float().to(DEVICE)
    cell_t = torch.from_numpy(cell).long().to(DEVICE)
    yv_t = torch.from_numpy(yv).to(DEVICE)
    log(f"参数 {sum(p.numel() for p in net.parameters())/1e6:.2f}M | 显存 {torch.cuda.memory_allocated()/1048576:.0f}MB")

    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        rng2 = np.random.RandomState(ep)
        total, tl, tn = 0, 0.0, 0.0
        n_batches = max(1, len(tr_pert) // args.bpert)
        n_cls_b = max(1, int(n_batches * args.cls_bal)) if (args.cls_bal > 0 and len(cls2drugs) > 1) else 0
        for bi in range(n_batches):
            idxs, groups = [], []
            if bi < n_cls_b:
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
                    groups.extend([gi] * k)
            b = torch.from_numpy(np.array(idxs)).long().to(DEVICE)
            g = torch.from_numpy(np.array(groups)).long().to(DEVICE)
            gc = cls_t[b]
            kb = kind_t[b]
            out, z, zc = net(kb, kv_t[b], cell_t[b])
            yb = yv_t[b]
            lm = F.mse_loss(out, yb)
            om = out - out.mean(1, keepdim=True)
            ym = yb - yb.mean(1, keepdim=True)
            pcc = (om * ym).sum(1) / (om.norm(dim=1) * ym.norm(dim=1) + 1e-8)
            lp = (1 - pcc).mean()
            ln = nce_loss(z, g, tau=args.tau, groups2=gc)
            loss = lm + args.pcc_w * lp + args.lam_nce * ln
            opt.zero_grad(); loss.backward(); opt.step()
            total += len(b); tl += loss.item() * len(b); tn += ln.item() * len(b) if torch.is_tensor(ln) else 0
        sched.step()
        secs = time.time() - t0
        eta = secs * (args.epochs - ep - 1)
        log(f"epoch {ep+1}/{args.epochs} | loss {tl/total:.4f} | nce {tn/total:.4f} | {secs:.0f}s | 剩余 {eta/60:.1f}min")
        if (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            torch.save({"net": net.state_dict(), "gene_vocab": gene_vocab, "drug_vocab": drug_vocab,
                        "cl_names": cl_names, "hvg": json.load(open(os.path.join(FUSION, "hvg.json")))}, args.save)
            log(f"  保存 {args.save}")
    log(f"训练完成 {args.save}")


if __name__ == "__main__":
    main()
