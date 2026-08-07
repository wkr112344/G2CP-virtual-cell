# -*- coding: utf-8 -*-
"""finetune_immune.py —— 用 GEO 免疫细胞数据(GSE22886+GSE60235)微调全基因模型。

- 基础: g2cp_full_asc.pt (143 系 / 4,994 基因扰动 / 12,328 输出)
- 扩展: cell_emb 143→152 (9 个免疫细胞系), gene_emb 4994→5004 (10 个 IMM_* 虚拟扰动)
- 免疫样本: 18 个代表样本 (每 epoch 上采样 20 遍), 全列监督 (无 nce, 每组1样本)
- 主样本: 每 epoch 少量混合 (防灾难性遗忘), 含 nce
- mask: 免疫样本/药物 全列; 真实基因扰动 仅 978 列 (gene_cols.json)
"""
import sys, os, json, time, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "data", "g2cp_cache_fullgene")
from train_g2cp_contrast import G2CPNet, nce_loss


class CorrectG2CPNet(G2CPNet):
    """正确语义: kind=0 基因 → gene_emb; kind=1 药物 → cp_lin(指纹)。"""

    def enc(self, kind, key):
        k0 = kind.unsqueeze(1)
        z = self.gene_emb(torch.clamp(key, 0, self.gene_emb.num_embeddings - 1)) * (1 - k0) +             self.cp_lin(self._fps[torch.clamp(key, 0, self._fps.shape[0] - 1)]) * k0
        return F.normalize(z, dim=1)
from unipret.compound_encoder import ECFP4_BITS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--headw", type=int, default=1024)
    ap.add_argument("--imm_repeat", type=int, default=24, help="免疫样本每 epoch 重复次数")
    ap.add_argument("--main_batch", type=int, default=16, help="每 epoch 主样本扰动数(防遗忘)")
    ap.add_argument("--imm_w", type=float, default=3.0, help="免疫损失权重")
    ap.add_argument("--main_w", type=float, default=0.3, help="主样本损失权重")
    ap.add_argument("--from", dest="load_from", default="g2cp_full_asc.pt")
    ap.add_argument("--save", default="g2cp_full_imm.pt")
    args = ap.parse_args()

    # ---------- 主缓存 ----------
    m = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
    kind, key, cell = m["kind"], m["key"], m["cell"]
    gene_vocab = [str(x) for x in m["gene_vocab"]]
    drug_vocab = [str(x) for x in m["drug_vocab"]]
    cl_names = [str(x) for x in m["cl_names"]]
    y = np.load(os.path.join(CACHE, "y.npy"), mmap_mode="r")
    fps = np.load(os.path.join(CACHE, "drug_fps.npy"))
    n_out = y.shape[1]
    n = len(kind)
    gene_cols = json.load(open(os.path.join(CACHE, "gene_cols.json")))
    n_real_gene = len(gene_vocab)
    n_cl_old = len(cl_names)
    log(f"主缓存: {n} 样本, 基因扰动 {n_real_gene}, 药 {len(drug_vocab)}, 系 {n_cl_old}, 输出 {n_out}")

    # ---------- GEO 补充数据(免疫 + 多类型细胞) ----------
    imm = json.load(open(os.path.join(BASE, "data", "geo_immune", "immune_samples.json")))
    cells = json.load(open(os.path.join(BASE, "data", "geo_cells", "cells_samples.json")))
    imm = imm + cells
    imm_cells = sorted(set(s["cell"] for s in imm))
    imm_perts = sorted(set(s["pert"] for s in imm))
    new_cl = cl_names + imm_cells
    new_gv = gene_vocab + imm_perts
    log(f"免疫: {len(imm)} 样本 | 新细胞系 {imm_cells} | 新扰动 {imm_perts}")
    log(f"扩展后: 系 {len(new_cl)} ({n_cl_old}+{len(imm_cells)}), 基因词表 {len(new_gv)} ({n_real_gene}+{len(imm_perts)})")

    imm_key = np.array([n_real_gene + imm_perts.index(s["pert"]) for s in imm], dtype=np.int64)
    imm_cl = np.array([n_cl_old + imm_cells.index(s["cell"]) for s in imm], dtype=np.int64)
    imm_y = np.array([s["expr"] for s in imm], dtype=np.float32)
    n_imm = len(imm)
    log(f"免疫 y 形状 {imm_y.shape} | 均值 {imm_y.mean():.4f} 方差 {imm_y.var():.4f}")

    # ---------- 模型 ----------
    ck0 = torch.load(args.load_from, map_location="cpu", weights_only=False)
    net = CorrectG2CPNet(len(new_gv), ECFP4_BITS, 512, len(new_cl), n_out, args.headw).to(DEVICE)
    net.cls_out = 0
    net._fps = torch.from_numpy(fps).to(DEVICE)
    sd0 = ck0["net"]
    # 按名字映射继承 cell_emb / gene_emb (新行保持随机初始化)
    old_cl_names = [str(x) for x in ck0.get("cl_names", [])]
    old_cl_idx = {c: i for i, c in enumerate(old_cl_names)}
    old_ce = sd0.get("cell_emb.weight")
    if old_ce is not None:
        ce = net.cell_emb.weight.data.clone()
        for i, c in enumerate(new_cl):
            j = old_cl_idx.get(c)
            if j is not None:
                ce[i] = old_ce[j]
        net.cell_emb.weight.data = ce
        log(f"cell_emb 按名字继承 {sum(1 for c in new_cl if c in old_cl_idx)}/{len(new_cl)} 行")
    old_gv = [str(x) for x in ck0.get("gene_vocab", [])]
    old_gv_idx = {g: i for i, g in enumerate(old_gv)}
    old_ge = sd0.get("gene_emb.weight")
    if old_ge is not None:
        ge = net.gene_emb.weight.data.clone()
        for i, g in enumerate(new_gv):
            j = old_gv_idx.get(g)
            if j is not None:
                ge[i] = old_ge[j]
        net.gene_emb.weight.data = ge
        log(f"gene_emb 按名字继承 {sum(1 for g in new_gv if g in old_gv_idx)}/{len(new_gv)} 行")
    rest = {k: v for k, v in sd0.items() if k not in ("cell_emb.weight", "gene_emb.weight")}
    net.load_state_dict(rest, strict=False)
    net.train()
    log(f"参数 {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

    # ---------- 主样本扰动索引(训练用) ----------
    pert_uid = np.where(kind == 0, key, n_real_gene + key)
    pert2idx = defaultdict(list)
    for i in range(n):
        pert2idx[int(pert_uid[i])].append(i)
    tr_perts = sorted(pert2idx.keys())
    log(f"主扰动 {len(tr_perts)}")

    # ---------- 优化器 ----------
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    gene_mask = np.zeros(n_out, dtype=np.float32)
    gene_mask[gene_cols] = 1.0
    gene_mask_t = torch.from_numpy(gene_mask).to(DEVICE)
    kv_t = torch.from_numpy(key).long().to(DEVICE)
    kind_t = torch.from_numpy(kind).float().to(DEVICE)
    cell_t = torch.from_numpy(cell).long().to(DEVICE)
    rng = np.random.RandomState(42)

    for ep in range(args.epochs):
        t0 = time.time()
        # === 1) 免疫样本上采样 ===
        imm_loss_sum, imm_n = 0.0, 0
        for rep in range(args.imm_repeat):
            perm = rng.permutation(n_imm)
            for s0 in range(0, n_imm, 8):
                b = perm[s0:s0 + 8]
                if len(b) < 2:
                    continue
                k = torch.from_numpy(imm_key[b]).long().to(DEVICE)
                c = torch.from_numpy(imm_cl[b]).long().to(DEVICE)
                kk = torch.zeros(len(b), device=DEVICE)  # kind=0 (走 gene_emb)
                out, z, _ = net(kk, k, c)
                yb = torch.from_numpy(imm_y[b]).to(DEVICE)
                lm = F.mse_loss(out, yb)
                oc = out - out.mean(1, keepdim=True)
                yc = yb - yb.mean(1, keepdim=True)
                den = oc.norm(dim=1) * yc.norm(dim=1) + 1e-8
                lp = (1 - (oc * yc).sum(1) / den).mean()
                loss = (lm + lp) * args.imm_w
                loss.backward()
                opt.step()
                opt.zero_grad()
                imm_loss_sum += loss.item() * len(b)
                imm_n += len(b)
        # === 2) 主样本保持 ===
        main_loss_sum, main_n = 0.0, 0
        pick = rng.choice(tr_perts, size=min(args.main_batch, len(tr_perts)), replace=False)
        idxs, groups = [], []
        for gi, p in enumerate(pick):
            ci = pert2idx[int(p)]
            kk = min(len(ci), 10)
            sel = rng.choice(ci, size=kk, replace=False)
            idxs.extend(sel)
            groups.extend([gi] * kk)
        b = torch.from_numpy(np.array(idxs)).long().to(DEVICE)
        g = torch.from_numpy(np.array(groups)).long().to(DEVICE)
        kb = kind_t[b]
        out, z, _ = net(kb, kv_t[b], cell_t[b])
        yb = torch.from_numpy(np.asarray(y[idxs])).to(DEVICE)
        keyb = kv_t[b]
        km = kb.unsqueeze(1)
        # mask: 真实基因扰动(key<n_real_gene) 978 列; 其余全列
        mask = torch.where((km > 0) | (keyb.unsqueeze(1) >= n_real_gene),
                           torch.ones_like(out), gene_mask_t.unsqueeze(0))
        om, ym = out * mask, yb * mask
        lm = F.mse_loss(om, ym)
        oc = om - om.mean(1, keepdim=True)
        yc = ym - ym.mean(1, keepdim=True)
        den = oc.norm(dim=1) * yc.norm(dim=1) + 1e-8
        lp = (1 - (oc * yc).sum(1) / den).mean()
        ln = nce_loss(z, g, tau=0.15)
        loss = (lm + lp + 0.3 * ln) * args.main_w
        loss.backward()
        opt.step()
        opt.zero_grad()
        main_loss_sum += loss.item() * len(b)
        main_n += len(b)
        secs = time.time() - t0
        log(f"ep {ep+1}/{args.epochs} | 免疫 loss {imm_loss_sum/max(imm_n,1):.4f} | 主 loss {main_loss_sum/max(main_n,1):.4f} | {secs:.0f}s")
        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            torch.save({"net": net.state_dict(), "gene_vocab": new_gv, "drug_vocab": drug_vocab,
                        "cl_names": new_cl, "hvg": json.load(open(os.path.join(CACHE, "hvg.json"))), "enc_semantics": "correct"}, args.save)
            log(f"  保存 {args.save}")
    log(f"完成 {args.save}")


if __name__ == "__main__":
    main()
