# -*- coding: utf-8 -*-
"""train_g2cp_full.py —— 全基因版训练（12,327 基因输出）。

- 数据: data/g2cp_cache_fullgene（药物= gctx 全基因, 基因扰动= pool 978 映射到全基因列）
- 模型: G2CPNet 输出 12,327；head 最后一层 12327
- 损失: MSE + (1-PCC) + infoNCE
  - 药物样本: 全 12327 列参与损失
  - 基因扰动样本: 只有 978 列有真值（gene_cols.json）, 其余列 mask 掉
- 显存: 4GB 用梯度累积; head 输出大 → batch 调小
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
from unipret.compound_encoder import ECFP4_BITS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CorrectG2CPNet(G2CPNet):
    """修正语义: kind=0 基因 → gene_emb; kind=1 药物 → cp_lin(指纹)。
    原版 enc 语义反了(kind=0 走 cp_lin / kind=1 走 gene_emb), 导致药物索引被 clamp
    到基因词表, 3.2 万药大部分共享一个嵌入 → 药物特异性全丢。"""

    def enc(self, kind, key):
        k0 = kind.unsqueeze(1)
        z = self.gene_emb(torch.clamp(key, 0, self.gene_emb.num_embeddings - 1)) * (1 - k0) + \
            self.cp_lin(self._fps[torch.clamp(key, 0, self._fps.shape[0] - 1)]) * k0
        return F.normalize(z, dim=1)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bpert", type=int, default=32)
    ap.add_argument("--headw", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--freeze_cp", action="store_true", help="冻结 gene_emb/head/cell_emb, 只训 cp_lin(药物编码器) → 基因侧指标完全不变")
    ap.add_argument("--lam_nce", type=float, default=0.3)
    ap.add_argument("--moa_pairs", type=int, default=12, help="每批强制抽取的同类药物对数(×2 个扰动来自 MoA 类, 保证 nce 正样本)")
    ap.add_argument("--pcc_w", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=4, help="梯度累积步数(模拟大 batch)")
    ap.add_argument("--save", default="g2cp_full.pt")
    ap.add_argument("--from", dest="load_from", default="")
    ap.add_argument("--max_cell", type=int, default=10)
    ap.add_argument("--lam_esm", type=float, default=0.05, help="ESM 锚定损失权重(基因嵌入前320维拉向ESM语义)")
    ap.add_argument("--esm_cache", default=os.path.join(BASE, "data", "depmap", "esm_cache_smoothed.pt"),
                    help="ESM 蛋白语义缓存(默认蛋白相似图传播后的版本)")
    ap.add_argument("--frac_chem", type=float, default=1.0, help="化学训练比例(论文20%化学训练协议, 0.2=只用20%化合物)")
    ap.add_argument("--lam_aug", type=float, default=0.0, help="增强损失: 结构相似(指纹Tanimoto)药物对 → 嵌入收敛(路线2)")
    ap.add_argument("--lam_cpi", type=float, default=0.0, help="CPI 基因-化学对齐损失: 靶同一基因的药物↔基因 嵌入拉近")
    ap.add_argument("--cpi_pairs", default=os.path.join(CACHE, "cpi_pairs_big.npy"), help="CPI 训练对文件")
    ap.add_argument("--moa_cls", default=os.path.join(CACHE, "moa_cls.npy"), help="精细 MoA 分类文件(Touchstone, 对齐ρ考试直接监督)")
    args = ap.parse_args()

    m = np.load(os.path.join(CACHE, "meta.npz"), allow_pickle=True)
    kind, key, cell = m["kind"], m["key"], m["cell"]
    gene_vocab = [str(x) for x in m["gene_vocab"]]
    drug_vocab = [str(x) for x in m["drug_vocab"]]
    cl_names = [str(x) for x in m["cl_names"]]
    n_gene, n_drug = len(gene_vocab), len(drug_vocab)
    y = np.load(os.path.join(CACHE, "y.npy"), mmap_mode="r")
    fps = np.load(os.path.join(CACHE, "drug_fps.npy"))
    n_out = y.shape[1]
    n = len(kind)
    gene_cols = json.load(open(os.path.join(CACHE, "gene_cols.json")))
    log(f"数据: {n} 样本（基因 {n_gene} / 药 {n_drug} / 系 {len(cl_names)}）→ 输出 {n_out} 基因")

    pert_uid = np.where(kind == 0, key, n_gene + key)
    pert2idx = defaultdict(list)
    for i in range(n):
        pert2idx[int(pert_uid[i])].append(i)
    perts = sorted(pert2idx.keys())

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
    moa_cls_arr = np.load(args.moa_cls) if os.path.isfile(args.moa_cls) and len(np.load(args.moa_cls)) == len(drug_vocab) else None
    for i in range(n):
        if kind[i] == 1:
            pid = drug_vocab[int(key[i])] if int(key[i]) < len(drug_vocab) else ""
            cls_arr[i] = pcl2id.get(drug_pcl.get(pid, ""), -1)
            if moa_cls_arr is not None and int(key[i]) < len(moa_cls_arr) and moa_cls_arr[int(key[i])] >= 0:
                cls_arr[i] = int(moa_cls_arr[int(key[i])]) + 1000  # +1000 避免与 CMAP id 冲突
    cls_t = torch.from_numpy(cls_arr).long().to(DEVICE)
    if moa_cls_arr is not None:
        n_swap = int((cls_arr >= 1000).sum())
        log(f"精细 MoA 监督: {n_swap} 药物样本改用 Touchstone 类(对齐ρ直接优化)")
    cls2drugs = {}
    for i in range(n):
        if kind[i] == 1 and cls_arr[i] >= 0:
            cls2drugs.setdefault(int(cls_arr[i]), []).append(i)

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[perm[:int(n * 0.9)]] = True
    tr_pert = sorted({int(pert_uid[i]) for i in range(n) if tr_mask[i]})
    tr_pert_set = set(tr_pert)
    # MoA 成对采样支持: 每个 batch 强制抽取"同类药物对" → nce_loss 的 MoA 正样本真正生效
    # (否则 32,039 药/820 类随机采样, 同类药同批出现概率≈0.05对, MoA 监督形同虚设)
    cls2pert = {}
    for c, idxs in cls2drugs.items():
        ps = set()
        for i in idxs:
            pu = int(pert_uid[i])
            if pu in tr_pert_set:
                ps.add(pu)
        if len(ps) >= 2:
            cls2pert[c] = sorted(ps)
    n_moa_pair_cls = len(cls2pert)
    log(f"MoA 成对采样: {n_moa_pair_cls} 类可成对 (每批强制同类药物对)")
    if args.frac_chem < 1.0:
        chem_perts = [p for p in tr_pert if p >= n_gene]
        gene_perts = [p for p in tr_pert if p < n_gene]
        rng_c = np.random.RandomState(7)
        n_keep = int(len(chem_perts) * args.frac_chem)
        keep = set(int(chem_perts[i]) for i in rng_c.choice(len(chem_perts), n_keep, replace=False))
        tr_pert = gene_perts + [p for p in chem_perts if p in keep]
        log(f"化学训练比例 {args.frac_chem}: 药物扰动 {len(keep)}/{len(chem_perts)} + 基因扰动 {len(gene_perts)}")
    log(f"训练扰动 {len(tr_pert)} / 留出扰动 {len(perts)-len(tr_pert)}")

    # ---------- 细胞系扩展: 若预训练权重含更多系(GEO 162系), 按名字继承 ----------
    ext_cl = None
    if args.load_from:
        _ck0 = torch.load(args.load_from, map_location="cpu", weights_only=False)
        _cl0 = [str(x) for x in _ck0.get("cl_names", [])]
        if len(_cl0) > len(cl_names):
            ext_cl = _cl0
            log(f"细胞系扩展: 缓存 {len(cl_names)} → 预训练 {len(ext_cl)}(GEO 全保留)")
            # 缓存 cell 索引 → 新索引(按名字)
            cache2new = [ext_cl.index(c) for c in cl_names]
            cell = np.array([cache2new[c] for c in cell], dtype=np.int32)
            cl_names = ext_cl
        del _ck0

    net = CorrectG2CPNet(n_gene, ECFP4_BITS, 512, len(cl_names), n_out, args.headw).to(DEVICE)
    net.cls_out = len(pcl_ids)
    net._fps = torch.from_numpy(fps).to(DEVICE)
    fps_t = torch.from_numpy(fps).to(DEVICE)

    # ---------- ESM 蛋白语义接入(在 load_from 之后执行, 避免被权重继承覆盖) ----------
    if args.load_from:
        ck0 = torch.load(args.load_from, map_location="cpu", weights_only=False)
        sd0 = ck0["net"]
        # gene_emb 按名字继承(预训练 gene_vocab 可能含 IMM_* 扩展, 只取缓存基因对应行)
        gv0 = [str(x) for x in ck0.get("gene_vocab", [])]
        if "gene_emb.weight" in sd0 and len(gv0) >= n_gene and tuple(sd0["gene_emb.weight"].shape[:1]) == (len(gv0),):
            ge0 = sd0["gene_emb.weight"]
            if tuple(ge0.shape) != tuple(net.gene_emb.weight.shape):
                ge_new = net.gene_emb.weight.data.clone()
                for i, g in enumerate(gene_vocab):
                    if g in gv0:
                        ge_new[i] = ge0[gv0.index(g)]
                net.gene_emb.weight.data = ge_new
                log(f"gene_emb 按名字继承 {len(gene_vocab)} 行 (预训练 {len(gv0)})")
                sd0 = {k: v for k, v in sd0.items() if k != "gene_emb.weight"}
        # 只继承形状匹配的键(支持 DepMap 预训练权重跨结构加载)
        cur_sd = net.state_dict()
        sd0 = {k: v for k, v in sd0.items()
               if k in cur_sd and tuple(v.shape) == tuple(cur_sd[k].shape)}
        # 旧 cp_lin(单层线性) → 新 CPEncoder: main=旧权重, res 残差=0 → 初始输出精确=旧模型
        # (非线性残差从 0 开始学, 不破坏已学语义)
        if "cp_lin.main.weight" in cur_sd and "cp_lin.weight" in ck0["net"]:
            with torch.no_grad():
                net.cp_lin.main.weight.data.copy_(ck0["net"]["cp_lin.weight"].float())
                net.cp_lin.main.bias.data.copy_(ck0["net"]["cp_lin.bias"].float())
                net.cp_lin.res[2].weight.data.zero_()
                net.cp_lin.res[2].bias.data.zero_()
            sd0.pop("cp_lin.weight", None)
            sd0.pop("cp_lin.bias", None)
            log("cp_lin 单层→CPEncoder 继承: main=旧权重, res=0 → 初始输出精确=旧模型")
        # head 输出维不同: 只继承编码器 + head 前几层, 最后一层重训
        sd = {k: v for k, v in sd0.items() if not k.startswith("head")}
        net.load_state_dict(sd, strict=False)
        # 若 head.1 尺寸匹配则继承(含输出层 head.3)
        if "head.1.weight" in sd0 and tuple(sd0["head.1.weight"].shape) == tuple(net.head[1].weight.shape):
            for k in ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias",
                      "head.2.weight", "head.2.bias", "head.3.weight", "head.3.bias"]:
                if k in sd0:
                    sd[k] = sd0[k]
            net.load_state_dict(sd, strict=False)
            log("head 全层继承(含输出层 head.3)")
        log(f"加载预训练 {args.load_from}（形状匹配 {len(sd0)} 个键）")

    # ---------- ESM 蛋白语义接入(load_from 之后: 基因嵌入后192维继承预训练, 前320维用蛋白图传播后的ESM覆盖) ----------
    esm_path = args.esm_cache
    if os.path.isfile(esm_path) and args.lam_esm > 0:
        esm_cache = torch.load(esm_path, map_location="cpu", weights_only=False)
        esm_mat = torch.zeros((n_gene, 320), dtype=torch.float32)
        hit = 0
        for i, g in enumerate(gene_vocab):
            v = esm_cache.get(g)
            if v is not None:
                t = torch.from_numpy(v if hasattr(v, "numpy") else v)
                esm_mat[i] = t / (t.norm() + 1e-8)
                hit += 1
        with torch.no_grad():
            net.gene_emb.weight.data[:, :320] = esm_mat
        esm_t = esm_mat.to(DEVICE)
        log(f"ESM 蛋白图初始化: {hit}/{n_gene} 基因(前320维, {os.path.basename(esm_path)}), 锚定权重 {args.lam_esm}")
    else:
        esm_t = None
        log("无 ESM 缓存, 基因嵌入为纯 ID 查表")

    if args.freeze_cp:
        for name, p in net.named_parameters():
            if name.startswith("cp_lin"):
                p.requires_grad = True
            else:
                p.requires_grad = False
        log("冻结模式: 只训练 cp_lin(药物编码器), gene_emb/head/cell_emb 全部冻结 → 基因侧指标不变")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    kv_t = torch.from_numpy(key).long().to(DEVICE)
    kind_t = torch.from_numpy(kind).float().to(DEVICE)
    cell_t = torch.from_numpy(cell).long().to(DEVICE)

    # CPI 对齐训练对 (drug_key, gene_idx)
    cpi_pairs = None
    if args.lam_cpi > 0 and os.path.isfile(args.cpi_pairs):
        cpi_arr = np.load(args.cpi_pairs)
        cpi_pairs = [(int(a), int(b)) for a, b in cpi_arr]
        log(f"CPI 对齐: {len(cpi_pairs)} 对 (权重 {args.lam_cpi})")

    # 基因扰动样本的损失掩码: 978 列有效
    gene_mask = np.zeros(n_out, dtype=np.float32)
    gene_mask[gene_cols] = 1.0
    gene_mask_t = torch.from_numpy(gene_mask).to(DEVICE)
    log(f"参数 {sum(p.numel() for p in net.parameters())/1e6:.2f}M | 显存 {torch.cuda.memory_allocated()/1048576:.0f}MB")

    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        rng2 = np.random.RandomState(ep)
        total, tl = 0, 0.0
        n_batches = max(1, len(tr_pert) // args.bpert)
        opt.zero_grad()
        for bi in range(n_batches):
            # MoA 成对采样: 先强制抽同类药物对, 再补随机
            pick = []
            if args.moa_pairs > 0 and cls2pert:
                cls_pool = list(cls2pert.keys())
                n_cls = min(args.moa_pairs, len(cls_pool))
                for c in rng2.choice(cls_pool, size=n_cls, replace=False):
                    ps = cls2pert[c]
                    if len(ps) >= 2:
                        pick.extend(rng2.choice(ps, size=2, replace=False).tolist())
            pick = list(dict.fromkeys(pick))
            n_need = args.bpert - len(pick)
            if n_need > 0:
                rest = [p for p in tr_pert if p not in pick]
                if rest:
                    pick += rng2.choice(rest, size=min(n_need, len(rest)), replace=False).tolist()
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
            yb = torch.from_numpy(np.asarray(y[idxs])).to(DEVICE)
            # 损失掩码: 基因扰动样本只算有效列
            km = kb.unsqueeze(1)
            mask = torch.where(km > 0, torch.ones_like(out), gene_mask_t.unsqueeze(0))
            ym = yb * mask
            om = out * mask
            lm = F.mse_loss(om, ym)
            # PCC 只在有效列
            oc = om - om.mean(1, keepdim=True)
            yc = ym - ym.mean(1, keepdim=True)
            den = oc.norm(dim=1) * yc.norm(dim=1) + 1e-8
            pcc = (oc * yc).sum(1) / den
            lp = (1 - pcc).mean()
            # 对比损失只对药物样本算(基因样本不参与 → 基因侧零干扰, 保其他指标)
            dm = kb > 0
            if int(dm.sum()) >= 2:
                ln = nce_loss(z[dm], g[dm], tau=0.07, groups2=cls_t[b][dm])
            else:
                ln = torch.tensor(0.0, device=DEVICE)
            loss = (lm + args.pcc_w * lp + args.lam_nce * ln) / args.accum
            if esm_t is not None:
                loss = loss + args.lam_esm * F.mse_loss(net.gene_emb.weight[:n_gene, :320], esm_t) / args.accum
            # 增强损失(路线2): 结构相似药物对 → 嵌入收敛
            if args.lam_aug > 0:
                dmask = kb > 0
                if int(dmask.sum()) >= 2:
                    zb = z[dmask]  # 已 normalize
                    keyb = kv_t[b][dmask]
                    fpb = fps_t[keyb.clamp(0, fps_t.shape[0] - 1)]
                    fb = (fpb > 0).float()
                    a_cnt = fb.sum(1, keepdims=True)
                    inter = fb @ fb.T
                    union = a_cnt + a_cnt.T - inter
                    tani = inter / (union + 1e-8)
                    sim = zb @ zb.T
                    msel = (tani > 0.3) & ~torch.eye(zb.shape[0], dtype=torch.bool, device=DEVICE)
                    if int(msel.sum()) > 0:
                        la = (tani[msel] * (1 - sim[msel])).sum() / msel.sum()
                        loss = loss + args.lam_aug * la / args.accum
            # CPI 基因-化学对齐(考试3短板): 靶基因↔药物 嵌入拉近
            if args.lam_cpi > 0 and cpi_pairs is not None:
                n_cpi = min(128, len(cpi_pairs))
                ci_idx = rng2.choice(len(cpi_pairs), n_cpi, replace=False)
                dk = [cpi_pairs[i][0] for i in ci_idx]
                gi = [cpi_pairs[i][1] for i in ci_idx]
                zc = F.normalize(net.cp_lin(fps_t[torch.tensor(dk, device=DEVICE)]), dim=1)
                zg = F.normalize(net.gene_emb(torch.tensor(gi, device=DEVICE).long()), dim=1)
                pos = (zc * zg).sum(1)
                neg_gi = torch.randint(0, n_gene, (n_cpi,), device=DEVICE)
                zgn = F.normalize(net.gene_emb(neg_gi), dim=1)
                neg = (zc * zgn).sum(1)
                lc = (1 - pos).mean() + torch.clamp(neg + 0.1, min=0).mean()
                loss = loss + args.lam_cpi * lc / args.accum
            loss.backward()
            if (bi + 1) % args.accum == 0 or bi == n_batches - 1:
                opt.step()
                opt.zero_grad()
            total += len(b)
            tl += loss.item() * len(b) * args.accum
            if bi % 200 == 0:
                log(f"  batch {bi}/{n_batches} | loss {tl/max(total,1):.4f} | 显存 {torch.cuda.memory_allocated()/1048576:.0f}MB")
        sched.step()
        secs = time.time() - t0
        eta = secs * (args.epochs - ep - 1)
        log(f"epoch {ep+1}/{args.epochs} | loss {tl/total:.4f} | {secs:.0f}s | 剩余 {eta/60:.1f}min")
        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            torch.save({"net": net.state_dict(), "gene_vocab": gene_vocab, "drug_vocab": drug_vocab,
                        "cl_names": cl_names, "hvg": json.load(open(os.path.join(CACHE, "hvg.json"))),
                        "enc_semantics": "correct"}, args.save)
            log(f"  保存 {args.save}")
    log(f"训练完成 {args.save}")


if __name__ == "__main__":
    main()
