# -*- coding: utf-8 -*-
"""G2CP 论文级重训 v4：infoNCE 对比预训练 + 表型（MSE+PCC）联合训练。
- 对比：同一扰动跨细胞系的嵌入应一致（正样本对），不同扰动互斥 → 统一空间对齐（UniPert 核心）
- 表型：预测 978 HVG 表达（MSE + 1-PCC 感知）
带实时进度侦测：每 epoch 写 train_progress.json + 无缓冲日志。"""
import sys, os, json, time, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, "data", "g2cp_cache")
from unipret.compound_encoder import ECFP4_BITS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

class CPEncoder(nn.Module):
    """药物编码器: 主线性投影(继承旧模型) + 残差MLP(初始0, 逐步学非线性)。
    输出 = main(fps) + res(fps) → 初始精确=旧模型, 表达能力升级为非线性(论文 dense embedding 投影的加强版)。
    所有旧代码 net.cp_lin(fps) 调用点零改动(CPEncoder 可调用)。"""

    def __init__(self, fp_dim, emb, hid=1024):
        super().__init__()
        self.main = nn.Linear(fp_dim, emb)
        self.res = nn.Sequential(
            nn.Linear(fp_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, emb),
        )
        nn.init.kaiming_normal_(self.main.weight)
        nn.init.kaiming_normal_(self.res[0].weight)
        nn.init.zeros_(self.res[2].weight)   # 残差输出=0 → 初始输出=主路径(精确继承旧模型)
        nn.init.zeros_(self.res[2].bias)

    def forward(self, x):
        return self.main(x) + self.res(x)


class G2CPNet(nn.Module):
    def __init__(self, n_gene, fp_dim, emb=512, n_cell=5, n_out=978, head_w=1024):
        super().__init__()
        self.gene_emb = nn.Embedding(n_gene, emb)
        self.cp_lin = CPEncoder(fp_dim, emb, 1024)
        self.cell_emb = nn.Embedding(n_cell, 32)
        self.head = nn.Sequential(nn.LayerNorm(emb + 32), nn.Linear(emb + 32, head_w), nn.GELU(),
                                  nn.Linear(head_w, n_out))
        self.cls_head = nn.Linear(emb, 256)
    def enc(self, kind, key):
        k0 = kind.unsqueeze(1)
        z = self.gene_emb(torch.clamp(key, 0, self.gene_emb.num_embeddings - 1)) * k0 + \
            self.cp_lin(self._fps[torch.clamp(key, 0, self._fps.shape[0] - 1)]) * (1 - k0)
        return F.normalize(z, dim=1)
    def forward(self, kind, key, cell):
        z = self.enc(kind, key)
        c = self.cell_emb(cell)
        return self.head(torch.cat([z, c], dim=1)), z, self.cls_head(z)

def nce_loss(z, groups, tau=0.15, groups2=None):
    """NT-Xent：同扰动 或 同 MoA 类别 互为正样本。"""
    sim = (z @ z.T) / tau
    N = z.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    mask = (groups[:, None] == groups[None, :]) & ~eye
    if groups2 is not None:
        same_cls = (groups2[:, None] == groups2[None, :]) & ~eye & (groups2[:, None] >= 0)
        mask = mask | same_cls
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    exp = sim.exp()
    denom = exp.sum(1) - exp.diag()
    num = (exp * mask.float()).sum(1)
    pos_n = mask.sum(1).float().clamp(min=1)
    return -(num / denom / pos_n * pos_n).clamp(min=1e-8).log().mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bpert", type=int, default=64, help="每 batch 的扰动数")
    ap.add_argument("--emb", type=int, default=512)
    ap.add_argument("--headw", type=int, default=1024)
    ap.add_argument("--pcc_w", type=float, default=0.5, help="PCC 感知损失权重")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam_nce", type=float, default=0.5)
    ap.add_argument("--save", default="g2cp_model_v4.pt")
    ap.add_argument("--progress", default="train_progress.json")
    ap.add_argument("--phase", default="joint", choices=["joint", "contrast", "finetune"])
    ap.add_argument("--from", dest="load_from", default="")
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--max_cell", type=int, default=20)
    ap.add_argument("--lam_cls", type=float, default=0.0)
    ap.add_argument("--lam_struct", type=float, default=0.0)
    ap.add_argument("--cls_bal", type=float, default=0.6, help="类别平衡采样比例（0=关闭）")
    ap.add_argument("--cls_per_batch", type=int, default=16)
    ap.add_argument("--freeze_enc", action="store_true", help="冻结编码器只训 head")
    args = ap.parse_args()

    m = np.load(os.path.join(CACHE_DIR, "meta.npz"), allow_pickle=True)
    kind, key, cell = m["kind"], m["key"], m["cell"]
    gene_vocab = [str(x) for x in m["gene_vocab"]]
    drug_vocab = [str(x) for x in m["drug_vocab"]]
    cl_names = [str(x) for x in m["cl_names"]]
    n_gene, n_drug = len(gene_vocab), len(drug_vocab)
    # prep_extra 已按 valid 过滤（y/meta 均为有效行），直接读取
    yv = np.load(os.path.join(CACHE_DIR, "y.npy"))
    fps = np.load(os.path.join(CACHE_DIR, "drug_fps.npy"))
    n = len(kind)
    log(f"数据: {n} 样本（基因 {n_gene} / 药物 {n_drug} / 细胞系 {len(cl_names)}）→ {yv.shape[1]} HVG")

    # 扰动唯一标签（基因/药物分开），按扰动-细胞系组织样本
    pert_uid = np.where(kind == 0, key, n_gene + key)
    pert2idx = defaultdict(list)
    for i in range(n):
        pert2idx[int(pert_uid[i])].append(i)
    perts = sorted(pert2idx.keys())
    # 药物 MoA 类别（CMAP_mmc1 PCL）：同类药互为对比正对
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
    log(f"药物类别（MoA）: {n_pcl} 类，覆盖药物样本 {int((cls_arr >= 0).sum())}")
    cls_t = torch.from_numpy(cls_arr).long().to(DEVICE)
    # 类别 → 药物样本索引（用于类别平衡采样）
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

    net = G2CPNet(n_gene, ECFP4_BITS, args.emb, len(cl_names), yv.shape[1], args.headw).to(DEVICE)
    net.cls_out = n_pcl
    net._fps = torch.from_numpy(fps).to(DEVICE)
    if args.load_from:
        ck0 = torch.load(args.load_from, map_location="cpu", weights_only=False)
        sd = ck0["net"]
        h1 = sd.get("head.1.weight")
        if h1 is not None and tuple(h1.shape) == tuple(net.head[1].weight.shape):
            net.load_state_dict(sd)  # head 尺寸匹配 → 全量继承（续训场景）
            log(f"已加载预训练 {args.load_from}（全量，含 head）")
        else:
            sd2 = {k: v for k, v in sd.items() if not k.startswith("head")}
            net.load_state_dict(sd2, strict=False)
            log(f"已加载预训练 {args.load_from}（仅编码器，head 新训）")
    if args.freeze_enc:
        for nm, p in net.named_parameters():
            if not nm.startswith("head"):
                p.requires_grad = False
        log("已冻结编码器（gene_emb/cp_lin/cell_emb），只训表型 head")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    kv_t = torch.from_numpy(key).long().to(DEVICE)
    kind_t = torch.from_numpy(kind).float().to(DEVICE)
    cell_t = torch.from_numpy(cell).long().to(DEVICE)
    yv_t = torch.from_numpy(yv).to(DEVICE)
    log(f"模型参数: {sum(p.numel() for p in net.parameters())/1e6:.2f}M | 显存 {torch.cuda.memory_allocated()/1048576:.0f} MB | 设备 {DEVICE}")

    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        rng2 = np.random.RandomState(ep)
        total, tl, tn = 0, 0.0, 0.0
        n_batches = max(1, len(tr_pert) // args.bpert)
        n_cls_b = max(1, int(n_batches * args.cls_bal)) if (args.cls_bal > 0 and len(cls2drugs) > 1 and args.phase != "finetune") else 0
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
            if args.phase == "contrast":
                loss = args.lam_nce * ln
            elif args.phase == "finetune":
                loss = lm + args.pcc_w * lp + 0.2 * ln
            else:
                loss = lm + args.pcc_w * lp + args.lam_nce * ln
            if args.lam_cls > 0:
                cm = gc >= 0
                if cm.any():
                    cl = nn.functional.cross_entropy(zc[cm][:, :net.cls_out], gc[cm])
                    loss = loss + args.lam_cls * cl
            opt.zero_grad(); loss.backward(); opt.step()
            total += len(b); tl += loss.item() * len(b); tn += ln.item() * len(b) if torch.is_tensor(ln) else 0
        sched.step()
        if args.lam_struct > 0:
            for _ in range(10):
                pick_d = rng2.choice(n_drug, size=64, replace=False)
                pd = torch.from_numpy(pick_d).long().to(DEVICE)
                zd = F.normalize(net.cp_lin(net._fps[pd]), dim=1)
                cosd = zd @ zd.T
                fa = net._fps[pd]
                inter = fa @ fa.T
                union = fa.sum(1)[:, None] + fa.sum(1)[None, :] - inter
                tgt = inter / (union + 1e-8)
                ls = F.mse_loss(cosd, tgt)
                (args.lam_struct * ls).backward()
                opt.step(); opt.zero_grad()
        secs = time.time() - t0
        eta = secs * (args.epochs - ep - 1)
        prog = {"epoch": ep + 1, "epochs": args.epochs, "loss": round(tl / total, 4),
                "nce": round(tn / total, 4), "secs_per_epoch": round(secs, 1),
                "eta_min": round(eta / 60, 1), "lr": round(opt.param_groups[0]["lr"], 6)}
        json.dump(prog, open(args.progress, "w"))
        log(f"epoch {ep+1}/{args.epochs} | loss {prog['loss']:.4f} | nce {prog['nce']:.4f} | {secs:.0f}s | 剩余 {eta/60:.1f} min")
        if (ep + 1) % 20 == 0:
            torch.save({"net": net.state_dict(), "gene_vocab": gene_vocab, "drug_vocab": drug_vocab,
                        "cl_names": cl_names, "hvg": json.load(open(os.path.join(CACHE_DIR, "hvg.json")))}, args.save)
            log(f"  中间权重 {args.save}")
    torch.save({"net": net.state_dict(), "gene_vocab": gene_vocab, "drug_vocab": drug_vocab,
                "cl_names": cl_names, "hvg": json.load(open(os.path.join(CACHE_DIR, "hvg.json")))}, args.save)
    log(f"训练完成 {args.save}")
    try:
        if os.path.isfile(args.progress):
            os.remove(args.progress)
    except Exception:
        pass


def prep_extra():
    """全细胞系数据预处理：读 data/lincs_extra/merged.h5ad（或 all_cells.h5ad）
    → 生成训练缓存（meta.npz / y.npy / drug_fps.npy / hvg.json）。
    - 细胞系数动态（任意多）
    - 环境变量 QC_ONLY=1 时只保留 qc_pass==1 样本
    - 药物 SMILES 用 processed_CMAP_compound_info.csv 重建（覆盖新药）
    """
    import anndata as ad
    import pandas as pd
    from unipret.compound_encoder import smiles_to_ecfp4

    src = os.path.join(BASE, "data", "lincs_extra", "merged.h5ad")
    if not os.path.isfile(src):
        src = os.path.join(BASE, "data", "lincs_extra", "all_cells.h5ad")
    if not os.path.isfile(src):
        log("prep_extra: 找不到合并数据，请先跑 parse_gctx/retrain_all")
        return

    log(f"读取 {src} ...")
    a = ad.read_h5ad(src, backed="r")
    obs = a.obs
    mask = (obs["pert_type"].astype(str) == "trt_cp") | (obs["pert_type"].astype(str) == "trt_xpr")
    if os.environ.get("QC_ONLY") == "1" and "qc_pass" in obs.columns:
        q = obs["qc_pass"].fillna(1).astype(int) == 1
        log(f"QC 过滤：{mask.sum()} → {(mask & q).sum()} 样本")
        mask = mask & q
    idx = np.where(mask.values)[0]
    log(f"有效样本 {len(idx)}")

    gacc = sorted(obs.loc[mask & (obs["pert_type"].astype(str) == "trt_xpr"), "cmap_name"]
                  .astype(str).str.replace("UniProt accession:", "", regex=False).unique().tolist())
    gene_vocab = {g: i for i, g in enumerate(gacc)}
    cp_ids = sorted(obs.loc[mask & (obs["pert_type"].astype(str) == "trt_cp"), "pert_id"].astype(str).unique().tolist())
    # 只保留标准 BRD- ID（能匹配 SMILES 映射）
    cp_ids = [p for p in cp_ids if p.startswith("BRD-")]
    drug_vocab = {p: i for i, p in enumerate(cp_ids)}
    log(f"基因 {len(gene_vocab)} | 药物 {len(drug_vocab)}")

    smi_path = os.path.join(BASE, "data", "g2cp", "data", "processed_CMAP_compound_info.csv")
    pid2sm = {}
    if os.path.isfile(smi_path):
        info = pd.read_csv(smi_path)
        if "pert_id" in info.columns and "canonical_smiles" in info.columns:
            pid2sm = {str(r["pert_id"]): str(r["canonical_smiles"])
                      for _, r in info.iterrows() if pd.notna(r["canonical_smiles"])}
    legacy = os.path.join(BASE, "data", "lincs_drug_smiles.json")
    if os.path.isfile(legacy):
        try:
            pid2sm.update(json.load(open(legacy)))
        except Exception:
            pass
    log(f"SMILES 映射共 {len(pid2sm)} 个药")
    fps = np.zeros((len(cp_ids), ECFP4_BITS), dtype=np.float32)
    n_ok = 0
    for i, pid in enumerate(cp_ids):
        sm = pid2sm.get(pid)
        if sm:
            f = smiles_to_ecfp4(sm)
            if f is not None and f.any():
                fps[i] = f
                n_ok += 1
    log(f"药物指纹解析 {n_ok}/{len(cp_ids)}")

    cl_vals = obs.iloc[idx]["cell_line"].astype(str).unique().tolist()
    cl_map = {v: i for i, v in enumerate(cl_vals)}
    log(f"细胞系 {len(cl_vals)} 个: {cl_vals}")

    kind = np.zeros(len(idx), dtype=np.int8)
    key = np.zeros(len(idx), dtype=np.int32)
    cell = np.zeros(len(idx), dtype=np.int32)  # 289+ 细胞系，必须 int32
    rows = obs.iloc[idx]
    for j, (_, r) in enumerate(rows.iterrows()):
        if str(r["pert_type"]) == "trt_xpr":
            acc = str(r["cmap_name"]).replace("UniProt accession:", "")
            kind[j] = 0
            key[j] = gene_vocab.get(acc, -1)
        else:
            pid = str(r["pert_id"])
            kind[j] = 1
            key[j] = drug_vocab.get(pid, -1)
        cell[j] = cl_map.get(str(r["cell_line"]), 0)
    valid = key >= 0
    # 无指纹药物样本过滤（防止全零指纹垃圾信号）
    zero_fp = ~fps.any(axis=1)
    if zero_fp.any():
        bad = 0
        for i in range(len(idx)):
            if kind[i] == 1 and key[i] >= 0 and zero_fp[key[i]]:
                valid[i] = False
                bad += 1
        log(f"过滤无指纹药物样本 {bad} 个")
    np.savez(os.path.join(CACHE_DIR, "meta.npz"),
             kind=kind[valid], key=key[valid], cell=cell[valid],
             cl_names=np.array(cl_vals, dtype=object),
             gene_vocab=np.array(gacc, dtype=object),
             drug_vocab=np.array(cp_ids, dtype=object))
    np.save(os.path.join(CACHE_DIR, "drug_fps.npy"), fps)
    np.save(os.path.join(CACHE_DIR, "valid.npy"), valid)
    n = valid.sum()
    log(f"有效样本 {n}（基因 {(kind[valid]==0).sum()} + 药物 {(kind[valid]==1).sum()}）")
    y = np.zeros((n, a.shape[1]), dtype=np.float32)
    vj = np.where(valid)[0]
    for b in range(0, len(vj), 5000):
        bi = vj[b:b+5000]
        Xb = a.X[idx[bi]]
        if hasattr(Xb, "toarray"):
            Xb = Xb.toarray()
        y[b:b+len(bi)] = np.asarray(Xb, dtype=np.float32)
    np.save(os.path.join(CACHE_DIR, "y.npy"), y)
    hvg = list(a.var_names)
    json.dump(hvg, open(os.path.join(CACHE_DIR, "hvg.json"), "w"))
    log(f"y 矩阵 {y.shape} | HVG {len(hvg)} | 缓存完成")
    a.file.close()


if __name__ == "__main__":
    main()
