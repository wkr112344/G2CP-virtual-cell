# -*- coding: utf-8 -*-
"""G2CP 论文级重训：LINCS 93,369 扰动样本（7,860 药 ECFP4 + 4,994 基因嵌入）→ 978 HVG 表达回归。
带实时进度侦测：每 epoch 写 train_progress.json + 无缓冲日志，可随时 tail 查看。"""
import sys, os, json, time, argparse, random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "g2cp", "data", "LINCS", "pool")
POOL_H5AD = os.path.join(DATA, "pool_gene_chem_ctrl_adata.h5ad")
DRUG_SMILES = os.path.join(BASE, "data", "lincs_drug_smiles.json")
CACHE_DIR = os.path.join(BASE, "data", "g2cp_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

from unipret.compound_encoder import smiles_to_ecfp4, ECFP4_BITS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def prep():
    """一次性预处理：样本元数据 + 药物指纹 + y 矩阵（存缓存）"""
    if os.path.isfile(os.path.join(CACHE_DIR, "meta.npz")) and os.path.isfile(os.path.join(CACHE_DIR, "y.npy")):
        log("预处理缓存已存在，跳过")
        return
    import anndata as ad
    log("读取 pool h5ad（93k 样本）...")
    a = ad.read_h5ad(POOL_H5AD, backed="r")
    obs = a.obs
    mask = (obs["pert_type"] == "trt_cp") | (obs["pert_type"] == "trt_xpr")
    idx = np.where(mask.values)[0]
    log(f"有效样本 {len(idx)}（trt_cp 药物 + trt_xpr 基因，排除对照）")
    # 药物 SMILES 表
    drug_smiles = json.load(open(DRUG_SMILES, encoding="utf-8"))
    # 基因 vocab（UniProt accession）
    gacc = sorted(obs.loc[obs["pert_type"] == "trt_xpr", "cmap_name"].astype(str).str.replace("UniProt accession:", "", regex=False).unique().tolist())
    gene_vocab = {g: i for i, g in enumerate(gacc)}
    log(f"基因 vocab: {len(gene_vocab)}（UniProt accession）")
    # 药物指纹表
    cp_ids = sorted(obs.loc[obs["pert_type"] == "trt_cp", "pert_id"].astype(str).unique().tolist())
    drug_vocab = {p: i for i, p in enumerate(cp_ids)}
    fps = np.zeros((len(cp_ids), ECFP4_BITS), dtype=np.float32)
    n_ok = 0
    for i, pid in enumerate(cp_ids):
        sm = drug_smiles.get(pid, "")
        fp = smiles_to_ecfp4(sm) if sm else None
        if fp is not None and fp.any():
            fps[i] = fp
            n_ok += 1
    log(f"药物指纹: {len(cp_ids)} 个, 解析成功 {n_ok}（{n_ok*100//max(len(cp_ids),1)}%）")
    np.save(os.path.join(CACHE_DIR, "drug_fps.npy"), fps)
    # 样本元数据（向量化）
    rows = obs.iloc[idx]
    pt = rows["pert_type"].values
    cmap = rows["cmap_name"].astype(str).values
    pid = rows["pert_id"].astype(str).values
    clv = rows["cell_line"].astype(str).values
    kind = np.zeros(len(idx), dtype=np.int8)
    key = np.zeros(len(idx), dtype=np.int32)
    for j in range(len(idx)):
        if pt[j] == "trt_xpr":
            acc = cmap[j].replace("UniProt accession:", "")
            kind[j] = 0
            key[j] = gene_vocab.get(acc, -1)
        else:
            kind[j] = 1
            key[j] = drug_vocab.get(pid[j], -1)
    cl_vals = sorted(set(clv))
    cl_map = {v: i for i, v in enumerate(cl_vals)}
    cell = np.array([cl_map[v] for v in clv], dtype=np.int8)
    valid = key >= 0
    np.savez(os.path.join(CACHE_DIR, "meta.npz"),
             kind=kind[valid], key=key[valid], cell=cell[valid],
             cl_names=np.array(cl_vals, dtype=object), gene_vocab=np.array(gacc, dtype=object),
             drug_vocab=np.array(cp_ids, dtype=object))
    log(f"有效样本 {valid.sum()}（基因 {int((kind[valid]==0).sum())} + 药物 {int((kind[valid]==1).sum())}）")
    # y 矩阵（分块读，避免逐行慢）
    n_col = a.shape[1]
    y = np.zeros((len(idx), n_col), dtype=np.float32)
    for s in range(0, len(idx), 5000):
        e = min(s + 5000, len(idx))
        chunk = a.X[idx[s:e]]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        y[s:e] = np.asarray(chunk, dtype=np.float32)
        if s % 20000 == 0:
            log(f"  y 矩阵读取 {e}/{len(idx)}")
    np.save(os.path.join(CACHE_DIR, "y.npy"), y)
    np.save(os.path.join(CACHE_DIR, "valid.npy"), valid)
    hvg = list(a.var_names)
    json.dump(hvg, open(os.path.join(CACHE_DIR, "hvg.json"), "w"))
    log(f"y 矩阵已存 {y.shape}；HVG {len(hvg)} 个")
    a.file.close()

class G2CPNet(nn.Module):
    """统一空间：基因(嵌入) / 药物(ECFP4→Linear) → 256 维 → +细胞系 → 978 HVG 表达"""
    def __init__(self, n_gene, fp_dim, emb=256, n_cell=5, n_out=978):
        super().__init__()
        self.gene_emb = nn.Embedding(n_gene, emb)
        self.cp_lin = nn.Linear(fp_dim, emb)
        nn.init.kaiming_normal_(self.cp_lin.weight)
        self.cell_emb = nn.Embedding(n_cell, 16)
        self.head = nn.Sequential(nn.LayerNorm(emb + 16), nn.Linear(emb + 16, 512), nn.GELU(),
                                  nn.Linear(512, n_out))
    def forward(self, kind, key, cell):
        if kind == 0:
            z = self.gene_emb(key)
        else:
            z = self.cp_lin(key)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        c = self.cell_emb(cell)
        return self.head(torch.cat([z, c], dim=1))

def train(args):
    prep()
    m = np.load(os.path.join(CACHE_DIR, "meta.npz"), allow_pickle=True)
    kind, key, cell = m["kind"], m["key"], m["cell"]
    gene_vocab = [str(x) for x in m["gene_vocab"]]
    drug_vocab = [str(x) for x in m["drug_vocab"]]
    n_gene = len(gene_vocab)
    n_drug = len(drug_vocab)
    cl_names = [str(x) for x in m["cl_names"]]
    y = np.load(os.path.join(CACHE_DIR, "y.npy"), mmap_mode="r")
    valid = np.load(os.path.join(CACHE_DIR, "valid.npy"))
    fps = np.load(os.path.join(CACHE_DIR, "drug_fps.npy"))
    n = int(valid.sum())
    yv = y[valid]
    kv = key[valid]
    log(f"训练数据: {n} 样本（基因 {n_gene} / 药物 {n_drug} / 细胞系 {len(cl_names)}）→ {yv.shape[1]} HVG")
    # 随机 split（90% 训练 / 10% 评测预留）
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    n_tr = int(n * 0.9)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    net = G2CPNet(n_gene, ECFP4_BITS, args.emb, len(cl_names), yv.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    fps_t = torch.from_numpy(fps).to(DEVICE)
    log(f"模型参数: {sum(p.numel() for p in net.parameters())/1e6:.2f}M | 设备 {DEVICE}")
    # 数据一次性放 GPU，索引全在 GPU 上算（避免逐 batch 搬运）
    kv_t = torch.from_numpy(kv).long().to(DEVICE)
    kind_t = torch.from_numpy(kind[valid]).float().to(DEVICE)
    cell_t = torch.from_numpy(cell[valid]).long().to(DEVICE)
    yv_t = torch.from_numpy(yv).to(DEVICE)
    log(f"训练数据已载入 GPU（显存约 {torch.cuda.memory_allocated()/1048576:.0f} MB）")
    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        rng2 = np.random.RandomState(ep)
        bs = args.batch
        total, tot_loss = 0, 0.0
        order = rng2.permutation(tr_idx)
        for s in range(0, len(order), bs):
            b = torch.from_numpy(order[s:s + bs]).long().to(DEVICE)
            k = kv_t[b]
            kind0 = kind_t[b].unsqueeze(1)
            z = net.gene_emb(torch.clamp(k, 0, n_gene - 1)) * kind0 + \
                net.cp_lin(fps_t[torch.clamp(k, 0, n_drug - 1)]) * (1 - kind0)
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            c = net.cell_emb(cell_t[b])
            out = net.head(torch.cat([z, c], dim=1))
            yb = yv_t[b]
            loss_mse = nn.functional.mse_loss(out, yb)
            om = out - out.mean(1, keepdim=True)
            ym = yb - yb.mean(1, keepdim=True)
            pcc = (om * ym).sum(1) / (om.norm(dim=1) * ym.norm(dim=1) + 1e-8)
            loss_pcc = (1 - pcc).mean()
            loss = loss_mse + 0.5 * loss_pcc
            opt.zero_grad(); loss.backward(); opt.step()
            total += len(b); tot_loss += loss.item() * len(b)
        train_loss = tot_loss / total
        secs = time.time() - t0
        eta = secs * (args.epochs - ep - 1)
        prog = {"epoch": ep + 1, "epochs": args.epochs, "loss": round(train_loss, 4),
                "secs_per_epoch": round(secs, 1), "eta_min": round(eta / 60, 1),
                "device": str(DEVICE), "samples": total}
        json.dump(prog, open(args.progress, "w"))
        log(f"epoch {ep+1}/{args.epochs} | loss {train_loss:.4f} | {secs:.0f}s | 剩余约 {eta/60:.1f} min")
        if (ep + 1) % 5 == 0:
            torch.save({"net": net.state_dict(), "gene_vocab": gene_vocab, "drug_vocab": drug_vocab,
                        "cl_names": cl_names, "hvg": json.load(open(os.path.join(CACHE_DIR, "hvg.json")))},
                       args.save)
            log(f"  中间权重已存 {args.save}")
    torch.save({"net": net.state_dict(), "gene_vocab": gene_vocab, "drug_vocab": drug_vocab,
                "cl_names": cl_names, "hvg": json.load(open(os.path.join(CACHE_DIR, "hvg.json")))},
               args.save)
    log(f"训练完成，最终权重 {args.save}")
    if os.path.isfile(args.progress):
        os.remove(args.progress)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save", default="g2cp_model.pt")
    ap.add_argument("--progress", default="train_progress.json")
    train(ap.parse_args())
