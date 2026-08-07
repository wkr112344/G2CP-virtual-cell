# -*- coding: utf-8 -*-
"""pre_train_depmap.py —— 路线3 阶段A: DepMap 基因必需性预训练(论文遗传预训练)。

任务: 输入 (基因, DepMap细胞系) → 预测 CRISPR 基因效应评分(必需性)。
- gene_emb(4,994×512, 用 ESM 蛋白语义初始化) + cell_emb(1,095×64) → head → 1
- 学到: 基因嵌入携带"功能相似性"(在哪些细胞系共必需), 这是论文"遗传预训练"的核心
- 输出: pre_train_depmap.pt, 阶段B 用 --from 加载 gene_emb 迁移到主模型
"""
import sys, os, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


class DepMapNet(nn.Module):
    def __init__(self, n_gene, n_cell, emb=512, cell_emb=64):
        super().__init__()
        self.gene_emb = nn.Embedding(n_gene, emb)
        self.cell_emb = nn.Embedding(n_cell, cell_emb)
        self.head = nn.Sequential(nn.Linear(emb + cell_emb, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, g, c):
        z = torch.cat([self.gene_emb(g), self.cell_emb(c)], dim=1)
        return self.head(z).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--n_samples', type=int, default=400000, help='每 epoch 采样对数')
    ap.add_argument('--save', default='pre_train_depmap.pt')
    args = ap.parse_args()

    # 基因词表(主模型同款)
    import json
    ck = torch.load(os.path.join(BASE, 'g2cp_full_esm_all.pt'), map_location='cpu', weights_only=False)
    gene_vocab = [str(x) for x in ck['gene_vocab'] if not str(x).startswith('IMM_')]
    gv_idx = {g: i for i, g in enumerate(gene_vocab)}
    n_gene = len(gene_vocab)
    log(f'基因词表: {n_gene}')

    # DepMap 必需性矩阵(只取词表内基因)
    csv_p = os.path.join(BASE, 'data', 'depmap', 'CRISPRGeneEffect.csv')
    log('读取 DepMap(只取需要的列)...')
    use_cols = ['ModelID']
    for c in pd.read_csv(csv_p, nrows=0).columns:
        sym = c.split(' (')[0]
        if sym in gv_idx:
            use_cols.append(c)
    df = pd.read_csv(csv_p, usecols=use_cols)
    model_ids = df['ModelID'].tolist()
    raw = df.drop(columns=['ModelID']).values.astype(np.float32)  # (n_cell, n_gene_match)
    valid = ~np.isnan(raw)                 # 有测量的位置
    X = np.nan_to_num(raw, nan=0.0).astype(np.float32)
    n_cell = X.shape[0]
    cols = list(df.drop(columns=['ModelID']).columns)
    gene_col_idx = [gv_idx[c.split(' (')[0]] for c in cols]
    log(f'DepMap: {n_cell} 细胞系 × {X.shape[1]} 基因(词表内 {X.shape[1]}/{n_gene})')

    net = DepMapNet(n_gene, n_cell).to(DEVICE)
    # gene_emb 用 ESM 语义初始化(继承路线1)
    esm_cache = torch.load(os.path.join(BASE, 'data', 'depmap', 'esm_cache.pt'), map_location='cpu', weights_only=False)
    with torch.no_grad():
        for i, g in enumerate(gene_vocab):
            v = esm_cache.get(g)
            if v is not None:
                t = torch.from_numpy(v if hasattr(v, 'numpy') else v)
                net.gene_emb.weight.data[i, :320] = t / (t.norm() + 1e-8)
    log(f'gene_emb 已用 ESM 初始化 | 参数 {sum(p.numel() for p in net.parameters())/1e6:.1f}M')

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    Xt = torch.from_numpy(X).to(DEVICE)
    valid_t = torch.from_numpy(valid).to(DEVICE)
    gm = np.array(gene_col_idx, dtype=np.int64)
    gm_t = torch.from_numpy(gm).to(DEVICE)
    # 缺失基因列: 补零均值
    miss_mask = torch.ones(n_gene, dtype=torch.bool, device=DEVICE)
    miss_mask[gm_t] = False
    n_miss = int(miss_mask.sum())

    for ep in range(args.epochs):
        t0 = time.time()
        rng = np.random.RandomState(ep)
        ci = rng.randint(0, n_cell, args.n_samples)
        gi = rng.randint(0, n_gene, args.n_samples)
        total, n_seen = 0.0, 0
        for s0 in range(0, args.n_samples, args.batch):
            b = slice(s0, s0 + args.batch)
            cb = torch.from_numpy(ci[b]).long().to(DEVICE)
            gb = torch.from_numpy(gi[b]).long().to(DEVICE)
            # 目标: 词表内且有测量的基因用 DepMap 值, 缺失/未测给 0 并 mask
            target = torch.zeros(len(gb), device=DEVICE)
            m_valid = torch.zeros(len(gb), dtype=torch.bool, device=DEVICE)
            in_map = torch.isin(gb, gm_t)
            mapped = gb[in_map]
            if len(mapped):
                pos = torch.searchsorted(gm_t, mapped)
                valid_sel = valid_t[cb[in_map], pos]
                target[in_map] = Xt[cb[in_map], pos]
                m_valid[in_map] = valid_sel
            pred = net(gb, cb)
            mask = m_valid.float()
            if mask.sum() < 8:
                continue
            l = F.mse_loss(pred * mask, target * mask)
            l.backward()
            opt.step()
            opt.zero_grad()
            total += l.item() * len(gb)
            n_seen += len(gb)
        sched.step()
        log(f'ep {ep+1}/{args.epochs} | loss {total/n_seen:.4f} | {time.time()-t0:.0f}s')
        if (ep + 1) % 3 == 0 or ep + 1 == args.epochs:
            torch.save({'net': net.state_dict(), 'gene_vocab': gene_vocab, 'n_cell': n_cell,
                        'model_ids': model_ids}, args.save)
            log(f'  保存 {args.save}')
    log(f'完成 {args.save}')


if __name__ == '__main__':
    main()
