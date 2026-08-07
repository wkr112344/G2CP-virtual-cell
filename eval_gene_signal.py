# -*- coding: utf-8 -*-
"""批量评估 4,994 基因敲除的信号强度，挑出值得研究的基因候选。"""
import sys, os, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_g2cp_contrast import G2CPNet, CACHE_DIR
from unipret.compound_encoder import ECFP4_BITS

ck = torch.load("g2cp_v10.pt", map_location="cpu", weights_only=False)
gene_vocab = [str(x) for x in ck["gene_vocab"]]
hvg = list(ck["hvg"])
emb = ck["net"]["head.0.weight"].shape[0] - 32
headw = ck["net"]["head.1.weight"].shape[0]
net = G2CPNet(len(gene_vocab), ECFP4_BITS, emb, len(ck["cl_names"]), len(hvg), headw)
net.load_state_dict(ck["net"], strict=False)
net.eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(DEVICE)

with torch.no_grad():
    idx = torch.arange(len(gene_vocab), device=DEVICE).long()
    z = F.normalize(net.gene_emb(idx), dim=1)
    cell = torch.zeros(len(gene_vocab), device=DEVICE).long()  # A375
    out = net.head(torch.cat([z, net.cell_emb(cell)], dim=1)).cpu().numpy()  # [4994, 978]

n = out.shape[0]
rows = []
for i in range(n):
    v = out[i]
    av = np.abs(v)
    rows.append((float(np.sort(av)[-5:].mean()), int((av > 2).sum()),
                 int((av > 1.5).sum()), int((av > 1).sum()), i))
rows.sort(reverse=True)

# 排除管家/结构性基因（无研究故事）
black = ("RPL", "RPS", "MRPL", "MRPS", "MT-", "HIST", "SNRNP", "ATP5", "COX",
         "NDUF", "UQCR", "RP[0-9]", "SNR", "TUBB", "ACT", "GAPDH")
cands = []
for t5, n2, n15, n1, i in rows:
    g = gene_vocab[i]
    if any(g.startswith(b) for b in black):
        continue
    cands.append((g, t5, n2, n15, n1, i))

print(f"=== 4,994 基因敲除信号强度 Top 30（A375，排除管家基因）===")
print(f"{'基因':12s} {'top5|z|':>8s} {'|z|>2':>6s} {'|z|>1.5':>7s} {'|z|>1':>6s}")
for g, t5, n2, n15, n1, i in cands[:30]:
    print(f"{g:12s} {t5:8.2f} {n2:6d} {n15:7d} {n1:6d}")
