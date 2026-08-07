# -*- coding: utf-8 -*-
"""批量面检验 v2：目标集 = 978 HVG 可观察的 DNA 修复基因（12 个）"""
import sys, os, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_g2cp_contrast import G2CPNet, CACHE_DIR
from unipret.compound_encoder import ECFP4_BITS
from scipy.stats import hypergeom

DNA_REPAIR_ALL = ["BRCA1","BRCA2","ATM","ATR","RAD51","RPA1","RPA2","RPA3","MRE11","RAD50","NBN","EXO1",
                  "PALB2","FANCD2","FANCA","MLH1","MSH2","MSH6","TP53BP1","MDC1","CHEK1","CHEK2","BLM",
                  "WRN","LIG1","LIG3","XRCC4","XRCC5","XRCC6","ERCC1","ERCC2","ERCC4","ERCC5","PARP1",
                  "POLD1","POLE","DDB1","DDB2","RAD23B","NTHL1","UNG","OGG1","MUTYH","PNKP","APEX1"]
REPAIR_KO = ["BRCA1","BRCA2","ATM","ATR","RAD51","MRE11","NBN","EXO1","FANCA","TP53BP1",
             "CHEK1","CHEK2","BLM","LIG1","XRCC6","PARP1","MDC1"]

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

gv = {g: i for i, g in enumerate(gene_vocab)}
hset = set(hvg)
TARGET = [g for g in DNA_REPAIR_ALL if g in hset]
n_obs = len(TARGET)
print(f"目标集（978 可观察修复基因）: {n_obs} 个: {TARGET}")


def ko_down(gene, topn=20):
    i = gv[gene]
    with torch.no_grad():
        z = F.normalize(net.gene_emb(torch.tensor([i], device=DEVICE).long()), dim=1)
        cell = torch.zeros(1, device=DEVICE).long()
        out = net.head(torch.cat([z, net.cell_emb(cell)], dim=1)).cpu().numpy()[0]
    return [hvg[j] for j in np.argsort(out)[:topn]]


hits = []
for g in REPAIR_KO:
    if g not in gv:
        continue
    down = ko_down(g)
    hit = [x for x in down if x in set(TARGET)]
    hits.append(len(hit))
    print(f"  {g:8s} 敲除 → 下调含修复基因 {len(hit)} 个: {hit}")

rng = np.random.RandomState(0)
non_repair = [g for g in gene_vocab if g not in set(REPAIR_KO)]
ctrl = []
for _ in range(50):
    pick = rng.choice(non_repair, size=len(REPAIR_KO), replace=False)
    hh = [len(set(ko_down(g)) & set(TARGET)) for g in pick if g in gv]
    ctrl.append(float(np.mean(hh)))
cm, cs = float(np.mean(ctrl)), float(np.std(ctrl))
rm = float(np.mean(hits))
M, n, k = len(hvg), n_obs, 20
pvals = [hypergeom.sf(h - 1, M, n, k) for h in hits]
n_sig = sum(1 for p in pvals if p < 0.05)
print()
print("===== 批量面检验结果 =====")
print(f"修复组：平均命中 {rm:.2f} 个修复基因（{len(REPAIR_KO)} 个修复基因敲除）")
print(f"随机组：平均命中 {cm:.2f} ± {cs:.2f}（50 轮对照）")
print(f"超几何富集 p 值（单基因最小 5 个）: {[f'{p:.3g}' for p in sorted(pvals)[:5]]}")
print(f"p<0.05 的基因数: {n_sig}/{len(REPAIR_KO)}")
verdict = "✅ 系统性学到 DNA 修复通路语义（显著富集）" if (rm > cm + 2 * cs and n_sig >= 3) else "⚠️ 富集不显著，单点吻合可能属巧合"
print(verdict)
