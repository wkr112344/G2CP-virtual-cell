"""
两阶段训练脚本（G2CP：遗传→化学迁移，且 G2CP 直接吃 UniPert 统一嵌入）
============================================================
流程（论文对齐）：
  [UniPert 预训练]（可选 --stage pretrain）：用本地 CPI 正样本做跨域对比(info_nce) +
                  基因图自监督($L_{enhance}$)，产出 UniPert 统一编码器权重。
  [阶段 A 遗传预训练]：ONE UniPert（基因编码器=混合模式）+ head_gene，
                      在 sciPlex3/Norman 全部基因 CRISPR 扰动上训「基因扰动→表达变化」，
                      同时加 $L_{enhance}$（用扰动响应功能图）正则基因嵌入。
  [阶段 B 化学微调]：复用阶段 A 的 UniPert（基因编码器+head_gene 原样迁移），
                      加化合物路（UniPert 化合物编码器 + head_comp），
                      用对齐化合物真实转录组微调「药物扰动→表达变化」。

关键：G2CP 的 forward_gene/forward_compound 直接调 UniPert.encode_*，
      所以 G2CP 的输入就是 UniPert 的统一嵌入；基因编码器在 A→B 间共享迁移。
      不再像旧版那样各训各的独立嵌入表。

用法：
  python unipret/train_stages.py --demo                 # 阶段A管线快速验证(3050Ti)
  python unipret/train_stages.py --stage a --data norman --epochs 20
  python unipret/train_stages.py --stage a --data sciplex3 --epochs 20   # 全量基因预训练
  python unipret/train_stages.py --stage b --load stageA.pt --epochs 30
  python unipret/train_stages.py --stage pretrain        # UniPert 对比+图自监督预训练
"""
import os
import sys
import re
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unipret.config import (DEVICE, EMBED_DIM, BATCH_SIZE, GRAD_ACCUM,
                            LEARNING_RATE, WEIGHT_DECAY, CELL_LINE_NAMES, SEED,
                            W_ENHANCE, W_ALIGN, W_MOA)
from unipret.data_bridge import (PerturbationReader, load_local_dataset, build_samples,
                                  select_hvg, control_means_by_line, CompoundGraphCache,
                                  build_gene_samples_for_pretrain,
                                  build_compound_samples_chembl)
from unipret.compound_encoder import smiles_to_graph
from unipret.gene_encoder import GeneEmbeddingEncoder
from unipret.model import UniPert
from unipret.effect_model import PerturbationEffectModel
from unipret.protein_graph import (GeneGraphEnhance, build_gene_response_graph,
                                   build_sequence_graph)
from unipret.contrastive import info_nce
from unipret.io_adapters import load_local_maps

torch.manual_seed(SEED)
np.random.seed(SEED)

NORMAN = "C:/Users/wkr20/Desktop/virtual_cell_real_data/genetic/NormanWeissman2019_filtered.h5ad"
SCIPLEX3 = "C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad"
LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset.json")


def _make_unipert(num_genes, gene_vocab=None, gene_mode="esm"):
    """统一创建 UniPert；esm 模式失败自动退回 hybrid（保证不因 ESM 依赖中断训练）。"""
    from unipret.model import UniPert
    if gene_mode == "esm":
        try:
            return UniPert(num_genes=num_genes, gene_encoder_mode="esm",
                           gene_vocab=gene_vocab)
        except Exception as e:
            print(f"    ⚠️ ESM 模式初始化失败（{e}），退回 hybrid", flush=True)
    return UniPert(num_genes=num_genes, gene_encoder_mode=gene_mode,
                   gene_vocab=gene_vocab)


# ----------------------------------------------------------- 数据集
class GenePertDataset(Dataset):
    """基因扰动样本：gene_ids([k], padding 0) + seqs(list[str]|None) + cell_line + expr_delta。"""
    def __init__(self, samples):
        self.samples = samples
        self.max_k = max(len(s["gene_ids"]) for s in samples) if samples else 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        gids = s["gene_ids"] + [0] * (self.max_k - len(s["gene_ids"]))
        return {
            "gene_ids": torch.tensor(gids, dtype=torch.long),
            "seqs": s.get("seqs"),
            "cell_line": s["cell_line_idx"],
            "label": torch.tensor(s["expr_delta"], dtype=torch.float32),
        }


class CompoundPertDataset(Dataset):
    """化合物扰动样本：compound_graph + cell_line + expr_delta。"""
    def __init__(self, samples, cache):
        self.samples = samples
        self.cache = cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "graph": self.cache.get(s["smiles"]) if s.get("smiles") else None,
            "cell_line": s["cell_line_idx"],
            "label": torch.tensor(s["expr_delta"], dtype=torch.float32),
        }


def collate_gene(batch):
    seqs = [b["seqs"] for b in batch]
    # 只要有一个样本缺序列，整批退化为纯 ID 嵌入（GeneEncoder 容错）
    if any(s is None for s in seqs):
        seqs = None
    return {
        "gene_ids": torch.stack([b["gene_ids"] for b in batch]),
        "seqs": seqs,
        "cell_line": torch.tensor([b["cell_line"] for b in batch], dtype=torch.long),
        "label": torch.stack([b["label"] for b in batch]),
    }


def _attach_seqs(samples, name_to_seq):
    """给基因样本补 seqs（按 gene_names 查本地序列，查不到为 None）。"""
    for s in samples:
        seqs = [name_to_seq.get(nm.upper()) or name_to_seq.get(nm) for nm in s.get("gene_names", [])]
        s["seqs"] = seqs if any(seqs) else None
    return samples


# ----------------------------------------------------------- 训练循环（含可选 enhance）
def _train_loop(model, loader, optimizer, desc, epochs, is_gene=True,
                enhance=None, gene_graph=None):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    crit = torch.nn.MSELoss()
    for ep in range(epochs):
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        for step, batch in enumerate(loader):
            label = batch["label"].to(DEVICE)
            cl = batch["cell_line"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                if is_gene:
                    pred = model.forward_gene(batch["gene_ids"].to(DEVICE),
                                              seqs=batch.get("seqs"), cell_line_idx=cl)
                else:
                    pred = model.forward_compound(batch["graph"], cl)
                loss = crit(pred, label) / GRAD_ACCUM
                # $L_{enhance}$：基因图自监督（仅基因路有意义）
                if enhance is not None and gene_graph is not None and is_gene:
                    emb = model.unipert.gene_id_embedding()
                    loss = loss + (W_ENHANCE * enhance(emb, gene_graph[1],
                                                       maskable=gene_graph[0])) / GRAD_ACCUM
            scaler.scale(loss).backward()
            running += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if (step + 1) % GRAD_ACCUM != 0:  # flush 残余
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        print(f"  [{desc}] epoch {ep+1}/{epochs}  loss={running/len(loader):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)


# ----------------------------------------------------------- 阶段 A
def train_stage_a(data_path, hvg_n=2000, epochs=20, max_genes=None,
                  unipert_ckpt=None, ckpt="stageA.pt", with_enhance=True,
                  gene_mode="esm"):
    print(f">>> 阶段 A 遗传预训练：{os.path.basename(data_path)}", flush=True)
    reader = PerturbationReader(data_path, backed=True)
    reader.load_full_sparse()   # 整块稀疏矩阵进内存（仅一次慢读），之后切片秒级
    print("    ", reader.summary(), flush=True)
    print(">>> 选 HVG (top-%d) ..." % hvg_n, flush=True)
    hvg = select_hvg(reader, n=hvg_n)
    print("    HVG 选定 (%d 维)" % len(hvg), flush=True)
    t0 = time.time()
    samples, gene_vocab = build_gene_samples_for_pretrain(
        reader, hvg, max_cells=2000, max_genes=max_genes)
    # 补序列（混合模式用）
    _, name_to_smiles, _ = load_local_maps(LOCAL)
    name_to_seq = {k: v for k, v in
                   load_local_dataset(LOCAL).get("proteins", {}).items()
                   if isinstance(v, dict) and v.get("seq")}
    name_to_seq = {k.upper(): v["seq"] for k, v in name_to_seq.items()}
    samples = _attach_seqs(samples, name_to_seq)
    print(f"    基因样本 {len(samples)} 个，基因词典 {len(gene_vocab)} 个 "
          f"({time.time()-t0:.0f}s)", flush=True)
    if not samples:
        print("    !! 无基因样本，退出", flush=True)
        reader.close()
        return None

    # ONE UniPert（全基因词表），可选从预训练权重初始化重叠基因
    num_genes = len(gene_vocab) + 1
    unipert = _make_unipert(num_genes, gene_vocab=gene_vocab, gene_mode=gene_mode).to(DEVICE)
    if unipert_ckpt and os.path.isfile(unipert_ckpt):
        _init_unipert_from(unipert, unipert_ckpt, gene_vocab)

    effect = PerturbationEffectModel.build_stage_a(unipert, hvg_n).to(DEVICE)
    enhance = GeneGraphEnhance().to(DEVICE) if with_enhance else None
    gene_graph = None
    if enhance is not None:
        print("    构建扰动响应功能图（基因图自监督用，复用已建样本）...", flush=True)
        node_ids, edges = build_gene_response_graph(
            hvg=hvg, samples=samples, max_genes=1500, k=8)
        gene_graph = (node_ids, edges)
        print(f"    基因图：{len(node_ids)} 节点 / {len(edges)} 边", flush=True)

    ds = GenePertDataset(samples)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_gene)
    opt = torch.optim.AdamW(effect.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    _train_loop(effect, loader, opt, "stageA", epochs, is_gene=True,
                enhance=enhance, gene_graph=gene_graph)
    torch.save({"unipert": unipert.state_dict(), "effect": effect.state_dict(),
                "gene_vocab": gene_vocab, "hvg_dim": hvg_n, "gene_graph": gene_graph}, ckpt)
    print(f"    阶段 A 权重已存 {ckpt}", flush=True)
    reader.close()
    return ckpt


# ----------------------------------------------------------- 阶段 B
def train_stage_b(stage_a_ckpt, hvg_n=2000, epochs=30, ckpt="stageB.pt",
                  with_enhance=True, gene_mode="esm"):
    print(">>> 阶段 B 化学微调", flush=True)
    sa = torch.load(stage_a_ckpt, map_location=DEVICE)
    gene_vocab = sa["gene_vocab"]
    hvg_n = sa.get("hvg_dim", hvg_n)
    reader = PerturbationReader(SCIPLEX3, backed=True)
    reader.load_full_sparse()   # 够内存则整块进内存；sciPlex3 太大自动回退磁盘
    local = load_local_dataset(LOCAL)
    # sciPlex3 基因名损坏(nan)且规模大，直接在 sciPlex3 上按方差选 HVG（分层抽样加速）
    hvg = select_hvg(reader, n=hvg_n, max_cells=50000)
    # P2：优先走 chembl 直连（sciPlex3 全量 ~150 药、3 细胞系）；无 chembl 文件才退回名字匹配
    chem_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "sciplex3_chems.json")
    if os.path.isfile(chem_map_path):
        chem_map = json.load(open(chem_map_path, encoding="utf-8"))
        samples = build_compound_samples_chembl(reader, chem_map, hvg)
        for s in samples:
            s["smiles"] = chem_map[s["name"]]["smiles"]
        lines = sorted({s["cell_line_idx"] for s in samples})
        print(f"    [chembl] 化合物样本 {len(samples)} 个 / {len(set(s['name'] for s in samples))} 药"
              f" / 细胞系 {lines}", flush=True)
    else:
        samples = build_samples(reader, local, kind="compound", hvg=hvg)
        for s in samples:
            s["smiles"] = local["drugs"][s["local_idx"]].get("smiles", "")
        print(f"    [name-match] 化合物样本 {len(samples)} 个", flush=True)
    if not samples:
        print("    !! 无化合物样本", flush=True)
        reader.close()
        return None
    # 补 SMILES（chembl 路径已带；name-match 路径兜底）+ 丢无 SMILES 的
    if not chem_map_path or not os.path.isfile(chem_map_path):
        for s in samples:
            s["smiles"] = local["drugs"][s["local_idx"]].get("smiles", "")
    samples = [s for s in samples if s.get("smiles")]   # 丢掉无 SMILES 的（无法构图）
    cache = CompoundGraphCache()
    ds = CompoundPertDataset(samples, cache)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: {
                            "graph": [x["graph"] for x in b],
                            "cell_line": torch.tensor([x["cell_line"] for x in b], dtype=torch.long),
                            "label": torch.stack([x["label"] for x in b])})

    # 重建 UniPert（载入阶段 A 基因编码器，实现迁移；gene_mode 须与阶段 A 一致）
    unipert = _make_unipert(len(gene_vocab) + 1, gene_vocab=gene_vocab,
                            gene_mode=gene_mode).to(DEVICE)
    unipert.load_state_dict(sa["unipert"])
    effect_a = PerturbationEffectModel.build_stage_a(unipert, hvg_n).to(DEVICE)
    effect_a.load_state_dict(sa["effect"])
    effect_b = PerturbationEffectModel.build_stage_b(effect_a).to(DEVICE)
    print(f"    模型参数量: {effect_b.num_params()/1e3:.1f}K（基因编码器已由阶段 A 迁移）", flush=True)

    enhance = GeneGraphEnhance().to(DEVICE) if with_enhance else None
    gene_graph = None
    if enhance is not None:
        # sciPlex3 无基因扰动，基因图直接复用阶段 A（Norman）已建好的，避免重算/空图
        node_ids, edges = sa.get("gene_graph", (None, None))
        if node_ids:
            gene_graph = (node_ids, edges)
            print(f"    基因图（复用阶段 A）：{len(node_ids)} 节点 / {len(edges)} 边", flush=True)
        else:
            print("    ⚠️ 阶段 A 未提供基因图，跳过 L_enhance", flush=True)

    opt = torch.optim.AdamW(effect_b.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    _train_loop(effect_b, loader, opt, "stageB", epochs, is_gene=False,
                enhance=enhance, gene_graph=gene_graph)
    torch.save({"unipert": unipert.state_dict(), "effect": effect_b.state_dict(),
                "gene_vocab": gene_vocab, "hvg_dim": hvg_n}, ckpt)
    print(f"    阶段 B 权重已存 {ckpt}", flush=True)
    reader.close()
    return ckpt


# ----------------------------------------------------------- UniPert 预训练（对比 + 图自监督）
def _init_unipert_from(unipert, ckpt_path, gene_vocab):
    """把预训练 UniPert 权重迁移进当前模型。

    预训练词表（本地 20 靶基因）与阶段 A 词表（Norman 扰动基因）不同，
    直接整体 load_state_dict 必然 shape 报错。因此：
      ① 固定形状子模块（compound_encoder / cell_line / align）整体载入；
      ② 基因嵌入表按基因名字只拷贝重叠行（其余保持随机）。
    """
    from unipret.data_bridge import norm as _norm
    sd = torch.load(ckpt_path, map_location=DEVICE)
    up = sd.get("unipert", sd)
    pre_vocab = sd.get("gene_vocab", {})
    n_ok = 0
    for sub in ("compound_encoder", "cell_line", "align"):
        sub_sd = {k[len(sub) + 1:]: v for k, v in up.items()
                  if k.startswith(sub + ".")}
        if not sub_sd:
            continue
        try:
            getattr(unipert, sub).load_state_dict(sub_sd)
            n_ok += 1
        except Exception as e:
            print(f"    ⚠️ {sub} 载入跳过：{e}", flush=True)
    # 基因嵌入按名字对齐
    ge = unipert.gene_encoder
    cur = None
    if hasattr(ge, "gene_id_emb"):
        cur = ge.gene_id_emb.weight.data
    elif hasattr(ge, "emb"):
        cur = ge.emb.weight.data
    pre_emb = None
    for k, v in up.items():
        if k.endswith("gene_id_emb.weight"):
            pre_emb = v
            break
    if cur is not None and pre_emb is not None and cur.size(1) == pre_emb.size(1):
        pre_idx = {_norm(k): i for k, i in pre_vocab.items()}
        cnt = 0
        for name, i in gene_vocab.items():
            p = pre_idx.get(_norm(name))
            if p is not None:
                cur[i] = pre_emb[p]
                cnt += 1
        print(f"    ✅ 预训练初始化：{n_ok}/3 子模块载入，基因嵌入按名迁移 {cnt}/{len(gene_vocab)}", flush=True)
    else:
        print(f"    ✅ 预训练初始化：{n_ok}/3 子模块载入（基因嵌入维度不一致，跳过对齐）", flush=True)


def train_pretrain_unipert(epochs=10, ckpt="unipert_pretrain.pt", gene_mode="esm"):
    print(">>> UniPert 预训练（跨域对比 info_nce + 基因图自监督 L_enhance）", flush=True)
    local = load_local_dataset(LOCAL)
    # P3：扩 CPI——本地 218 药 CPI + sciPlex3 136 药 target 映射 CPI
    genes_all = list(local["genes"])
    gene2idx_all = {g: i for i, g in enumerate(genes_all)}
    drugs_all = [{"name": d["name"], "smiles": d.get("smiles", ""),
                  "targets": d.get("targets", [])} for d in local["drugs"]]
    name2di = {d["name"].lower(): i for i, d in enumerate(drugs_all)}
    pairs, meta = [], []
    # 本地 CPI
    for di, d in enumerate(local["drugs"]):
        for g in d.get("targets", []):
            if g in gene2idx_all:
                pairs.append((gene2idx_all[g], di))
                meta.append((g, d["name"], d["smiles"]))
    # sciPlex3 target CPI（P3 扩量）
    chem_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "sciplex3_chems.json")
    added_genes = 0
    if os.path.isfile(chem_path):
        from unipret.target2genes import target_to_genes, build_sciplex3_cpi
        chem_map = json.load(open(chem_path, encoding="utf-8"))
        for name, v in chem_map.items():
            if name.lower() not in name2di:      # sciPlex3 独有的药并入 drug 列表
                name2di[name.lower()] = len(drugs_all)
                drugs_all.append({"name": name, "smiles": v.get("smiles", ""),
                                  "targets": []})
            for g in target_to_genes(v.get("target", "")):
                if g not in gene2idx_all:
                    gene2idx_all[g] = len(genes_all)
                    genes_all.append(g)
                    added_genes += 1
        extra_pairs, extra_meta = build_sciplex3_cpi(chem_map, gene2idx_all, name2di)
        pairs += extra_pairs
        meta += extra_meta
        print(f"    [P3 扩 CPI] sciPlex3 靶点基因 +{added_genes} 个，共 {len(genes_all)} 基因；"
              f"sciPlex3 CPI 对 +{len(extra_pairs)}", flush=True)
    # LINCS 海量 CPI（P4b：3.4 万药自带 target 基因，造几万对药-靶正样本）
    lincs_tsv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "LINCS_small_molecules.tsv")
    lincs_max = int(os.environ.get("LINCS_MAX_DRUGS", "6000"))
    if os.path.isfile(lincs_tsv):
        import csv as _csv
        lincs = {}
        with open(lincs_tsv, encoding="utf-8") as f:
            for row in _csv.DictReader(f, delimiter="\t"):
                sm = (row.get("canonical_smiles") or "").strip()
                tg = (row.get("target") or "").strip()
                if sm and sm != "-" and tg and tg != "-":
                    lincs[row["pert_name"]] = (tg, sm)
        lincs_names = sorted(lincs.keys())
        if len(lincs_names) > lincs_max:
            lincs_names = sorted(np.random.default_rng(0).choice(
                lincs_names, lincs_max, replace=False).tolist())
        n_lincs_pairs = 0
        for nm in lincs_names:
            tg, sm = lincs[nm]
            if nm.lower() not in name2di:
                name2di[nm.lower()] = len(drugs_all)
                drugs_all.append({"name": nm, "smiles": sm, "targets": []})
            for g in re.split(r"[,\s]+", tg):
                g = g.strip()
                if g and g != "-":
                    if g not in gene2idx_all:
                        gene2idx_all[g] = len(genes_all)
                        genes_all.append(g)
                        added_genes += 1
                    pairs.append((gene2idx_all[g], name2di[nm.lower()]))
                    meta.append((g, nm, sm))
                    n_lincs_pairs += 1
        print(f"    [LINCS 扩 CPI] 药 {len(lincs_names)}，靶点基因 +{added_genes} 个"
              f"（共 {len(genes_all)}），LINCS CPI 对 +{n_lincs_pairs}", flush=True)
    local2 = {"genes": genes_all, "gene2idx": gene2idx_all,
              "drugs": drugs_all, "seqs": local["seqs"]}
    print(f"    CPI 正样本对：{len(pairs)}（基因 {len(genes_all)}，药物 {len(drugs_all)}）", flush=True)
    if not pairs:
        print("    !! 无 CPI 对，退出", flush=True)
        return None
    unipert = _make_unipert(len(genes_all), gene_vocab=gene2idx_all,
                            gene_mode=gene_mode).to(DEVICE)
    enhance = GeneGraphEnhance().to(DEVICE)
    # 序列图（本地 ~20 基因；扩词表后其余节点无边，L_enhance 自动跳过）
    seqs = {k: v["seq"] for k, v in local.get("proteins", {}).items() if isinstance(v, dict) and v.get("seq")}
    _, edges = build_sequence_graph(seqs, k=3)
    node_ids = [gene2idx_all[n] for n in seqs if n in gene2idx_all]
    cache = CompoundGraphCache()
    # 并行预构图（ECFP4 指纹；药多时懒解析会拖慢每个 batch）
    from unipret.compound_encoder import smiles_to_ecfp4 as _stg
    from concurrent.futures import ThreadPoolExecutor
    all_smiles = sorted({d["smiles"] for d in drugs_all if d.get("smiles")})
    with ThreadPoolExecutor(max_workers=8) as _ex:
        for _s, _g in zip(all_smiles, _ex.map(_stg, all_smiles)):
            if _g is not None and _g.any():
                cache.cache[_s] = _g
    print(f"    预构图 {len(cache.cache)}/{len(all_smiles)} 个药物", flush=True)
    opt = torch.optim.AdamW(list(unipert.parameters()) + list(enhance.parameters()),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 准备正样本 batch
    from unipret.data_bridge import LocalPairDataset, collate as pair_collate
    ds = LocalPairDataset(local2, pairs, meta)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pair_collate)
    # P5-B：LINCS moa 同机制药对（drug-drug 正样本，增强 MoA 语义）
    moa_groups = []
    if os.path.isfile(lincs_tsv):
        import csv as _csv2
        moa_map = {}
        with open(lincs_tsv, encoding="utf-8") as f:
            for row in _csv2.DictReader(f, delimiter="\t"):
                mo = (row.get("moa") or "").strip()
                sm = (row.get("canonical_smiles") or "").strip()
                if mo and sm and sm != "-":
                    moa_map.setdefault(mo, []).append(sm)
        for mo, sms in moa_map.items():
            uniq = sorted(set(sms))
            if len(uniq) >= 2:
                moa_groups.append(uniq)
        print(f"    [moa] {len(moa_groups)} 个机制组可造同机制药对", flush=True)
    for ep in range(epochs):
        unipert.train(); enhance.train()
        run = 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            gids = batch["gene_ids"].to(DEVICE)
            cl = batch["cell_lines"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                g, c = unipert(batch["gene_seqs"], gids, batch["compound_graphs"], cl)
                loss = (info_nce(g, c) * W_ALIGN) / GRAD_ACCUM
                # P5-B：同机制药对对比（每 4 步采一批，增强 MoA 聚类）
                if moa_groups and step % 4 == 0:
                    import random as _rnd
                    grp = _rnd.choice(moa_groups)
                    picks = _rnd.sample(grp, min(16, len(grp)))
                    gs = [cache.get(s) for s in picks]
                    gs = [g_ for g_ in gs if g_ is not None]
                    if len(gs) >= 6:
                        half = len(gs) // 2
                        clz = torch.zeros(half, dtype=torch.long, device=DEVICE)
                        za = unipert.encode_compound(gs[:half], clz)
                        zb = unipert.encode_compound(gs[half:2 * half], clz)
                        # diag 对 = 同机制组内两药（弱正）；跨对 = 不同机制（负）→ 同机制药嵌入互相靠近
                        loss = loss + (W_MOA * info_nce(za, zb)) / GRAD_ACCUM
                emb = unipert.gene_id_embedding()
                loss = loss + (W_ENHANCE * enhance(emb, edges, maskable=node_ids)) / GRAD_ACCUM
            scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
            scaler.scale(loss).backward()
            run += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        print(f"  [pretrain] epoch {ep+1}/{epochs}  loss={run/len(loader):.4f}", flush=True)
    torch.save({"unipert": unipert.state_dict(), "gene_vocab": gene2idx_all,
                "hvg_dim": None}, ckpt)
    print(f"    UniPert 预训练权重已存 {ckpt}", flush=True)
    return ckpt


# ----------------------------------------------------------- demo（Norman 小子集快速验证）
def demo(epochs=4, max_genes=30):
    print(">>> DEMO：阶段 A 管线快速验证（Norman 小子集，3050Ti）", flush=True)
    reader = PerturbationReader(NORMAN, backed=True)
    reader.preload(need_gb=99)  # 保守跳过整块预加载，走磁盘连续读
    hvg = select_hvg(reader, n=2000)
    samples, gene_vocab = build_gene_samples_for_pretrain(
        reader, hvg, max_cells=800, max_genes=max_genes)
    local = load_local_dataset(LOCAL)
    name_to_seq = {k.upper(): v["seq"] for k, v in local.get("proteins", {}).items()
                   if isinstance(v, dict) and v.get("seq")}
    samples = _attach_seqs(samples, name_to_seq)
    print(f"    基因样本 {len(samples)} 个，基因词典 {len(gene_vocab)} 个", flush=True)
    unipert = UniPert(num_genes=len(gene_vocab) + 1).to(DEVICE)
    effect = PerturbationEffectModel.build_stage_a(unipert, 2000).to(DEVICE)
    enhance = GeneGraphEnhance().to(DEVICE)
    node_ids, edges = build_gene_response_graph(reader, hvg, max_genes=max_genes, k=6)
    gene_graph = (node_ids, edges)
    print(f"    基因图：{len(node_ids)} 节点 / {len(edges)} 边", flush=True)
    ds = GenePertDataset(samples)
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_gene)
    opt = torch.optim.AdamW(effect.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    _train_loop(effect, loader, opt, "demo", epochs, is_gene=True,
                enhance=enhance, gene_graph=gene_graph)
    print(">>> DEMO 完成 ✅ UniPert统一嵌入→G2CP表型头 接线在 3050Ti 上可训练", flush=True)
    reader.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--stage", choices=["a", "b", "pretrain"])
    ap.add_argument("--data", default="norman", choices=["norman", "sciplex3"])
    ap.add_argument("--load", default=None, help="阶段 B 用的阶段 A 权重")
    ap.add_argument("--unipert", default=None, help="阶段 A 初始化用的 UniPert 预训练权重")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_genes", type=int, default=None)
    ap.add_argument("--gene-mode", default="esm", choices=["esm", "hybrid", "embedding"],
                    help="基因编码器模式（P1 默认 esm=ESM2-8M；无缓存自动回退 hybrid）")
    args = ap.parse_args()

    if args.demo:
        demo(epochs=args.epochs or 4, max_genes=args.max_genes or 30)
    elif args.stage == "a":
        dp = NORMAN if args.data == "norman" else SCIPLEX3
        train_stage_a(dp, epochs=args.epochs, max_genes=args.max_genes,
                      unipert_ckpt=args.unipert, gene_mode=args.gene_mode)
    elif args.stage == "b":
        train_stage_b(args.load or "stageA.pt", epochs=args.epochs,
                      gene_mode=args.gene_mode)
    elif args.stage == "pretrain":
        train_pretrain_unipert(epochs=args.epochs, gene_mode=args.gene_mode)
    else:
        ap.print_help()
