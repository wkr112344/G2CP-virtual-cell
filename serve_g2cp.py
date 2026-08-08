# -*- coding: utf-8 -*-
"""论文级模型 API 服务（端口 8766）—— 双模型路由版（融合模型 = V5 SMD × V10 PCC）。

路由策略（"既要又要"）：
- 5 系（HT29/A375/A549/MCF7/PC3）→ 用 V10（PCC 0.463 高精度表型头，5系专属）
- 其余系 / 新药 → 用主模型 g2cp_full_cpi_v7.pt（162系全覆盖）
- /health 返回两个模型信息；/predict /gene /similar /cascade 自动按 cell 路由

提供 /gui/<fn> 前端静态服务、/cells /drugs /genes 词表接口。
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory, make_response, redirect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.join(BASE, "data", "gui")
CACHE_FUSION = os.path.join(BASE, "data", "g2cp_cache_fullgene")
from train_g2cp_contrast import G2CPNet
from unipret.compound_encoder import smiles_to_ecfp4, ECFP4_BITS
from rdkit import Chem


class CorrectG2CPNet(G2CPNet):
    """正确语义: kind=0 基因 → gene_emb; kind=1 药物 → cp_lin(指纹)。
    (原版 enc 语义反了: 药物走 gene_emb 且被 clamp, 3.2 万药大部分共享嵌入)"""

    def enc(self, kind, key):
        k0 = kind.unsqueeze(1)
        z = self.gene_emb(torch.clamp(key, 0, self.gene_emb.num_embeddings - 1)) * (1 - k0) + \
            self.cp_lin(self._fps[torch.clamp(key, 0, self._fps.shape[0] - 1)]) * k0
        return F.normalize(z, dim=1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIVE = ["HT29", "A375", "A549", "MCF7", "PC3"]
CKPT_FUSION = os.path.join(BASE, "g2cp_full_cpi_v7.pt")
CKPT_V10 = os.path.join(BASE, "g2cp_v10.pt")


def load_model(ckpt_path, cache_dir):
    print(f">>> 加载模型 {os.path.basename(ckpt_path)} (缓存 {os.path.basename(cache_dir)})...", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gene_vocab = [str(x) for x in ck["gene_vocab"]]
    drug_vocab = [str(x) for x in ck["drug_vocab"]]
    hvg = list(ck["hvg"])
    cl_names = [str(x) for x in ck["cl_names"]]
    emb = ck["net"]["head.0.weight"].shape[0] - 32
    headw = ck["net"]["head.1.weight"].shape[0]
    net_cls = CorrectG2CPNet if ck.get("enc_semantics") == "correct" else G2CPNet
    net = net_cls(len(gene_vocab), ECFP4_BITS, emb, len(cl_names), len(hvg), headw)
    # ---- 兼容旧版 Linear cp_lin 权重: cp_lin.weight -> cp_lin.main.* (res 保持 0) ----
    sd = dict(ck["net"])
    if "cp_lin.weight" in sd and "cp_lin.main.weight" not in sd:
        w, b = sd.pop("cp_lin.weight"), sd.pop("cp_lin.bias")
        sd["cp_lin.main.weight"] = w
        sd["cp_lin.main.bias"] = b
        # res 分支保持 0 初始化(等价于旧 Linear), 显式写入当前初始化值
        net.cp_lin.res[0].weight.data.normal_()
        net.cp_lin.res[0].bias.data.zero_()
        net.cp_lin.res[2].weight.data.zero_()
        net.cp_lin.res[2].bias.data.zero_()
        print(f"    [兼容] 旧 Linear cp_lin -> CPEncoder.main (res=0)", flush=True)
    net.load_state_dict(sd, strict=False)
    net.cls_out = 256
    net.to(DEVICE).eval()
    fps = np.load(os.path.join(cache_dir, "drug_fps.npy"))
    net._fps = torch.from_numpy(fps).to(DEVICE)
    print(f"    就绪：药 {len(drug_vocab)} / 基因 {len(gene_vocab)} / 细胞系 {len(cl_names)} / HVG {len(hvg)}", flush=True)
    return net, gene_vocab, drug_vocab, hvg, cl_names, fps


# ---------- 加载两个模型 ----------
fusion_net, f_gv, f_dv, f_hvg, f_cl, f_fps = load_model(CKPT_FUSION, CACHE_FUSION)
f_dv_set = set(f_dv)
f_gv_set = set(f_gv)
f_dv_idx = {d: i for i, d in enumerate(f_dv)}
f_cl_idx = {c: i for i, c in enumerate(f_cl)}

v10_net, v_gv, v_dv, v_hvg, v_cl, v_fps = load_model(CKPT_V10, os.path.join(BASE, "data", "g2cp_cache_5cell"))
v_dv_set = set(v_dv)
v_gv_set = set(v_gv)
v_dv_idx = {d: i for i, d in enumerate(v_dv)}
v_cl_idx = {c: i for i, c in enumerate(v_cl)}

print(f"[OK] 双模型就绪：V10(5系) + FUSION(162系)", flush=True)


def find_similar_proteins(emb):
    """在词表蛋白中找最相似的"""
    embs = fusion_net.gene_emb.weight.cpu()
    sim = F.cosine_similarity(emb.cpu(), embs, dim=1)
    topk = sim.topk(5)
    return [{"g": f_gv[i], "s": float(topk.values[j])} for j, i in enumerate(topk.indices)]


# 药物显示名（pool obs cmap_name）
DRUG_NAMES = {}
try:
    import anndata as ad
    a = ad.read_h5ad(os.path.join(BASE, "data", "g2cp", "data", "LINCS", "pool", "pool_gene_chem_ctrl_adata.h5ad"), backed="r")
    sub = a.obs[a.obs["pert_type"] == "trt_cp"][["pert_id", "cmap_name"]].drop_duplicates()
    DRUG_NAMES = {str(r["pert_id"]): str(r["cmap_name"]) for _, r in sub.iterrows()}
    a.file.close()
except Exception as e:
    print("药物名加载失败:", e)


def get_esm_embedding(seq):
    """用预加载的 ESM2-8M 编码蛋白序列 → 512-dim（已投影至药物嵌入空间）"""
    global _esm_model, _esm_batch_converter, _esm_proj
    if _esm_batch_converter is None:
        return None
    _, _, batch_tokens = _esm_batch_converter([("", seq)])
    with torch.no_grad():
        results = _esm_model(batch_tokens, repr_layers=[6])  # CPU inference
        raw = torch.mean(results["representations"][6][0, 1:-1], dim=0, keepdim=True)  # [1,320]
        return _esm_proj(raw).to(DEVICE)  # project and move to GPU for downstream

# 预加载 ESM
_esm_model = None
_esm_batch_converter = None
_esm_proj = None
try:
    import esm
    from esm.pretrained import ESM2

    ckpt_path = os.path.expanduser(r"~\.cache\torch\hub\checkpoints\esm2_t6_8M_UR50D.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ESM checkpoint not found at {ckpt_path}")

    print("[ESM] 加载 ESM2-8M (6层 320维, ~8M参数)...", flush=True)
    _alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    _esm_model = ESM2(num_layers=6, embed_dim=320, attention_heads=20, alphabet=_alphabet, token_dropout=False)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _esm_model.load_state_dict(sd, strict=False)
    _esm_model.eval()  # stay on CPU, avoids competing with main model for 4GB VRAM
    _esm_batch_converter = _alphabet.get_batch_converter()
    _esm_proj = torch.nn.Linear(320, 512, bias=False)  # CPU, will .to(DEVICE) in get_esm_embedding
    print("[OK] ESM2-8M 蛋白编码器已就绪 (CPU)", flush=True)
except Exception as e:
    print("[WARN] ESM2-8M 加载失败:", e, flush=True)


app = Flask(__name__)
NAME2ID = {}
for _pid, _nm in DRUG_NAMES.items():
    if _nm and str(_nm) != "nan":
        NAME2ID[str(_nm).strip().lower()] = _pid

# 药效片段规则库（未知分子 → 化学骨架类别 → 真实同类参照药）
FRAG_RULES = [
    (Chem.MolFromSmarts("[cX3][NX3][SX4](=O)(=O)[cX3]"), "二芳基磺酰胺类（Ar-SO2-NH-Ar'）",
     "磺酰胺 N 连接第二个芳环——COX-2 选择性抑制剂常用骨架（如尼美舒利）",
     ["nimesulide", "sulfaphenazole"]),
    (Chem.MolFromSmarts("[NX3]C(=O)[NX3][SX4](=O)(=O)[cX3]"), "磺酰脲类",
     "磺酰脲结构（对甲苯磺酰脲，降糖药）",
     ["tolbutamide"]),
    (Chem.MolFromSmarts("c[SX4](=O)(=O)[NX3]"), "芳环磺酰胺类",
     "芳环直接连磺酰胺（含昔布类 SO2NH2、碳酸酐酶抑制剂）",
     ["celecoxib", "acetazolamide", "valdecoxib"]),
    (Chem.MolFromSmarts("OC(=O)CC1=CC=CC=C1"), "芳基乙酸类 NSAID",
     "芳基乙酸骨架（双氯芬酸/吲哚美辛类）",
     ["diclofenac"]),
]


def frag_ref(mol):
    for _sm, _cls, _note, _drugs in FRAG_RULES:
        try:
            if mol.HasSubstructMatch(_sm):
                return {"cls": _cls, "note": _note, "drugs": _drugs, "top": _drugs[0]}
        except Exception:
            continue
    return None


@app.route("/")
def root():
    return redirect("/gui/workbench.html")


@app.route("/gui/<path:fn>")
def gui(fn):
    resp = make_response(send_from_directory(GUI_DIR, fn))
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "models": {
            "fusion": {"ckpt": os.path.basename(CKPT_FUSION), "drugs": len(f_dv), "genes": len(f_gv),
                       "cells": len(f_cl), "hvg": len(f_hvg), "note": "162系 ESM蛋白语义版 | 基因扰动PCC 0.442 | 新药PCC 0.305"},
            "v10": {"ckpt": os.path.basename(CKPT_V10), "drugs": len(v_dv), "genes": len(v_gv),
                    "cells": len(v_cl), "hvg": len(v_hvg), "note": "5系专属，PCC 0.463 高精度"}
        },
        "route": "5系→V10 / 其余→FUSION"
    })


@app.route('/params')
def params():
    return jsonify({'drugs': len(f_dv), 'genes': len(f_gv)})


@app.route("/cells")
def cells():
    return jsonify({"cells": f_cl, "n": len(f_cl)})


@app.route("/drugs")
def drugs():
    seen, out = set(), []
    for d in f_dv:
        if d in seen:
            continue
        seen.add(d)
        out.append({"id": d, "name": DRUG_NAMES.get(d, d)})
        if len(out) >= 2000:
            break
    return jsonify({"drugs": out})


@app.route("/genes")
def genes():
    # 融合模型的基因词表（真实基因符号, 过滤 IMM_* 虚拟扰动）
    real = [g for g in f_gv if not g.startswith("IMM_")]
    return jsonify({"genes": real, "n": len(real)})


def resolve_drug(value):
    """返回 (drug_id, fp_np) 或 None"""
    vid = f_dv_idx.get(value)
    if vid is not None:
        return value, f_fps[vid]
    nm = value.strip().lower()
    pid = NAME2ID.get(nm)
    if pid and pid in f_dv_idx:
        return pid, f_fps[f_dv_idx[pid]]
    fp = smiles_to_ecfp4(value)
    if fp is not None and fp.any():
        return value, np.asarray(fp, dtype=np.float32)
    return None


# 脂肪分化程序面板（医学研究者视角：关注成脂程序标志基因方向）
FAT_PANEL_GENES = [
    "PPARG", "FABP4", "CEBPA", "CEBPB", "CD36", "LPL", "ADIPOQ", "SREBF1",
    "DGAT1", "ACSL1", "PLIN1", "PLIN2", "LEP", "PPARGC1A", "GPD1", "SLC2A4",
    "ACSL3", "FASN", "SCD", "ELOVL6", "LIPE", "PNPLA2", "CIDEC", "NR1H3",
]


def fat_panel(out, hvg):
    """从模型输出提取脂肪分化程序基因的方向（值 + 排名）"""
    order = np.argsort(-out)
    panel = []
    hvg_set = set(hvg)
    for g in FAT_PANEL_GENES:
        if g in hvg_set:
            idx = hvg.index(g)
            if idx >= len(out):
                continue
            rank = int(np.where(order == idx)[0][0])
            panel.append({"g": g, "v": round(float(out[idx]), 3),
                          "rank": rank + 1, "dir": "up" if out[idx] > 0 else "down"})
    return panel


def embed(net, kind_key, key_t, fp):
    """嵌入公式（正确语义）: kind=0 基因 → gene_emb(key); kind=1 药物 → cp_lin(指纹)"""
    if kind_key == 0:
        k = torch.tensor([key_t], device=DEVICE).long()
        z = net.gene_emb(torch.clamp(k, 0, net.gene_emb.num_embeddings - 1))
    else:
        z = net.cp_lin(torch.from_numpy(fp).float().unsqueeze(0).to(DEVICE))
    return F.normalize(z, dim=1)


def run_predict(net, cl_names, cl_idx, kind, key_t, fp, cell_name):
    if cell_name not in cl_idx:
        return None
    ci = cl_idx[cell_name]
    with torch.no_grad():
        z = embed(net, kind, key_t, fp)
        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
    order = np.argsort(-out)
    up = [{"g": cl_names and "" or "", "v": 0.0}]  # placeholder
    up = [{"g": h, "v": float(out[i])} for i, h in zip(order[:10], [net._dummy if hasattr(net, "_dummy") else ""])]
    return out, order


@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json(force=True)
    value = (d.get("drug") or "").strip()
    cell_name = (d.get("cell_name") or "").strip()
    cell = int(d.get("cell", -1))
    if not cell_name and cell >= 0:
        cell_name = f_cl[cell] if cell < len(f_cl) else ""
    if not cell_name:
        return jsonify({"error": "请提供 cell_name 或 cell 索引"})
    resolved = resolve_drug(value)
    if resolved is None:
        return jsonify({"error": "SMILES 无法解析"})
    drug_id, fp = resolved

    # 路由：5系 → V10；否则 → FUSION
    if cell_name in v_cl_idx and drug_id in v_dv_set:
        net, cl_names, cl_idx, dv, dv_set = v10_net, v_cl, v_cl_idx, v_dv, v_dv_set
        model_tag = "v10"
    else:
        net, cl_names, cl_idx, dv, dv_set = fusion_net, f_cl, f_cl_idx, f_dv, f_dv_set
        model_tag = "fusion"
    if cell_name not in cl_idx:
        return jsonify({"error": f"细胞系 {cell_name} 不在模型词表"})

    kind = 0 if drug_id in f_gv_set else 1  # 基因名走基因分支, 其余(含 SMILES 新药)走药物指纹
    if kind == 0:
        key_t = f_gv.index(drug_id) if drug_id in f_gv else -1
        if key_t < 0:
            return jsonify({"error": f"{drug_id} 不在基因词表"})
    else:
        # 药物走 cp_lin(指纹): 已知药用词表索引, 新药(SMILES)直接指纹编码, 无需词表
        key_t = dv.index(drug_id) if drug_id in dv_set else -1

    ci = cl_idx[cell_name]
    with torch.no_grad():
        z = embed(net, kind, key_t, fp)
        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
    order = np.argsort(-out)
    hvg = f_hvg if model_tag == "fusion" else v_hvg
    up = [{"g": hvg[i], "v": float(out[i])} for i in order[:10] if not hvg[i].startswith("Gene_")]
    dn = [{"g": hvg[i], "v": float(out[i])} for i in order[-10:] if not hvg[i].startswith("Gene_")]
    # 统一空间检索靶点：药物嵌入与基因嵌入的余弦相似度
    z_cpu = z.cpu()
    gene_embs = net.gene_emb.weight.cpu()  # 4994 x 512
    sim = torch.nn.functional.cosine_similarity(z_cpu, gene_embs, dim=1)
    topk = sim.topk(6)
    targets = [{"g": f_gv[i], "s": float(topk.values[j])} for j, i in enumerate(topk.indices)]
    ref = None
    if drug_id not in f_dv_set and drug_id not in f_gv_set:
        mol = Chem.MolFromSmiles(value)
        if mol is not None:
            ref = frag_ref(mol)
    return jsonify({"drug": drug_id, "name": DRUG_NAMES.get(drug_id, drug_id),
                    "cell": cell_name, "model": model_tag, "up": up, "down": dn, "ref": ref,
                    "targets": targets, "fat_panel": fat_panel(out, hvg)})


@app.route("/custom_drug", methods=["POST"])
def custom_drug():
    """SMILES 新药：实时算 ECFP4 指纹 → 嵌入 → 预测表达变化。"""
    d = request.get_json(force=True)
    smiles = (d.get("smiles") or "").strip()
    cell_name = (d.get("cell_name") or "").strip()
    if not smiles:
        return jsonify({"error": "请提供 SMILES"})
    if not cell_name:
        return jsonify({"error": "请提供 cell_name"})
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "SMILES 无法解析"})
    fp = smiles_to_ecfp4(smiles)
    if fp is None or not fp.any():
        return jsonify({"error": "SMILES 无法解析"})
    if cell_name not in f_cl_idx:
        return jsonify({"error": f"细胞系 {cell_name} 不在模型词表"})
    key_t = -1
    ci = f_cl_idx[cell_name]
    with torch.no_grad():
        z = embed(fusion_net, 1, key_t, np.asarray(fp, dtype=np.float32))
        out = fusion_net.head(torch.cat([z, fusion_net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
    order = np.argsort(-out)
    hvg = f_hvg
    up = [{"g": hvg[i], "v": float(out[i])} for i in order[:10] if not hvg[i].startswith("Gene_")]
    dn = [{"g": hvg[i], "v": float(out[i])} for i in order[-10:] if not hvg[i].startswith("Gene_")]
    z_cpu = z.cpu()
    gene_embs = fusion_net.gene_emb.weight.cpu()
    sim = torch.nn.functional.cosine_similarity(z_cpu, gene_embs, dim=1)
    topk = sim.topk(6)
    targets = [{"g": f_gv[i], "s": float(topk.values[j])} for j, i in enumerate(topk.indices)]
    return jsonify({
        "smiles": smiles, "name": "Custom Molecule",
        "cell": cell_name, "model": "fusion", "up": up, "down": dn,
        "targets": targets, "fat_panel": fat_panel(out, hvg),
        "ref": frag_ref(mol)
    })


@app.route("/gene", methods=["POST"])
def gene():
    """基因敲除预测：融合模型 4,994 基因 → 978 HVG 表达变化。"""
    d = request.get_json(force=True)
    gname = (d.get("gene") or "").strip().upper()
    cell_name = (d.get("cell_name") or "").strip()
    cell = int(d.get("cell", 0))
    if not cell_name:
        cell_name = f_cl[cell] if cell < len(f_cl) else ""
    if gname not in f_gv_set:
        return jsonify({"error": f"基因 {gname} 不在可扰动列表（{sum(1 for g in f_gv if not g.startswith('IMM_'))} 基因）"})
    if gname.startswith("IMM_"):
        return jsonify({"error": f"{gname} 是免疫实验内部条件, 不能作为基因敲除"})
    key_t = f_gv.index(gname)
    if cell_name in v_cl_idx and gname in v_gv_set:
        net, cl_names, cl_idx = v10_net, v_cl, v_cl_idx
        model_tag = "v10"
    else:
        net, cl_names, cl_idx = fusion_net, f_cl, f_cl_idx
        model_tag = "fusion"
    if cell_name not in cl_idx:
        return jsonify({"error": f"细胞系 {cell_name} 不在词表"})
    ci = cl_idx[cell_name]
    with torch.no_grad():
        z = net.gene_emb(torch.tensor([key_t], device=DEVICE).long())
        z = F.normalize(z, dim=1)
        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
    order = np.argsort(-out)
    hvg = f_hvg if model_tag == "fusion" else v_hvg
    up = [{"g": hvg[i], "v": float(out[i])} for i in order[:10] if not hvg[i].startswith("Gene_")]
    dn = [{"g": hvg[i], "v": float(out[i])} for i in order[-10:] if not hvg[i].startswith("Gene_")]
    panel = fat_panel(out, hvg)
    return jsonify({"gene": gname, "cell": cell_name, "perturbation": "knockout", "model": model_tag,
                    "note": "CRISPR 敲除（基因功能丧失）的转录组预测", "up": up, "down": dn, "fat_panel": panel})


@app.route("/similar", methods=["POST"])
def similar():
    """相似药检索：模型嵌入 + ECFP4 Tanimoto 混合（主模型 162系 词表）"""
    d = request.get_json(force=True)
    value = (d.get("drug") or "").strip()
    beta = float(d.get("beta", 0.5))
    resolved = resolve_drug(value)
    if resolved is None:
        return jsonify({"error": "SMILES 无法解析"})
    drug_id, fpi = resolved
    net, dv, dv_set, fps = fusion_net, f_dv, f_dv_set, f_fps
    with torch.no_grad():
        zi = F.normalize(net.cp_lin(torch.from_numpy(fpi).float().unsqueeze(0).to(DEVICE)), dim=1).cpu().numpy()[0]
        Z = net.cp_lin(net._fps).cpu().numpy()
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    cos = Zn @ zi
    cnt = fps.sum(1)
    inter = fps @ fpi
    den = cnt + fpi.sum() - inter
    tani = np.divide(inter, den, out=np.zeros_like(inter), where=den != 0)
    score = beta * cos + (1 - beta) * tani
    order = np.argsort(-score)[:12]
    hits = [{"id": dv[i], "name": DRUG_NAMES.get(dv[i], dv[i]), "score": round(float(score[i]), 3)} for i in order]
    return jsonify({"query": drug_id, "hits": hits})


@app.route("/cascade", methods=["POST"])
def cascade():
    """扰动跟踪链：药物 → 预测基因(↑/↓) → 表型"""
    d = request.get_json(force=True)
    value = (d.get("drug") or d.get("value") or "").strip()
    cell_name = (d.get("cell_name") or "").strip()
    cell = int(d.get("cell", 0))
    if not cell_name:
        cell_name = f_cl[cell] if cell < len(f_cl) else ""
    resolved = resolve_drug(value)
    if resolved is None:
        return jsonify({"error": "SMILES 无法解析"})
    drug_id, fp = resolved
    net, cl_names, cl_idx, dv, dv_set = fusion_net, f_cl, f_cl_idx, f_dv, f_dv_set
    if cell_name not in cl_idx:
        return jsonify({"error": f"细胞系 {cell_name} 不在词表"})
    ci = cl_idx[cell_name]
    kind = 0 if drug_id in f_gv_set else 1
    key_t = (f_gv.index(drug_id) if kind == 0 else f_dv.index(drug_id) if drug_id in f_dv_set else -1)
    with torch.no_grad():
        z = embed(net, kind, key_t, fp)
        out = net.head(torch.cat([z, net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
    order = np.argsort(-out)
    ups = [f_hvg[i] for i in order[:6] if not f_hvg[i].startswith("Gene_")]
    dns = [f_hvg[i] for i in order[-6:] if not f_hvg[i].startswith("Gene_")]
    chain = [
        {"stage": 0, "kind": "start", "name": DRUG_NAMES.get(drug_id, drug_id), "mech": "药物扰动起点", "dir": None, "next": ups[0] if ups else ""},
    ]
    for i, g in enumerate(ups[:3]):
        chain.append({"stage": i + 1, "kind": "transcript", "name": "↑ " + g,
                      "mech": f"模型预测表达上调（978 HVG 中排名第 {i+1}）", "dir": "up", "v": float(out[order[i]]), "next": ""})
    for i, g in enumerate(dns[:2]):
        chain.append({"stage": 5 + i, "kind": "transcript", "name": "↓ " + g,
                      "mech": "模型预测表达下调", "dir": "dn", "v": float(out[order[-(i+1)]]), "next": ""})
    for i in range(len(chain) - 1):
        chain[i]["next"] = chain[i + 1]["name"]
    chain[-1]["next"] = "表型终点"
    chain.append({"stage": 99, "kind": "phenotype",
                  "name": f"细胞表型改变（{', '.join(ups[:3]) if ups else ''}↑ / {', '.join(dns[:2]) if dns else ''}↓）",
                  "mech": f"由模型预测的 {len(ups)} 个显著上调基因（如 {', '.join(ups[:3])}）和 {len(dns)} 个显著下调基因（如 {', '.join(dns[:2])}）推断：细胞转录状态向扰动方向偏移。点击链上任一节点可查看该级微观细节。",
                  "dir": None, "next": None})
    return jsonify({"chain": chain, "pathway_name": "转录组重编程（论文级模型）"})


@app.route("/custom_protein", methods=["POST"])
def custom_protein():
    """未知蛋白：基因名或序列 → 预测表达变化"""
    d = request.get_json(force=True)
    seq = (d.get("sequence") or "").strip().upper()
    cell_name = (d.get("cell_name") or "").strip()
    if cell_name not in f_cl_idx:
        return jsonify({"error": "细胞系不在词表"})
    ci = f_cl_idx[cell_name]

    # 尝试 ESM 编码，失败则用基因名查找
    emb = None
    if _esm_batch_converter and len(seq) >= 20:
        try:
            emb = get_esm_embedding(seq)
        except Exception:
            pass
    if emb is None and seq in f_gv_set:
        with torch.no_grad():
            emb = fusion_net.gene_emb(torch.tensor([f_gv.index(seq)], device=DEVICE).long())
    if emb is None:
        return jsonify({"error": "蛋白编码失败：输入基因符号或≥20氨基酸序列"})

    with torch.no_grad():
        z = F.normalize(emb, dim=1)
        out = fusion_net.head(torch.cat([z, fusion_net.cell_emb(torch.tensor([ci], device=DEVICE).long())], dim=1)).cpu().numpy()[0]
    order = np.argsort(-out)
    up = [{"g": f_hvg[i], "v": float(out[i])} for i in order[:10] if not f_hvg[i].startswith("Gene_")]
    dn = [{"g": f_hvg[i], "v": float(out[i])} for i in order[-10:] if not f_hvg[i].startswith("Gene_")]
    sim = find_similar_proteins(emb.cpu()) if emb is not None else []
    return jsonify({"seq_len": len(seq), "cell": cell_name, "up": up, "down": dn, "pathway_name": "蛋白扰动", "similar": sim})


@app.route("/interact", methods=["POST"])
def interact():
    """药物-蛋白或蛋白-蛋白互作：余弦相似度"""
    d = request.get_json(force=True)
    mode = d.get("mode", "drug_protein")
    cell_name = (d.get("cell_name") or "").strip()

    def encode_protein(seq):
        emb = None
        if _esm_batch_converter and seq and len(seq) >= 20:
            try:
                emb = get_esm_embedding(seq)
            except Exception:
                pass
        if emb is None and seq in f_gv_set:
            with torch.no_grad():
                emb = fusion_net.gene_emb(torch.tensor([f_gv.index(seq)], device=DEVICE).long())
        return emb

    with torch.no_grad():
        smiles = (d.get("smiles") or "").strip()
        seq_a = (d.get("sequence") or d.get("seqA") or "").strip().upper()
        seq_b = (d.get("seqB") or "").strip().upper()

        if mode == "drug_protein":
            if not smiles or not seq_a:
                return jsonify({"error": "请提供 SMILES 和蛋白名/序列"})
            fp = smiles_to_ecfp4(smiles)
            if fp is None:
                return jsonify({"error": "SMILES 无法解析"})
            za = F.normalize(fusion_net.cp_lin(torch.from_numpy(np.asarray(fp, dtype=np.float32)).unsqueeze(0).to(DEVICE)), dim=1)
            zb = encode_protein(seq_a)
            if zb is None:
                hint = ""
                if seq_a and seq_a not in f_gv_set:
                    sim = sorted([g for g in f_gv_set if len(seq_a)>=3 and seq_a[:3].upper() in g], key=lambda x: abs(len(x)-len(seq_a)))[:3]
                    hint = f"（{seq_a} 不在 4,994 基因词表中；试试 {', '.join(sim)}）" if sim else f"（{seq_a} 不在 4,994 基因词表中，请用 ≥20 aa 序列）"
                return jsonify({"error": "蛋白编码失败："+hint if hint else "请输入基因符号或 ≥20 aa 氨基酸序列"})
            zb = F.normalize(zb, dim=1)
        else:
            if not seq_a or not seq_b:
                return jsonify({"error": "请提供两条蛋白名/序列"})
            za = encode_protein(seq_a)
            zb = encode_protein(seq_b)
            if za is None or zb is None:
                return jsonify({"error": "蛋白编码失败（需要 ESM 或有效基因名）"})
            za = F.normalize(za, dim=1)
            zb = F.normalize(zb, dim=1)
        sim = float(F.cosine_similarity(za, zb).item())

    return jsonify({"mode": mode, "similarity": round(sim, 4),
                    "verdict": "嵌入空间邻近（仅反映统一表征余弦相似度，不直接等价于物理结合强度）"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    print(f"[OK] 双模型 API 就绪：http://127.0.0.1:{args.port}", flush=True)
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)
