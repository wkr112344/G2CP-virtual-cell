"""
对齐桥（data bridge）—— 把本地 218 药 / 20 基因 接到 sciPlex3 / Norman 真实转录组
=====================================================================================
两个真实数据源（scPerturb 标准 .h5ad 格式）：
  - sciPlex3 ：188 化合物 + ~7500 基因 CRISPR，3 细胞系（K562/Jurkat/293T），双扰动。
  - Norman   ：纯基因 CRISPR KO，单细胞系 K562（本机已完整下载，用于基因侧验证）。

本模块“不训练”，只把原始 h5ad 整理成模型能吃的样本：
  · 基因侧（阶段 A 预训练）：基因序列 → 该基因被 KO 后的“表达变化向量”（真实转录组标签）
  · 化合物侧（阶段 B 微调）：化合物图 → 该药处理后的“表达变化向量”（真实转录组标签）
  · 对比正样本：本地 dataset.json 的 drug.targets 给出 (基因, 化合物) 对（已在 build_positive_pairs）

设计要点（适配 4GB 显存本机）：
  · PerturbationReader 用 anndata backed='r' 只读模式，obs/var 全加载，X 表达矩阵按需切片，
    读 2.3GB sciPlex3 也不会把内存撑爆。
  · 表达标签默认降维到 top-N 高变基因（HVG，默认 2000 维），避免 33694 维标签拖垮训练。
  · 表达变化 = 同细胞系下 (扰动细胞均值 − 对照细胞均值)，即“扰动效应”。

用法见文件底部 selftest()；训练脚本 train.py 稍后接入 build_samples()。
"""
import os
import re
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .compound_encoder import smiles_to_graph
    from .config import CELL_LINE_NAMES, DEVICE
except ImportError:  # 作为脚本直接运行时
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from unipret.compound_encoder import smiles_to_graph
    from unipret.config import CELL_LINE_NAMES, DEVICE

# ----------------------------------------------------------- 1. 本地 dataset.json
def load_local_dataset(path):
    """返回 dict: genes(list), gene2idx, drugs([{name,smiles,targets}]), seqs(gene->蛋白序列)"""
    d = json.load(open(path, encoding="utf-8"))
    genes = d["genes"]
    gene2idx = {g: i for i, g in enumerate(genes)}
    seqs = {g: d["proteins"][g]["seq"] for g in genes if g in d.get("proteins", {})}
    drugs = d["drugs"]
    return {"genes": genes, "gene2idx": gene2idx, "drugs": drugs, "seqs": seqs}


def build_positive_pairs(dataset):
    """对比学习正样本对： drug.targets 里每个 (药, 靶点基因) = 一对。"""
    gene2idx = dataset["gene2idx"]
    pairs, meta = [], []
    for di, drug in enumerate(dataset["drugs"]):
        for g in drug.get("targets", []):
            if g in gene2idx:
                pairs.append((gene2idx[g], di))
                meta.append((g, drug["name"], drug["smiles"]))
    return pairs, meta


class CompoundGraphCache:
    """把 SMILES 解析成 ECFP4 指纹（np [2048]）并缓存，避免每个 batch 重复解析。
    （ECFP4 路线：类名保留兼容旧引用，内部已从分子图切换为指纹。）"""
    def __init__(self):
        self.cache = {}
    def get(self, smiles):
        if smiles not in self.cache:
            from unipret.compound_encoder import smiles_to_ecfp4
            self.cache[smiles] = smiles_to_ecfp4(smiles)
        return self.cache[smiles]


class LocalPairDataset(Dataset):
    """把正样本对整理成 (基因序列, 基因下标, 化合物图, 细胞系下标) 样本。
    基因嵌入模式下 gene_seq 字段不被编码器使用，但保留以便兼容。"""
    def __init__(self, dataset, pairs, meta, cell_line_idx=0):
        self.dataset = dataset
        self.pairs = pairs
        self.meta = meta
        self.seqs = dataset["seqs"]
        self.graph_cache = CompoundGraphCache()
        self.cell_line_idx = cell_line_idx
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        gene_idx, drug_idx = self.pairs[i]
        gene = self.dataset["genes"][gene_idx]
        smiles = self.meta[i][2]
        return {
            "gene_seq": self.seqs.get(gene, ""),
            "gene_id": gene_idx,
            "compound_graph": self.graph_cache.get(smiles),
            "cell_line": self.cell_line_idx,
        }


def collate(batch):
    return {
        "gene_seqs": [b["gene_seq"] for b in batch],
        "gene_ids": torch.tensor([b["gene_id"] for b in batch], dtype=torch.long),
        "compound_graphs": [b["compound_graph"] for b in batch],
        "cell_lines": torch.tensor([b["cell_line"] for b in batch], dtype=torch.long),
    }


# ----------------------------------------------------------- 2. 名字归一化与别名
def norm(s):
    """归一化：去所有非字母数字、转小写。用于模糊匹配。"""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# 常见盐/溶剂/酸后缀（去掉后便于“主名”匹配，如 Sorafenib Tosylate -> sorafenib）
SALT_SUFFIXES = {
    "citrate", "tosylate", "ditosylate", "mesylate", "besylate", "hcl", "2hcl",
    "hydrochloride", "phosphate", "diphosphate", "tartaric", "fumarate", "maleate",
    "succinate", "acetate", "sodium", "potassium", "chloride", "sulfate", "sulphate",
    "nitrate", "bromide", "hydrobromide", "hydrate", "dihydrate", "hemihydrate",
    "monohydrate", "ethyl", "methyl", "ethanol", "water", "lactate", "malate",
    "oxalate", "pamoate", "tartrate", "edta", "ammonium", "freebase", "acid",
    "salt", "base", "trihydrate", "diphosphate",
}


def norm_loose(s):
    """化合物名宽松归一：去括号及内容、去盐/溶剂/酸后缀、去标点空格、小写。
    用于匹配 'Lapatinib (GW-572016) Ditosylate' <-> 'LAPATINIB' 这类命名差异。"""
    s = str(s).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    toks = re.split(r"[\s\-/,+_]+", s)
    toks = [t for t in toks if t and t not in SALT_SUFFIXES]
    return "".join(toks)


# 少数常见化合物别名/不同拼写（按需扩充；离线时优先用本地 SMILES 名，查不到才靠它）
COMPOUND_ALIASES = {
    "tamoxifen": ["tam", "nolvadex"],
    "dexamethasone": ["dex"],
    "dabrafenib": ["gsk2118436"],
}


# ----------------------------------------------------------- 3. h5ad 读取（内存友好）
def _find_col(obs, candidates):
    """在 obs 列里按候选名自动探测真实列名（精确优先，再子串）。"""
    cols = list(obs.columns)
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        for col in cols:
            if c in col.lower():
                return col
    return None


class PerturbationReader:
    """
    只读读取一个 scPerturb 风格 .h5ad，解析扰动注释。
    backed=True 时 X 留在磁盘，切片才读，2.3GB 文件也不爆内存。
    """

    def __init__(self, h5ad_path, backed=True):
        import anndata
        if backed:
            self.ad = anndata.read_h5ad(h5ad_path, backed="r")
        else:
            self.ad = anndata.read_h5ad(h5ad_path)
        self.path = h5ad_path
        self.obs = self.ad.obs
        self.var_names = [str(g) for g in self.ad.var_names]
        self.gene_index = {g: i for i, g in enumerate(self.var_names)}
        self.n_cells = self.ad.n_obs
        self.n_genes = self.ad.n_vars
        # 自动探测关键列（scPerturb 各数据集列名略有差异，统一兜底）
        self.pert_col = _find_col(self.obs, ["perturbation", "pert"])
        self.type_col = _find_col(self.obs, ["perturbation_type", "pert_type"])
        self.line_col = _find_col(self.obs, ["cell_line", "cell-line", "celltype"])
        self.dose_col = _find_col(self.obs, ["dose", "dosage", "concentration"])
        self.chembl_col = _find_col(self.obs, ["chembl-id", "chembl_ID", "chembl"])
        self.pert_labels = self._pert_labels()
        self.line_labels = self._line_labels()
        self.chembl_labels = self._chembl_labels()
        self._X_cache = None  # 预加载缓存（preload() 时填充，None 则走磁盘）
        self._hvg_slice = None  # backed 模式：缓存的 hvg 列子矩阵（csr，约 320MB）
        self._hvg_key = None    # 对应的 hvg 指纹，变化时重建缓存

    def _pert_labels(self):
        if self.pert_col is None:
            return [None] * self.n_cells
        import pandas as pd
        out = []
        for v in self.obs[self.pert_col]:
            if pd.isna(v) or str(v) in ("nan", "None", ""):
                out.append(None)
            else:
                out.append(str(v).strip())
        return out

    def _line_labels(self):
        if self.line_col is None:
            return [None] * self.n_cells
        import pandas as pd
        out = []
        for v in self.obs[self.line_col]:
            if pd.isna(v) or str(v) in ("nan", "None", ""):
                out.append(None)
            else:
                out.append(str(v).strip())
        return out

    def _chembl_labels(self):
        """每个细胞对应的 ChEMBL id（P2：化合物 SMILES 直连用）。"""
        if self.chembl_col is None:
            return [None] * self.n_cells
        import pandas as pd
        out = []
        for v in self.obs[self.chembl_col]:
            if pd.isna(v) or str(v) in ("nan", "None", ""):
                out.append(None)
            else:
                out.append(str(v).strip())
        return out

    def labels_unique(self):
        s = set(self.pert_labels)
        s.discard(None)
        return s

    def expression(self, idx):
        """读取某细胞（或细胞列表）的表达向量，返回 np.float32 一维/二维。

        backed 模式下 ad.X 是 h5py 数据集：若 idx 是一串乱序行号，
        h5py 会逐行随机寻址（极慢）。这里先排序、再切成连续段批量读，
        把随机寻址降到最少，提速几十倍。若已 preload() 则直接走内存缓存。
        """
        if self._X_cache is not None:
            x = self._X_cache[idx]
            if hasattr(x, "todense"):
                x = x.todense()
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return x
        if isinstance(idx, (int, np.integer)):
            x = self.ad.X[int(idx)]
            if hasattr(x, "todense"):
                x = x.todense()
            return np.asarray(x, dtype=np.float32)
        idx = np.asarray(idx)
        if idx.ndim == 0:
            idx = idx.reshape(1)
        if idx.size == 0:
            return np.empty((0, self.n_genes), dtype=np.float32)
        order = np.argsort(idx)
        sorted_idx = idx[order]
        # 把升序 idx 切成连续段 [a, b)
        segs = []
        start = 0
        for i in range(1, sorted_idx.size):
            if sorted_idx[i] != sorted_idx[i - 1] + 1:
                segs.append((int(sorted_idx[start]), int(sorted_idx[i - 1]) + 1))
                start = i
        segs.append((int(sorted_idx[start]), int(sorted_idx[-1]) + 1))
        parts = []
        for a, b in segs:
            chunk = self.ad.X[a:b]            # 连续切片，h5py 高效
            if hasattr(chunk, "todense"):
                chunk = chunk.todense()
            parts.append(np.asarray(chunk, dtype=np.float32))
        x = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        # 还原到原始 idx 顺序
        x = x[np.argsort(order)]
        return x

    def load_full_sparse(self, chunk=20000):
        """把 backed 稀疏矩阵一次性物化为内存中的 scipy CSC（仅一次慢读），
        之后所有切片从内存走，秒级。用 ctypes 探测空闲内存，不够则跳过回退磁盘。
        返回是否成功。
        """
        import ctypes, time
        import scipy.sparse as sp

        def free_ram_gb():
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MS(); st.dwLength = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullAvailPhys / (1024 ** 3)

        try:
            samp = self.ad.X[0:10000]
            samp = samp.tocsr() if hasattr(samp, "tocsr") else sp.csr_matrix(samp)
            fill = samp.nnz / (samp.shape[0] * samp.shape[1])
        except Exception:
            fill = 0.12
        est_gb = self.n_cells * self.n_genes * fill * 8 / 1e9 * 1.1
        avail = free_ram_gb()
        if avail is not None and avail < est_gb:
            print("  [preload] 空闲 %.1fGB < 估算 %.1fGB，跳过整块加载，回退磁盘"
                  % (avail, est_gb))
            return False
        print("  [preload] 整块加载稀疏矩阵到内存 (估算 %.1fGB)..." % est_gb)
        t0 = time.time()
        parts = []
        n = self.n_cells
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            x = self.ad.X[s:e]
            x = x.tocsr() if hasattr(x, "tocsr") else sp.csr_matrix(x)
            parts.append(x)
        X = sp.vstack(parts).tocsc()
        self._X_cache = X
        print("  [preload] 完成 %.1fs 稀疏矩阵 %s 占用 ~%.2fGB"
              % (time.time() - t0, X.shape,
                 (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e9))
        return True

    def X_hvg(self, idx, hvg):
        """读取 idx 细胞、降维到 hvg 列。

        backed 模式走「先按列切（CSC 列切廉价）并缓存 hvg 子矩阵，
        再按行取」——避免对 CSC 直接做 ad.X[idx, hvg]（会先按行切、
        物化全 11 万列）导致 OOM（sciPlex3 实测触发）。
        """
        idx = np.asarray(idx)
        if self._X_cache is not None:
            sub = self._X_cache[:, hvg][idx]      # 内存 CSC：列切 + 行取
            if hasattr(sub, "todense"):
                sub = sub.todense()
            return np.asarray(sub, dtype=np.float32)
        # backed 模式：懒加载并缓存 hvg 列切片（约 320MB，一次性），之后按行取即时
        hvg_key = hvg.tobytes() if isinstance(hvg, np.ndarray) else hvg
        if self._hvg_slice is None or self._hvg_key != hvg_key:
            self._hvg_slice = self.ad.X[:, hvg]   # 列切：CSC 廉价，返回 csr(全细胞 × hvg)
            self._hvg_key = hvg_key
        sub = self._hvg_slice[idx]                # 仅在 hvg 维小矩阵上按行取（即时）
        if hasattr(sub, "toarray"):
            out = sub.toarray()
        elif hasattr(sub, "todense"):
            out = sub.todense()
        else:
            out = np.asarray(sub)
        if np.ndim(out) == 1:
            out = np.asarray(out).reshape(1, -1)
        return np.asarray(out, dtype=np.float32)

    def control_mask(self):
        """标记为对照的细胞（control / non-targeting / scramble / unperturbed）。"""
        ctrl_kw = ["control", "non-targeting", "non_targeting", "nt",
                   "unperturbed", "scramble", "scrambled", "vector"]
        mask = np.zeros(self.n_cells, dtype=bool)
        for i, l in enumerate(self.pert_labels):
            if l is None or any(k in l.lower() for k in ctrl_kw):
                mask[i] = True
        return mask

    def close(self):
        try:
            self.ad.file.close()
        except Exception:
            pass

    def summary(self):
        return (f"cells={self.n_cells} genes={self.n_genes} "
                f"pert_col={self.pert_col} type_col={self.type_col} "
                f"line_col={self.line_col} dose_col={self.dose_col} "
                f"unique_perturbations={len(self.labels_unique())}")


# ----------------------------------------------------------- 4. 名字对齐
def align_genes(local_genes, data_gene_names):
    """本地基因符号 ↔ 数据里出现的基因符号（扰动标注 / var）。返回 {local_idx: data_name}。"""
    data_set = {norm(g): g for g in data_gene_names}
    m = {}
    for i, g in enumerate(local_genes):
        if norm(g) in data_set:
            m[i] = data_set[norm(g)]
    return m


def align_compounds(local_drugs, data_compound_names):
    """本地药名 ↔ 数据化合物标签（token 化 + 宽松归一匹配）。
    返回 {local_drug_idx: data_label}。同一药的不同盐形式会映射到同一数据标签。
    示例：'LAPATINIB' -> 'Lapatinib (GW-572016) Ditosylate'。
    注意：只用“去括号+去盐后主名相等”匹配，不做危险子串包含（避免误匹配）。
    """
    # 数据标签 -> 拆出的 token(norm)；记录 norm->首个原标签
    token_to_label = {}
    for lab in data_compound_names:
        if lab is None:
            continue
        for tok in re.split(r"[+_/]", lab):
            t = norm_loose(tok)
            if len(t) >= 4:
                token_to_label.setdefault(t, lab)
    m = {}
    for i, d in enumerate(local_drugs):
        ln = norm_loose(d.get("name", ""))
        if ln in token_to_label:
            m[i] = token_to_label[ln]
            continue
        # 别名表兜底（保守，仍需主名相等）
        for alias in COMPOUND_ALIASES.get(norm(d.get("name", "")), []):
            al = norm_loose(alias)
            if al in token_to_label:
                m[i] = token_to_label[al]
                break
    return m


def resolve_perturbation(label, gene_norm_set, compound_norm_set):
    """
    判断一个扰动标注属于哪类，返回 (kind, payload)。
      kind: 'control' | 'gene' | 'compound' | 'multi' | 'unknown'
    组合扰动用 '_' 或 '+' 拆分（scPerturb 标准）。
    """
    if label is None:
        return ("control", None)
    l = label.strip()
    if any(k in l.lower() for k in
           ["control", "non-targeting", "non_targeting", "scramble",
            "scrambled", "unperturbed", "vector"]):
        return ("control", None)
    parts = re.split(r"[+_/]", l)
    kinds = []
    for p in parts:
        pn = norm(p)
        if pn in gene_norm_set:
            kinds.append(("gene", pn))
        elif pn in compound_norm_set:
            kinds.append(("compound", pn))
    if not kinds:
        return ("unknown", l)
    if len(kinds) == 1:
        return kinds[0]
    return ("multi", kinds)


# ----------------------------------------------------------- 5. 高变基因降维
def select_hvg(reader, n=2000, chunk=10000, max_cells=None):
    """
    估算每基因表达方差，返回 top-n 高变基因索引。
    优先用整块加载的内存 CSC（indptr 一致、列切片正确、最快）；
    未整块加载（backed 视图列切片有坑）时回退逐块行读 Welford（慢但正确）。
    """
    import scipy.sparse as sp
    cache = getattr(reader, "_X_cache", None)
    if cache is not None and sp.isspmatrix(cache):
        return _select_hvg_csc(cache, n)
    X = reader.ad.X
    if sp.isspmatrix(X):
        return _select_hvg_csc(X, n)
    return _select_hvg_welford(reader, n, chunk, max_cells)


def _select_hvg_csc(X, n):
    """内存 scipy 稀疏矩阵（CSC）上列分块聚合方差。

    关键修正：不再用 X[:, c0:c1] 造子矩阵再做 reduceat——
    scipy CSC 列切返回的子矩阵其 indptr 与 data 可能不一致（indptr 仍指向
    原全量偏移），直接 reduceat 会越界。这里直接基于「完整 CSC 的 indptr」
    按列范围在原 data 上聚合，零副本、indptr 始终一致。
    """
    import scipy.sparse as sp
    if not sp.isspmatrix_csc(X):
        X = X.tocsc()
    n_genes = X.shape[1]
    indptr = X.indptr
    col_sum = np.zeros(n_genes, dtype=np.float64)
    col_sumsq = np.zeros(n_genes, dtype=np.float64)
    col_cnt = np.zeros(n_genes, dtype=np.float64)
    group = 2000
    ngroups = (n_genes + group - 1) // group
    for c0 in range(0, n_genes, group):
        c1 = min(c0 + group, n_genes)
        start = indptr[c0]
        end = indptr[c1]
        bounds = indptr[c0:c1 + 1] - start          # 本组每列在原 data 段内的起止（一致）
        seg = X.data[start:end].astype(np.float64)   # 仅本组列的数据段（不造全子矩阵）
        seg_p = np.concatenate([seg, [0.0]])         # 尾部补哨兵 0：空列归段(起点=len)时安全
        col_sum[c0:c1] = np.add.reduceat(seg_p, bounds[:-1])
        col_sumsq[c0:c1] = np.add.reduceat(seg_p * seg_p, bounds[:-1])
        col_cnt[c0:c1] = np.diff(bounds)
        print("  [select_hvg] 列组 %d/%d" % (c0 // group + 1, ngroups), flush=True)
        del seg, seg_p, bounds
        import gc; gc.collect()
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_i = np.where(col_cnt > 0, col_sum / col_cnt, 0.0)
        var = np.where(col_cnt > 1, col_sumsq / col_cnt - mean_i ** 2, 0.0)
    var = np.nan_to_num(var)
    top = np.argsort(var)[::-1][:n]
    return np.sort(top)


def _select_hvg_welford(reader, n, chunk, max_cells):
    """backed 未整块加载兜底：分块读稀疏块，用 SparseDataset 原生列聚合算方差
    （避开手动 reduceat 在 backed 视图上的 indptr 越界）。"""
    n_genes = reader.n_genes
    col_sum = np.zeros(n_genes, dtype=np.float64)
    col_sumsq = np.zeros(n_genes, dtype=np.float64)
    col_cnt = np.zeros(n_genes, dtype=np.float64)
    n_total = reader.n_cells
    n_chunks = (n_total + chunk - 1) // chunk
    ci = 0
    for s in range(0, n_total, chunk):
        e = min(s + chunk, n_total)
        blk = reader.ad.X[s:e]   # backed SparseDataset 视图
        s_blk = np.asarray(blk.sum(axis=0)).ravel().astype(np.float64)
        blk2 = blk.multiply(blk)
        s2_blk = np.asarray(blk2.sum(axis=0)).ravel().astype(np.float64)
        cnt_blk = np.asarray(blk.getnnz(axis=0)).ravel().astype(np.float64)
        col_sum += s_blk
        col_sumsq += s2_blk
        col_cnt += cnt_blk
        ci += 1
        if ci % 2 == 0 or ci == n_chunks:
            print("  [select_hvg] %d/%d 块" % (ci, n_chunks), flush=True)
        del blk, blk2, s_blk, s2_blk, cnt_blk
        import gc; gc.collect()
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_i = np.where(col_cnt > 0, col_sum / col_cnt, 0.0)
        var = np.where(col_cnt > 1, col_sumsq / col_cnt - mean_i ** 2, 0.0)
    var = np.nan_to_num(var)
    top = np.argsort(var)[::-1][:n]
    return np.sort(top)


# ----------------------------------------------------------- 6. 样本构建
def control_means_by_line(reader, hvg, ctrl_mask, line_labels, max_cells=4000):
    """预计算每细胞系的对照表达均值（降维到 hvg），供 _expr_delta_for_label 复用，避免每个样本重算。"""
    means = {}
    for line in set(line_labels):
        if line is None:
            continue
        ci = [i for i in range(reader.n_cells)
              if ctrl_mask[i] and line_labels[i] == line]
        if not ci:
            continue
        if len(ci) > max_cells:
            ci = ci[:max_cells]
        x = reader.X_hvg(ci, hvg).astype(np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        means[line] = x.mean(0).astype(np.float32)
    return means


def _expr_delta_for_label(reader, label, ctrl_mask, line_labels, hvg=None,
                          max_cells=4000, ctrl_means=None):
    """
    计算某个扰动标签的表达变化向量（扰动均值 − 同细胞系对照均值）。
    返回 [n_hvg] 或 None。ctrl_means 为预计算的每细胞系对照均值（大幅加速）。
    """
    pert_idx = [i for i, l in enumerate(reader.pert_labels)
                if l == label and not ctrl_mask[i]]
    if not pert_idx:
        return None
    line = line_labels[pert_idx[0]] if pert_idx else None
    # 对照均值：优先同系预计算，否则全局预计算，最后才现场算（修原逻辑 Fulvestrant 返回 None 的坑）
    ctrl_mean = None
    if ctrl_means is not None:
        if line in ctrl_means:
            ctrl_mean = ctrl_means[line]
        elif len(ctrl_means) == 1:
            ctrl_mean = next(iter(ctrl_means.values()))
    if ctrl_mean is None:  # fallback 现场计算
        if line is not None:
            same_ctrl = [i for i in range(reader.n_cells)
                         if ctrl_mask[i] and line_labels[i] == line]
            if not same_ctrl:
                same_ctrl = [i for i in range(reader.n_cells) if ctrl_mask[i]]
        else:
            same_ctrl = [i for i in range(reader.n_cells) if ctrl_mask[i]]
        if not same_ctrl:
            return None
        if len(same_ctrl) > max_cells:
            same_ctrl = np.random.default_rng(0).choice(
                same_ctrl, max_cells, replace=False).tolist()
        xc = reader.expression(same_ctrl).astype(np.float32)
        if xc.ndim == 1:
            xc = xc.reshape(1, -1)
        if hvg is not None:
            xc = xc[:, hvg]
        ctrl_mean = xc.mean(0)
    # 仅切片扰动细胞（对照均值已预计算）
    pidx = pert_idx
    if len(pidx) > max_cells:
        pidx = np.random.default_rng(0).choice(pidx, max_cells, replace=False).tolist()
    xp = reader.expression(pidx).astype(np.float32)
    if xp.ndim == 1:
        xp = xp.reshape(1, -1)
    if hvg is not None:
        xp = xp[:, hvg]
    delta = xp.mean(0) - ctrl_mean
    return delta.astype(np.float32)


def build_samples(reader, local, kind="gene", hvg=None, max_cells=4000):
    """
    构建 (输入, 细胞系, 表达变化标签) 样本列表。
      kind='gene'    ：用本地基因（已对齐到数据扰动基因）做输入
      kind='compound'：用本地药物（已对齐到数据化合物）做输入
    返回 list[{'kind','local_idx','name','cell_line_idx','expr_delta'}]
    注意：cell_line_idx 取数据里出现的第一个细胞系在 CELL_LINE_NAMES 中的下标，
          映射不到则用 0（K562）。
    """
    ctrl_mask = reader.control_mask()
    ctrl_means = control_means_by_line(reader, hvg, ctrl_mask, reader.line_labels)
    if kind == "gene":
        aligned = align_genes(local["genes"], reader.labels_unique())
        # reader.labels_unique 是扰动标注集合，含基因名与化合物名混杂，
        # 这里只保留“确实是本地基因”的
        genes_norm_in_data = {norm(g) for g in reader.labels_unique()}
        aligned = {i: name for i, name in aligned.items()
                   if norm(name) in genes_norm_in_data}
    else:  # compound
        aligned = align_compounds(local["drugs"], reader.labels_unique())
    samples = []
    for li, data_name in aligned.items():
        # 该标签的扰动细胞按细胞系分组：每系独立一个纯净样本（同系扰动 − 同系对照）
        pert_cells = [i for i, l in enumerate(reader.pert_labels)
                      if l == data_name and not ctrl_mask[i]]
        by_line = defaultdict(list)
        for i in pert_cells:
            by_line[reader.line_labels[i]].append(i)
        for line, idxs in by_line.items():
            cl_idx = CELL_LINE_NAMES.index(line) if line in CELL_LINE_NAMES else 0
            ctrl_mean = ctrl_means.get(line)
            if ctrl_mean is None and ctrl_means:
                ctrl_mean = next(iter(ctrl_means.values()))
            if ctrl_mean is None:
                continue
            pidx = idxs
            if len(pidx) > max_cells:
                pidx = np.random.default_rng(0).choice(pidx, max_cells, replace=False).tolist()
            xp = reader.X_hvg(pidx, hvg).astype(np.float32)
            if xp.ndim == 1:
                xp = xp.reshape(1, -1)
            delta = (xp.mean(0) - ctrl_mean).astype(np.float32)
            samples.append({"kind": kind, "local_idx": li, "name": data_name,
                            "cell_line_idx": cl_idx, "expr_delta": delta})
    return samples


def build_compound_samples_chembl(reader, chem_map, hvg, max_cells=4000):
    """P2：用 chembl 直连的化合物构建样本（不依赖本地 218 药名字匹配）。

    chem_map: {perturbation: {chembl_id, smiles, target, pathway}}（sciplex3_chems.json）
    只保留 chem_map 里有 SMILES 的化合物 × 数据里出现的细胞系（sciPlex3 三系全用）。
    返回 samples: {kind:'compound', name, chembl_id, cell_line_idx, expr_delta}
    """
    ctrl_mask = reader.control_mask()
    ctrl_means = control_means_by_line(reader, hvg, ctrl_mask, reader.line_labels)
    by_key = defaultdict(list)
    for i, l in enumerate(reader.pert_labels):
        if l is None or ctrl_mask[i]:
            continue
        if l not in chem_map:
            continue            # 只保留有 SMILES 的化合物
        by_key[(l, reader.line_labels[i])].append(i)
    samples = []
    for (name, line), idxs in by_key.items():
        cl_idx = CELL_LINE_NAMES.index(line) if line in CELL_LINE_NAMES else 0
        ctrl_mean = ctrl_means.get(line)
        if ctrl_mean is None and ctrl_means:
            ctrl_mean = next(iter(ctrl_means.values()))
        if ctrl_mean is None:
            continue
        if len(idxs) > max_cells:
            idxs = np.random.default_rng(0).choice(idxs, max_cells, replace=False).tolist()
        xp = reader.X_hvg(idxs, hvg).astype(np.float32)
        if xp.ndim == 1:
            xp = xp.reshape(1, -1)
        delta = (xp.mean(0) - ctrl_mean).astype(np.float32)
        samples.append({"kind": "compound", "name": name,
                        "chembl_id": chem_map[name]["chembl_id"],
                        "cell_line_idx": cl_idx, "expr_delta": delta})
    return samples


# ----------------------------------------------------------- 6b. 全基因预训练样本（供 G2CP 阶段 A / 基因图）
def build_gene_samples_for_pretrain(reader, hvg, max_cells=2000, max_genes=None):
    """
    从 reader 提取所有“纯基因扰动”样本（单/双基因 KO），按 (标签, 细胞系) 分组产样本。
    返回 (samples, gene_vocab)，gene_vocab: 基因名→idx（0 为 padding）。
    样本 dict 含 gene_ids([k], padding 0) / gene_names / cell_line_idx / expr_delta。
    """
    ctrl_mask = reader.control_mask()
    ctrl_means = control_means_by_line(reader, hvg, ctrl_mask, reader.line_labels)
    var_set = set(reader.var_names)
    gene_vocab = {}

    def gid(name):
        if name not in gene_vocab:
            gene_vocab[name] = len(gene_vocab) + 1  # 0 = padding
        return gene_vocab[name]

    by_key = defaultdict(list)
    for i, l in enumerate(reader.pert_labels):
        if l is None or ctrl_mask[i]:
            continue
        parts = [p for p in re.split(r"[+_/]", l) if p]
        if not parts:
            continue
        genes = [p for p in parts if p in var_set]
        if len(genes) != len(parts):   # 含非基因（化合物）→ 留给化合物/组合路，跳过
            continue
        key = (l, reader.line_labels[i])
        by_key[key].append(i)

    keys = list(by_key.keys())
    if max_genes is not None:
        keys = keys[:max_genes]
    print("  [build_gene_samples] %d 个基因扰动组合，读取并降维到 HVG..."
          % len(keys))

    # 一次性把所有扰动细胞的表达读进内存并降维到 HVG，避免每组重复磁盘读
    all_idx = sorted({i for k in keys for i in by_key[k]})
    idx_to_row = {idx: r for r, idx in enumerate(all_idx)} if all_idx else {}
    X_hvg = None
    if all_idx:
        X_hvg = reader.X_hvg(all_idx, hvg)   # [N, len(hvg)] 已在内存/磁盘降维读取
        if X_hvg.ndim == 1:
            X_hvg = X_hvg.reshape(1, -1)

    samples = []
    for (l, line) in keys:
        genes = [p for p in re.split(r"[+_/]", l) if p in var_set]
        gids = [gid(g) for g in genes]
        cl_idx = CELL_LINE_NAMES.index(line) if line in CELL_LINE_NAMES else 0
        ctrl_mean = ctrl_means.get(line)
        if ctrl_mean is None and ctrl_means:
            ctrl_mean = next(iter(ctrl_means.values()))
        if ctrl_mean is None:
            continue
        idxs = by_key[(l, line)]
        if len(idxs) > max_cells:
            idxs = np.random.default_rng(0).choice(idxs, max_cells, replace=False).tolist()
        if X_hvg is not None:
            xp = X_hvg[[idx_to_row[i] for i in idxs]]   # 内存切片，秒级
        else:
            xp = reader.expression(idxs)
            if xp.ndim == 1:
                xp = xp.reshape(1, -1)
            if hvg is not None:
                xp = xp[:, hvg]
        delta = (xp.mean(0) - ctrl_mean).astype(np.float32)
        samples.append({"gene_ids": gids, "gene_names": genes,
                        "cell_line_idx": cl_idx, "expr_delta": delta})
    return samples, gene_vocab


# ----------------------------------------------------------- 7. self-test（用 Norman 验证基因侧）
def selftest(norman_path=None, local_path=None, hvg_n=2000):
    here = os.path.dirname(os.path.abspath(__file__))
    if local_path is None:
        local_path = os.path.join(os.path.dirname(here), "dataset.json")
    if norman_path is None:
        norman_path = ("C:/Users/wkr20/Desktop/virtual_cell_real_data/"
                       "genetic/NormanWeissman2019_filtered.h5ad")
    print(">>> 加载本地 dataset.json ...")
    local = load_local_dataset(local_path)
    print(f"    本地基因={len(local['genes'])} 药物={len(local['drugs'])} "
          f"有序列基因={len(local['seqs'])}")
    print(">>> 读取 Norman (backed) ...")
    r = PerturbationReader(norman_path, backed=True)
    print("    ", r.summary())
    print(">>> 基因名对齐（本地20基因 ∩ Norman扰动基因）...")
    aligned = align_genes(local["genes"], r.labels_unique())
    print(f"    对齐上 {len(aligned)} 个基因:")
    for i, name in aligned.items():
        print(f"      {local['genes'][i]:12s} -> {name}")
    print(">>> 选 HVG (top-%d) ..." % hvg_n)
    hvg = select_hvg(r, n=hvg_n)
    print(f"    HVG 维度={len(hvg)}")
    print(">>> 构建基因侧样本（表达变化标签）...")
    samples = build_samples(r, local, kind="gene", hvg=hvg)
    print(f"    成功构建 {len(samples)} 个基因样本")
    if samples:
        s0 = samples[0]
        print(f"    例: {s0['name']} cell_line_idx={s0['cell_line_idx']} "
              f"标签维度={s0['expr_delta'].shape}")
        print(f"    标签 L2 范数={float(np.linalg.norm(s0['expr_delta'])):.2f} "
              f"(≠0 说明确有扰动信号)")
    r.close()
    print(">>> self-test 完成 ✅")
    return {"aligned_genes": aligned, "n_samples": len(samples)}


if __name__ == "__main__":
    selftest()
