"""
化合物编码器（官方 UniPert 同款 ECFP4 路线）
============================================
输入：SMILES 字符串
流程：SMILES → ECFP4 指纹（RDKit Morgan 半径2, 2048 位）→ Linear(2048→EMBED_DIM)

说明：
  - 对齐官方 UniPert 实现（TencentAILabHealthcare/UniPert）：化合物编码器 = ECFP4 + 单线性层。
  - 旧的手写 GINE 图编码器保留为 _CompoundEncoderGNN（不再默认使用）。
  - 依赖 RDKit（pip install rdkit）。
"""
import re
import numpy as np
import torch
import torch.nn as nn

from .config import EMBED_DIM, COMPOUND_HIDDEN

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _RDKIT_OK = True
except Exception as _e:  # pragma: no cover
    _RDKIT_OK = False
    _RDKIT_ERR = _e

ECFP4_BITS = 2048


def smiles_to_ecfp4(smiles):
    """SMILES → ECFP4 指纹（Morgan 半径2, 2048 位）→ np.float32 [2048]。
    解析失败返回 None。与官方 AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) 一致。"""
    if not _RDKIT_OK:
        raise RuntimeError(f"RDKit 不可用: {_RDKIT_ERR}")
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ECFP4_BITS)
    return np.array(fp, dtype=np.float32)


# 常见原子（原子类型 one-hot 用）；含常见双字母元素，覆盖真实药物 SMILES
ATOMS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "H",
         "Ca", "Mg", "Fe", "Zn", "Na", "K", "Cu", "Mn", "Co", "Ni", "Se",
         "As", "Ag", "Au", "Hg", "Pb", "Sn", "Cr", "Al", "Ti", "V", "Ba",
         "Sr", "Cd", "Li", "Be", "Te", "Mo", "Ru", "Rh", "Pd", "Pt", "W"]
ATOM_SET = set(ATOMS)
NODE_DIM = len(ATOMS) + 2      # one-hot(原子) + 芳香性 + 度数(归一化)
EDGE_DIM = 6                   # 键型 one-hot(-, =, #, /, \, :)


# ---------------------------------------------------------------- SMILES 解析
def _tokenize(smiles):
    """把 SMILES 拆成 (类型, 值) 记号流：atom / bond / branch / ring。"""
    tokens = []
    i, n = 0, len(smiles)
    while i < n:
        c = smiles[i]
        if c == '[':
            j = smiles.index(']', i)
            tokens.append(('atom', smiles[i + 1:j])); i = j + 1
        elif c.isupper():
            if i + 1 < n and smiles[i + 1].islower() and smiles[i:i + 2] in ATOM_SET:
                tokens.append(('atom', smiles[i:i + 2])); i += 2
            else:
                tokens.append(('atom', c)); i += 1
        elif c.islower():
            tokens.append(('atom', c)); i += 1
        elif c in '-=#/\\:':
            tokens.append(('bond', c)); i += 1
        elif c == '(':
            tokens.append(('branch', '(')); i += 1
        elif c == ')':
            tokens.append(('branch', ')')); i += 1
        elif c.isdigit():
            if c == '%' and i + 2 < n:           # 两位数环号 %10
                tokens.append(('ring', smiles[i + 1:i + 3])); i += 3
            else:
                tokens.append(('ring', c)); i += 1
        else:                       # 忽略 ' ' '.' 等分隔符与未识别字符
            i += 1
    return tokens


def smiles_to_graph(smiles):
    """返回 (atom_types:List[str], edge_index:List[[a,b]], edge_type:List[str]) 无向去重。"""
    tokens = _tokenize(smiles)
    atom_types, edges = [], []
    ring_open, stack, last = {}, [], -1
    pending_bond = '-'
    for kind, val in tokens:
        if kind == 'atom':
            idx = len(atom_types)
            atom_types.append(val)
            if last >= 0:
                edges.append((last, idx, pending_bond))
            if stack:                 # 分支起点与当前原子相连
                p = stack[-1]
                edges.append((p, idx, pending_bond))
                stack[-1] = idx
            last = idx
            pending_bond = '-'
        elif kind == 'bond':
            pending_bond = val
        elif kind == 'branch':
            if val == '(':
                stack.append(last)
            else:
                if stack:
                    stack.pop()
        elif kind == 'ring':
            if val in ring_open:
                a = ring_open.pop(val)
                if a != last:
                    edges.append((a, last, pending_bond))
            else:
                ring_open[val] = last
            pending_bond = '-'
    # 去重成无向边（每条边存正反两次，供消息传递用）
    seen, edge_index, edge_type = set(), [], []
    for a, b, bt in edges:
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        edge_index.append([a, b]); edge_index.append([b, a])
        edge_type.append(bt); edge_type.append(bt)
    return atom_types, edge_index, edge_type


# ---------------------------------------------------------------- 图特征构造
def atom_features(atom_types, edge_index):
    deg = [0] * len(atom_types)
    for a, b in edge_index:
        deg[a] += 1; deg[b] += 1
    feats = []
    for at, d in zip(atom_types, deg):
        onehot = [1.0 if at == a else 0.0 for a in ATOMS]
        aromatic = 1.0 if (at and at[0].islower()) else 0.0   # 芳香小写原子 c/n/o...
        feats.append(onehot + [aromatic, min(float(d), 6.0) / 6.0])
    return torch.tensor(feats, dtype=torch.float)             # [n, NODE_DIM]


def bond_onehot(bt):
    order = {'-': 0, '=': 1, '#': 2, '/': 3, '\\': 4, ':': 5}
    v = [0.0] * EDGE_DIM
    v[order.get(bt, 0)] = 1.0
    return v


def _global_pool(x, batch, mode):
    """对拼接的大图按 batch 维度做全局池化。x:[N,H], batch:[N]。"""
    dev = x.device
    nb = int(batch.max().item()) + 1
    if mode == 'mean':
        out = torch.zeros((nb, x.size(1)), dtype=x.dtype, device=dev)
        cnt = torch.zeros(nb, dtype=x.dtype, device=dev)
        out.index_add_(0, batch, x)
        cnt.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        return out / cnt.clamp_min(1.0).unsqueeze(1)
    else:  # max
        out = torch.full((nb, x.size(1)), float('-inf'), dtype=x.dtype, device=dev)
        for i in range(x.size(0)):
            # 注意：不能用 out[batch[i]] = ... 原地写，会破坏 autograd 版本
            out = torch.maximum(out, x[i].unsqueeze(0))
        return out


# ---------------------------------------------------------------- 旧 GNN 编码器（保留）
class _CompoundEncoderGNN(nn.Module):
    """（旧路线，不再默认使用）手写 GINE 图编码器。ECFP4 改造前训练的旧权重仍用它加载。"""
    def __init__(self):
        super().__init__()
        self.node_lin = nn.Linear(NODE_DIM, COMPOUND_HIDDEN)
        self.edge_lin = nn.Linear(EDGE_DIM, COMPOUND_HIDDEN)
        # 3 层 GINE：h = MLP((1+eps)·h + Σ邻居h + 边)
        self.gine = nn.ModuleList([
            nn.Sequential(
                nn.Linear(COMPOUND_HIDDEN, COMPOUND_HIDDEN),
                nn.GELU(),
                nn.Linear(COMPOUND_HIDDEN, COMPOUND_HIDDEN),
            ) for _ in range(3)
        ])
        self.eps = nn.Parameter(torch.zeros(3))
        self.readout = nn.Linear(COMPOUND_HIDDEN * 2, EMBED_DIM)  # mean+max 拼接
        self.ln = nn.LayerNorm(EMBED_DIM)

    def forward(self, graphs):
        """
        graphs: list of (atom_types, edge_index, edge_type)  —— 一个 batch 多个分子
        返回 [B, EMBED_DIM]
        """
        device = next(self.parameters()).device
        node_feats, batch, edge_index, edge_attr = [], [], [], []
        offset = 0
        for bi, (atom_types, ei, et) in enumerate(graphs):
            nf = atom_features(atom_types, ei).to(device)
            node_feats.append(nf)
            batch += [bi] * len(atom_types)
            for (a, b) in ei:
                edge_index.append([a + offset, b + offset])
            for bt in et:
                edge_attr.append(bond_onehot(bt))
            offset += len(atom_types)

        x = torch.cat(node_feats, 0)                  # [N_total, NODE_DIM]
        x = self.node_lin(x)
        batch = torch.tensor(batch, dtype=torch.long, device=device)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()   # [2, E]
        edge_attr = self.edge_lin(torch.tensor(edge_attr, dtype=torch.float, device=device))  # [E, H]

        for li, layer in enumerate(self.gine):
            row, col = edge_index
            msg = x[col] + edge_attr                   # 邻居消息 + 边特征
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, msg)                # 按目标节点聚合
            x = layer((1.0 + self.eps[li]) * x + agg)

        mean = _global_pool(x, batch, 'mean')
        mx = _global_pool(x, batch, 'max')
        return self.ln(self.readout(torch.cat([mean, mx], 1)))


# ---------------------------------------------------------------- ECFP4 编码器（官方同款）
class CompoundEncoder(nn.Module):
    """官方 UniPert 同款化合物编码器：ECFP4 指纹 → 单线性层 → EMBED_DIM。

    对齐 TencentAILabHealthcare/UniPert 的 modules.py：
        ecfp4_emb = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        embs = self.linear_layer(embs)   # nn.Linear(2048, 256)

    forward 输入：smiles 列表 或 指纹列表（np.float32 [B,2048]），输出 [B, EMBED_DIM]。
    """
    def __init__(self, fp_dim=ECFP4_BITS):
        super().__init__()
        self.linear_layer = nn.Linear(fp_dim, EMBED_DIM)
        # LayerNorm：官方嵌入无归一化（norm≈47），下游回归头会饱和；LN 归一化尺度后可训练
        self.ln = nn.LayerNorm(EMBED_DIM)

    def forward(self, graphs):
        """graphs: list of np.ndarray [2048]（ECFP4 指纹）或 list of str（SMILES）。"""
        if graphs and isinstance(graphs[0], str):
            fps = [smiles_to_ecfp4(s) for s in graphs]
        else:
            fps = list(graphs)
        # 过滤解析失败（None / 全零）→ 用零向量占位（训练时不该出现；推理时给出空分子嵌入）
        xs = []
        for f in fps:
            if f is None or (isinstance(f, np.ndarray) and not f.any()):
                xs.append(np.zeros(ECFP4_BITS, dtype=np.float32))
            else:
                xs.append(np.asarray(f, dtype=np.float32))
        x = torch.from_numpy(np.stack(xs)).to(next(self.parameters()).device)
        return self.ln(self.linear_layer(x))
