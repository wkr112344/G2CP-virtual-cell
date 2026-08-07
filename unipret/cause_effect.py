"""
因果建模 + Mantel 检验（论文 第三部分）
=====================================
完整双层空间建模：
  扰动因空间 CS_cause：UniPert 统一嵌入的相似矩阵（只取决于分子本身，与细胞无关，通用基线）
  表型果空间 CS_effect：G2CP 预测的扰动响应相似矩阵（细胞系特异性）
用 Mantel 检验两个矩阵的相关性，量化「药物药理类(PCL)在某细胞里的敏感性」：
  相关性越高 → 该类分子在该细胞中诱导高度一致的转录应答（如 ER 药物在 MCF7 最敏感）。

本模块与具体编码器解耦：输入两组向量（cause 嵌入 / effect 预测），输出 Mantel r 与 p。
并提供 run_cause_effect_analysis() 直接从对齐桥样本构建两矩阵并检验（用我们手上的化合物子集演示）。
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


def cosine_matrix(V):
    """V:[M,D]（行=扰动）→ [M,M] 余弦相似度矩阵。"""
    V = np.asarray(V, dtype=np.float64)
    V = V - V.mean(1, keepdims=True)
    n = np.linalg.norm(V, axis=1, keepdims=True).clip(min=1e-12)
    V = V / n
    return V @ V.T


def mantel_test(cause_vecs, effect_vecs, n_perm=999, seed=0):
    """
    Mantel 检验：CS_cause 与 CS_effect 的向量化距离向量之间的相关性。
      cause_vecs / effect_vecs : np [M,D]（同一组扰动，顺序一致）
    返回 dict: r(原始相关), p(置换p值), n
    """
    cause_vecs = np.asarray(cause_vecs, dtype=np.float64)
    effect_vecs = np.asarray(effect_vecs, dtype=np.float64)
    assert cause_vecs.shape[0] == effect_vecs.shape[0], "两组扰动数必须一致"
    M = cause_vecs.shape[0]
    if M < 3:
        return {"r": float("nan"), "p": float("nan"), "n": M,
                "note": "扰动数 < 3，无法做 Mantel"}
    d_cause = pdist(cosine_matrix(cause_vecs), metric="euclidean")
    d_effect = pdist(cosine_matrix(effect_vecs), metric="euclidean")
    r_obs, _ = pearsonr(d_cause, d_effect)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(M)
        d_eff_perm = pdist(cosine_matrix(effect_vecs[perm]), metric="euclidean")
        r_p, _ = pearsonr(d_cause, d_eff_perm)
        if abs(r_p) >= abs(r_obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"r": float(r_obs), "p": float(p), "n": M}


def run_cause_effect_analysis(unipert, effect_model, reader, local, hvg_n=2000,
                              device="cpu", n_perm=199, max_samples=40):
    """
    对我们手上的对齐化合物子集，直接构建 CS_cause / CS_effect 并 Mantel 检验。
      unipert      : 训好的 UniPert（提供统一嵌入 = 因空间）
      effect_model : 训好的 G2CP（提供预测表达变化 = 果空间）
      reader/local : 对齐桥数据
    返回 dict 含 r, p, n, names
    """
    from .data_bridge import build_samples, select_hvg
    from .compound_encoder import CompoundGraphCache

    hvg = select_hvg(reader, n=hvg_n, sample_cells=2000)
    samples = build_samples(reader, local, kind="compound", hvg=hvg)
    if len(samples) > max_samples:
        samples = samples[:max_samples]
    if not samples:
        return {"error": "无对齐化合物样本"}

    unipert.eval(); effect_model.eval()
    cache = CompoundGraphCache()
    drug_smiles = {d["name"]: d["smiles"] for d in local["drugs"]}

    cause_list, effect_list, names = [], [], []
    with torch.no_grad():
        for s in samples:
            nm = s["name"]
            sm = drug_smiles.get(nm.split(" (")[0].split(" ")[0], "")
            if not sm:
                continue
            g = cache.get(sm)
            cl = torch.tensor([s["cell_line_idx"]], dtype=torch.long, device=device)
            c_rep = unipert.encode_compound([g], cell_line_idx=cl).detach().cpu().numpy()[0]
            # 果空间：G2CP 对该化合物在该细胞系的预测表达变化
            e_pred = effect_model.forward_compound([g], cell_line_idx=cl).detach().cpu().numpy()[0]
            cause_list.append(c_rep)
            effect_list.append(e_pred)
            names.append(nm)
    if len(cause_list) < 3:
        return {"error": "可用化合物不足 3 个", "n": len(cause_list)}
    res = mantel_test(np.stack(cause_list), np.stack(effect_list), n_perm=n_perm)
    res["names"] = names
    return res


# 便于在 notebook / CLI 里直接看 CS 矩阵
def similarity_report(cause_vecs, effect_vecs):
    return {
        "CS_cause": cosine_matrix(np.asarray(cause_vecs)).round(3).tolist(),
        "CS_effect": cosine_matrix(np.asarray(effect_vecs)).round(3).tolist(),
    }
