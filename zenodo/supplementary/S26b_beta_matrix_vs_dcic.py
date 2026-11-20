"""S26b: 用 beta2020 最终矩阵**实际选入**的 signature 成员重做配对核查。

为什么必须有 S26b（S26 的局限）
------------------------------
S26 比的是两侧**元数据**里符合条件的 signature：
  beta 侧  siginfo_beta.txt 中 pert_type==trt_cp 且细胞系可映射的条目
  dcic 侧  cp_coeff_mat.gctx 的 META/COL id
但 beta 最终矩阵不是照单全收。构建脚本
`2026-09-06-00-53-37/build_y.py::wanted()` 的条件是：

    pert_type == 'trt_cp'          （与 S26 同）
  且 pert_id ∈ drug_vocab          （S26 缺这一条）
  且 cell_id ∈ cl_names            （与 S26 同）

所以 S26 只证明「两侧元数据成员高度一致」，
不足以升级为「最终聚合实际使用的成员已逐一验证一致」。
本脚本原样重跑 wanted()，取出真正进入 beta2020 矩阵的 sig_id 清单，
再与 dcic2021 逐组合比对。

注意：本脚本不读 y.npy，只读 siginfo_beta.txt 与两个词表，
      因此不需要 7 GB 矩阵即可复现。
"""
import os
import sys
import gzip
import h5py
import numpy as np
import pandas as pd

ROOT = r'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37'
CACHE = os.path.join(ROOT, 'data', 'g2cp_cache_beta_v2')
SIGINFO = r'C:/Users/wkr20/Desktop/siginfo_beta.txt'
DCIC = r'C:/Users/wkr20/Desktop/cp_coeff_mat.gctx'
EVAL = (r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_figwork2/GigaScience_GTEx_checked'
        r'/source_data/frozen_multimetric_perrow_scores.csv')
OUT = os.path.dirname(os.path.abspath(__file__))

for p in (SIGINFO, DCIC, EVAL, CACHE):
    if not os.path.exists(p):
        sys.exit('缺少输入: %s' % p)

meta = np.load(os.path.join(CACHE, 'meta.npz'), allow_pickle=True)
DRUG_VOCAB = set(str(x) for x in meta['drug_vocab'])
CL_NAMES = [str(x) for x in meta['cl_names']]

# 复刻 build_y.py::wanted()：cell_id 用 cell_mfc_name / cell_iname 归一到 cl_names
hdr = pd.read_csv(SIGINFO, sep='\t', nrows=0).columns.tolist()
ci_mfc = hdr.index('cell_mfc_name')
ci_in = hdr.index('cell_iname')
ci_pt = hdr.index('pert_type')
ci_pid = hdr.index('pert_id')
ci_sid = hdr.index('sig_id')

CL_SET = set(CL_NAMES)
want = {}          # (cell, drug) -> set(sig_id)
n_trt = 0
n_drug_ok = 0
n_cell_ok = 0
with open(SIGINFO, 'r', encoding='utf-8', errors='replace') as fh:
    head = fh.readline().rstrip('\n').split('\t')
    assert head[ci_sid] == 'sig_id', head[:5]
    for ln in fh:
        r = ln.rstrip('\n').split('\t')
        if len(r) <= ci_sid:
            continue
        if r[ci_pt] != 'trt_cp':
            continue
        n_trt += 1
        pid = r[ci_pid]
        if pid not in DRUG_VOCAB:
            continue
        n_drug_ok += 1
        cell = r[ci_mfc] if r[ci_mfc] in CL_SET else (
            r[ci_in] if r[ci_in] in CL_SET else None)
        if cell is None:
            continue
        n_cell_ok += 1
        want.setdefault((cell, pid), set()).add(r[ci_sid])

print('build_y.py::wanted() 复刻筛选：')
print('  trt_cp 行            = %d' % n_trt)
print('  且 pert_id ∈ 词表     = %d' % n_drug_ok)
print('  且 cell ∈ cl_names   = %d  （进入 beta2020 矩阵的 signature 条数）' % n_cell_ok)
print('  唯一 (细胞系,药物)   = %d' % len(want))

# ---- dcic2021 侧 ----
with h5py.File(DCIC, 'r') as f:
    rd = lambda k: np.array(
        [x.decode() if isinstance(x, bytes) else str(x)
         for x in f['/0/META/COL'][k][:]], dtype=object)
    D = pd.DataFrame({'cell': rd('cell_line'), 'drug': rd('pert_id'), 'sig': rd('id')})
D['sig'] = D['sig'].astype(str)

ev = pd.read_csv(EVAL)
evkey = set(zip(ev.cell.astype(str), ev.drug.astype(str)))
Dm = D[[(c, d) in evkey for c, d in zip(D.cell, D.drug)]]
GD = {k: frozenset(v) for k, v in Dm.groupby(['cell', 'drug'])['sig']}

keys = sorted(set(want) & set(GD))
single_b = sorted(set(want) - set(GD))
single_d = sorted(set(GD) - set(want))
in_ev = [k for k in want if k in evkey]
print('\n评价集 %d 对中：beta 矩阵侧可定位 %d，dcic 侧可定位 %d，两侧都有 %d'
      % (len(evkey), len(in_ev), len(evkey) - len(single_d),
         len([k for k in keys if k in evkey])))

rows = []
for k in keys:
    sb, sd = want[k], GD[k]
    if sb == sd:
        rel = 'identical'
    elif sb < sd or sd < sb:
        rel = 'subset'
    elif sb & sd:
        rel = 'partial'
    else:
        rel = 'disjoint'
    rows.append({'cell': k[0], 'drug': k[1], 'relation': rel,
                 'in_eval_set': k in evkey,
                 'n_sig_beta_matrix': len(sb), 'n_sig_dcic_gctx': len(sd),
                 'n_shared_sig': len(sb & sd),
                 'n_only_beta': len(sb - sd), 'n_only_dcic': len(sd - sb)})
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, 'S26b_beta_matrix_vs_dcic_gctx.csv'), index=False)

cnt = R.relation.value_counts()
same = int(cnt.get('identical', 0))
summ = pd.DataFrame({
    'category': ['both_traceable', 'identical', 'subset', 'partial', 'disjoint',
                 'identical_pct', 'beta_matrix_signature_rows',
                 'beta_only_combination', 'dcic_only_combination'],
    'value': [len(keys), same, int(cnt.get('subset', 0)), int(cnt.get('partial', 0)),
              int(cnt.get('disjoint', 0)), round(100.0 * same / max(len(keys), 1), 4),
              n_cell_ok, len(single_b), len(single_d)]})
summ.to_csv(os.path.join(OUT, 'S26b_summary.csv'), index=False)

print('\n关系分布: %s' % {k: int(v) for k, v in cnt.items()})
print('完全一致 = %d / %d = %.2f%%' % (same, len(keys), 100.0 * same / max(len(keys), 1)))
print('交集为空的组合 = %d' % int(cnt.get('disjoint', 0)))
print('已写出 S26b_beta_matrix_vs_dcic_gctx.csv (%d 行) / S26b_summary.csv' % len(R))
