"""S26: 两套评价矩阵底层 signature 的逐组合配对核查。

要回答的问题（对应正文"在两套数据中都能追溯到底层 signature 的组合中，
signature 集合完全一致占 99.49%，没有任何组合使用完全不同的 signature 集合"）：

  beta2020 与 dcic2021 都包含的细胞系–药物组合里，两边各自用了哪些
  signature？这些 signature 是不是同一批？

匹配键与比较规则
----------------
1. 组合键 = (细胞系, 药物)。两侧都用同一套 162 个细胞系的规范名，
   beta2020 侧的 siginfo_beta.txt 给出 cell_iname / cell_mfc_name，
   统一到 g2cp 缓存的 cl_names；9 个写法不一致的细胞系按别名表映射
   （如 H1563 -> NCIH1563）。药物用 pert_id / pert_id 字符串直比。
2. signature 标识 = sig_id 原串，不做任何改写。sig_id 自带 batch、
   细胞系、扰动、剂量、时间，因此 sig_id 相同即视为同一条实验 signature。
3. 逐组合比较的是「sig_id 集合」，不是 sig_id 条数：
     identical  = 两侧集合严格相等
     subset     = 一侧是另一侧的真子集（成员相同，仅一侧有额外 signature）
     partial    = 两侧有交集但互不包含
     disjoint   = 交集为空
   正文引用的「signature 集合完全一致」= identical，即 11,217 / 11,274；
   「没有任何组合使用完全不同的 signature 集合」= disjoint 为 0。
   subset 不计入「完全一致」：这些组合的一侧确有额外 signature，
   逐条列在 S26_signature_pair_audit.csv 中，不做静默合并。

输出
----
S26_signature_pair_audit.csv        逐组合明细（每个组合一行）
S26_signature_pair_summary.csv     四类计数与占比（供正文数字溯源）

依赖：原始 siginfo_beta.txt 与 cp_coeff_mat.gctx 需在本机可读。
"""
import os
import sys
import h5py
import numpy as np
import pandas as pd

BETA_SIG = r'C:/Users/wkr20/Desktop/siginfo_beta.txt'
DCIC = r'C:/Users/wkr20/Desktop/cp_coeff_mat.gctx'
EVAL = (r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_figwork2/GigaScience_GTEx_checked'
        r'/source_data/frozen_multimetric_perrow_scores.csv')
CACHE = (r'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37/data'
         r'/g2cp_cache_beta_v2/meta.npz')
OUT = os.path.dirname(os.path.abspath(__file__))

for p in (BETA_SIG, DCIC, EVAL, CACHE):
    if not os.path.exists(p):
        sys.exit('缺少输入文件: %s' % p)

# ---- 评价集：两套矩阵共同包含的 (细胞系, 药物) ----
ev = pd.read_csv(EVAL)
evkey = set(zip(ev.cell.astype(str), ev.drug.astype(str)))
print('评价集组合数 =', len(evkey))

# ---- beta2020 侧：siginfo 只取 trt_cp ----
m = np.load(CACHE, allow_pickle=True)
V162 = set(str(x) for x in m['cl_names'])
ALIAS = {'AG062632': 'AG06263_2', 'AG06840A': 'AG06840_A', 'AG078N11': 'AG078N1_1',
         'H1563': 'NCIH1563', 'H1781': 'NCIH1781', 'H2110': 'NCIH2110',
         'ND347327': 'ND34732_7', 'U266': 'U266B1'}
REV = {v: v for v in V162}
REV.update({v: k for k, v in ALIAS.items()})


def resolve(cin, cmfc):
    for x in (cmfc, cin):
        if isinstance(x, str) and x in REV:
            return REV[x]
    return None


A = pd.read_csv(BETA_SIG, sep='\t', low_memory=False,
                usecols=['sig_id', 'pert_id', 'pert_type', 'cell_iname', 'cell_mfc_name'])
A = A[A.pert_type == 'trt_cp'].copy()
A['cell'] = [resolve(c, f) for c, f in zip(A.cell_iname, A.cell_mfc_name)]
A['drug'] = A.pert_id.astype(str)
A['sig'] = A.sig_id.astype(str)
A = A[A.cell.notna()]
Ae = A[[(c, d) in evkey for c, d in zip(A.cell, A.drug)]]

# ---- dcic2021 侧：gctx 的 META/COL ----
with h5py.File(DCIC, 'r') as f:
    rd = lambda k: np.array(
        [x.decode() if isinstance(x, bytes) else str(x)
         for x in f['/0/META/COL'][k][:]], dtype=object)
    D = pd.DataFrame({'sig': rd('id'), 'cell': rd('cell_line'), 'drug': rd('pert_id')})
De = D[[(c, d) in evkey for c, d in zip(D.cell, D.drug)]]

nb, nd = Ae.sig.nunique(), De.sig.nunique()
inter = set(Ae.sig) & set(De.sig)
print('beta 侧 sig_id %d | dcic 侧 %d | 交集 %d' % (nb, nd, len(inter)))
print('signature key 覆盖率 = %.2f%%' % (100.0 * len(inter) / nb))

GB = {k: frozenset(v) for k, v in Ae.groupby(['cell', 'drug']).sig}
GD = {k: frozenset(v) for k, v in De.groupby(['cell', 'drug']).sig}
keys = sorted(set(GB) & set(GD))
single = sorted(set(evkey) - set(GB) - set(GD))
only_b = sorted(set(GB) - set(GD))
only_d = sorted(set(GD) - set(GB))
print('两侧都能定位到底层 signature 的组合 = %d / %d' % (len(keys), len(evkey)))
print('  仅 beta2020 侧可定位 = %d；仅 dcic2021 侧可定位 = %d；两侧都不可定位 = %d'
      % (len(only_b), len(only_d), len(single)))
for k in single + only_b + only_d:
    print('    未配对组合: %s | %s' % k)

rows = []
for k in keys:
    sb, sd = GB[k], GD[k]
    if sb == sd:
        rel = 'identical'
    elif sb < sd or sd < sb:
        rel = 'subset'
    elif sb & sd:
        rel = 'partial'
    else:
        rel = 'disjoint'
    rows.append({'cell': k[0], 'drug': k[1], 'relation': rel,
                 'n_sig_beta2020': len(sb), 'n_sig_dcic2021': len(sd),
                 'n_shared_sig': len(sb & sd),
                 'n_only_beta2020': len(sb - sd), 'n_only_dcic2021': len(sd - sb)})

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, 'S26_signature_pair_audit.csv'), index=False)

cnt = R.relation.value_counts()
same = int(cnt.get('identical', 0))
subsets = R[R.relation == 'subset']
summ = pd.DataFrame({
    'category': ['both_traceable', 'identical', 'subset', 'partial', 'disjoint',
                 'identical_pct_of_traceable', 'subset_pct_of_traceable',
                 'untraceable_in_both', 'traceable_only_in_beta2020',
                 'traceable_only_in_dcic2021'],
    'value': [len(keys), same, int(cnt.get('subset', 0)),
              int(cnt.get('partial', 0)), int(cnt.get('disjoint', 0)),
              round(100.0 * same / len(keys), 4),
              round(100.0 * len(subsets) / len(keys), 4),
              len(single), len(only_b), len(only_d)]})
summ.to_csv(os.path.join(OUT, 'S26_signature_pair_summary.csv'), index=False)

print('\n完全一致(集合严格相等) = %d / %d = %.2f%%' % (same, len(keys), 100.0 * same / len(keys)))
print('关系分布:', {k: int(v) for k, v in cnt.items()})
print('subset 组合：beta 侧独有 signature 合计 %d，dcic2021 侧独有合计 %d；'
      '其中 beta 侧更少 %d 个、更多 %d 个'
      % (int(subsets.n_only_beta2020.sum()), int(subsets.n_only_dcic2021.sum()),
         int((subsets.n_sig_beta2020 < subsets.n_sig_dcic2021).sum()),
         int((subsets.n_sig_beta2020 > subsets.n_sig_dcic2021).sum())))
print('交集为空的组合数 = %d' % int(cnt.get('disjoint', 0)))
print('已写出 S26_signature_pair_audit.csv (%d 行) / S26_signature_pair_summary.csv'
      % len(R))
if cnt.get('disjoint', 0):
    print('\n注意：存在交集为空的组合，需人工核对')
