"""S25 修复版：疾病榜单内部稳定性（逆转方向统一版）。

修复的三个问题
--------------
1. **逆转方向**。签名定义为「疾病 vs 正常相邻组织」，UP 是疾病中上调的基因、
   DN 是疾病中下调的基因。逆转（reversal）指药物把疾病状态推回正常方向，
   即把 UP 压低、把 DN 抬高。逐行分数

       r_row = (mean(y[UP]) - mean(y[DN])) / sd(y[UP ∪ DN]) * sqrt(|UP|·|DN|) / |UP ∪ DN|

   因此 **r_row 越小 = 越逆转**，榜单应取 r_row 最小的药物。
   旧版直接对 r_row 取最大的 10 名（`argsort(-a)`），方向与逆转相反，
   观测 top-10 重合数因此与正文／图 4 互补（1/3/2 ↔ 3/1/4）。

2. **观测与 bootstrap 同向**。旧版 OBS 取 `mean(r_row)`、bootstrap 也重抽
   r_row 的行，两者方向一致但都反了。修复版把分数统一定义为

       score = -r_row        （越大 = 越逆转）

   OBS 与 bootstrap 都用 score，方向自动一致。

3. **实际用于区间的是前 R 个复本的两两比较**。B=2,000 个复本全部生成，
   但区间由前 R=150 个复本的 C(150,2)=11,175 对两两比较得到。这些对
   两两共享复本，**不相互独立**，因此区间只作经验分位数描述，
   不当作 11,175 个独立重复，也不用于构造检验。
   旧版方法描述写「2,000 次」，与代码不符，本版已在输出中注明。

输入与输出
----------
输入  _dis_rowscores.npz（逐行 r_row）、_disease_kept1399.json（1,399 药物）
输出  S25_disease_within_reference_stability.csv（81 行 = 3 签名 × 9 指标 × 3 scope）
      S25_direction_check.csv（方向自检：与正文／图 4 的 1/3/2 逐位对齐）
"""
import json
import os
import numpy as np
import pandas as pd

Z = np.load(r'C:/Users/wkr20/WorkBuddy/Claw/_dis_rowscores.npz', allow_pickle=True)
kept = json.load(open(r'C:/Users/wkr20/WorkBuddy/Claw/_disease_kept1399.json'))
N = len(kept)
SHORT = ['Liver_Acevedo', 'Thyroid_ATC', 'APL_Casorelli']
KS = [10, 20, 50, 100, 200]
B = 2000          # 生成复本数
R = 150           # 实际用于两两比较的复本数，C(150,2) = 11,175 对
SEED = 20261004
OUT = os.path.dirname(os.path.abspath(__file__))

ROOT = r'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37'
dv = [str(x) for x in np.load(os.path.join(ROOT, 'data', 'g2cp_cache_beta_v2',
                                           'meta.npz'), allow_pickle=True)['drug_vocab']]

# ---- 每个药物在每库的行位置 ----
per = {}
for t in ['beta2020', 'dcic2021']:
    dr = Z['%s|drug' % t]
    order = np.argsort(dr, kind='stable')
    drs = dr[order]
    bnd = np.where(np.diff(drs) > 0)[0]
    starts = np.concatenate([[0], bnd + 1])
    ends = np.concatenate([bnd, [len(drs) - 1]])
    kpos = {d: j for j, d in enumerate(kept)}
    for a, b in zip(starts, ends):
        g = dv[drs[a]]
        if g in kpos:
            per.setdefault((t, kpos[g]), []).append(order[a:b + 1].copy())
    for j in range(N):
        per.setdefault((t, j), [])

# ---- 逐行 r_row 矩阵，缺行补 NaN ----
ROW, KN = {}, {}
for t in ['beta2020', 'dcic2021']:
    for s in SHORT:
        r = Z['%s|%s' % (t, s)]
        maxK = max((sum(len(ix) for ix in per[(t, j)]) for j in range(N)), default=1)
        M = np.full((N, max(1, maxK)), np.nan)
        k_n = np.zeros(N, dtype=np.int64)
        for j in range(N):
            ix = (np.concatenate(per[(t, j)]) if per[(t, j)]
                  else np.array([], dtype=np.int64))
            if len(ix):
                M[j, :len(ix)] = r[ix]
                k_n[j] = len(ix)
        ROW[(t, s)] = M
        KN[(t, s)] = k_n
    cnt = np.array([len(np.concatenate(per[(t, j)])) if per[(t, j)] else 0
                    for j in range(N)])
    print('%s: 有行药物 = %d/%d  行数 中位=%.0f 合计=%d'
          % (t, int((cnt > 0).sum()), N, np.median(cnt), cnt.sum()))


def score_of(M, k_n, rows_idx):
    """score = -mean(r_row)：越大 = 越逆转（UP 被压低、DN 被抬高）。"""
    out = np.full(len(rows_idx), np.nan)
    for oi, j in enumerate(rows_idx):
        ix = rows_idx[oi]
        if len(ix):
            out[oi] = -np.nanmean(M[j, ix])
    return out


def observed(t, s):
    M, k_n = ROW[(t, s)], KN[(t, s)]
    out = np.full(N, np.nan)
    for j in range(N):
        if k_n[j]:
            out[j] = -M[j, :k_n[j]].mean()
    return out


def rk(v):
    return pd.Series(v).rank(ascending=False, method='average').values


def topk(v, k):
    """取 score 最大的 k 个（score 越大 = 越逆转）。"""
    return set(np.argsort(-v, kind='stable')[:k])


def metrics(a, b):
    out = {'spearman': float(pd.Series(rk(a)).corr(pd.Series(rk(b)), method='pearson'))}
    ta, tb = topk(a, 10), topk(b, 10)
    out['top10_overlap'] = float(len(ta & tb))
    out['top10_retention_pct'] = 100.0 * len(ta & tb) / 10.0
    b50 = topk(b, 50)
    out['topA_out_b50_pct'] = 100.0 * len(ta - b50) / 10.0
    for k in KS:
        out['ov@%d' % k] = float(len(topk(a, k) & topk(b, k)))
    return out


def boot(t, s, rng, n_rep):
    """重抽行 → score（与 observed 同向同算法）。"""
    M, k_n = ROW[(t, s)], KN[(t, s)]
    out = np.empty((n_rep, N))
    for b in range(n_rep):
        for j in range(N):
            k = int(k_n[j])
            if k == 0:
                out[b, j] = np.nan
            elif k == 1:
                out[b, j] = -M[j, 0]
            else:
                out[b, j] = -M[j, rng.integers(0, k, k)].mean()
    return out


OBS = {(t, s): observed(t, s) for t in ['beta2020', 'dcic2021'] for s in SHORT}

# ---- 方向自检：必须复现正文／图 4 的 1 / 3 / 2 ----
EXPECT = {'Liver_Acevedo': 1, 'Thyroid_ATC': 3, 'APL_Casorelli': 2}
chk = []
for s in SHORT:
    got = metrics(OBS[('beta2020', s)], OBS[('dcic2021', s)])['top10_overlap']
    chk.append({'signature': s, 'expected_top10_overlap': EXPECT[s],
                'observed_top10_overlap': got, 'match': int(got) == EXPECT[s]})
    print('方向自检 %-14s 期望 %d，实得 %.0f  %s'
          % (s, EXPECT[s], got, 'OK' if int(got) == EXPECT[s] else 'MISMATCH'))
if not all(c['match'] for c in chk):
    raise SystemExit('方向自检失败，拒绝写出结果')
pd.DataFrame(chk).to_csv(os.path.join(OUT, 'S25_direction_check.csv'), index=False)

rows = []
summary = []
for s in SHORT:
    obs = metrics(OBS[('beta2020', s)], OBS[('dcic2021', s)])
    rng = np.random.default_rng(SEED)
    W = {}
    for t in ['beta2020', 'dcic2021']:
        V = boot(t, s, rng, B)[:R]
        keys = list(obs.keys())
        M = {k: [] for k in keys}
        for i in range(R):
            for j in range(i + 1, R):
                mm = metrics(V[i], V[j])
                for k in keys:
                    M[k].append(mm[k])
        for k in keys:
            v = np.array(M[k], dtype=float)
            rows.append({'signature': s, 'quantity': k, 'scope': 'within_' + t,
                         'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
                         'p2_5': float(np.percentile(v, 2.5)),
                         'p97_5': float(np.percentile(v, 97.5)),
                         'n_replicates': B, 'n_pairs_used': R * (R - 1) // 2,
                         'pairs_independent': False})
        W[t] = M
    for k, v in obs.items():
        rows.append({'signature': s, 'quantity': k, 'scope': 'observed_cross_reference',
                     'mean': float(v), 'sd': np.nan, 'p2_5': np.nan, 'p97_5': np.nan,
                     'n_replicates': np.nan, 'n_pairs_used': np.nan,
                     'pairs_independent': np.nan})

    # Compare the observed value against EACH reference's own interval, one row
    # per reference. The previous version merged the two intervals into
    #   lo = max(beta_lo, dcic_lo),  hi = min(beta_hi, dcic_hi)
    # which is not a valid interval: for APL / spearman it produced
    # lo = 0.661576 > hi = 0.660530, an empty range that no value can satisfy.
    # The two references are independent single-matrix noise distributions, so
    # each is its own reference band; intersecting them fabricates a band that
    # neither matrix actually produces.
    for q in ['spearman', 'top10_overlap', 'top10_retention_pct',
              'topA_out_b50_pct'] + ['ov@%d' % k for k in KS]:
        for t in ['beta2020', 'dcic2021']:
            v = np.array(W[t][q], dtype=float)
            lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
            assert lo <= hi, ('empty interval', s, q, t, lo, hi)
            inside = lo <= obs[q] <= hi
            summary.append({'signature': s, 'quantity': q, 'reference': t,
                            'observed': obs[q], 'internal_lo': lo, 'internal_hi': hi,
                            'verdict': '区间内' if inside else (
                                '低于区间' if obs[q] < lo else '高于区间')})
            print('%-14s %-20s %-10s obs=%8.3f  内部 [%7.4f, %7.4f]  → %s'
                  % (s, q, t, obs[q], lo, hi,
                     '区间内' if inside else ('低于区间' if obs[q] < lo else '高于区间')))

D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, 'S25_disease_within_reference_stability.csv'), index=False)
pd.DataFrame(summary).to_csv(
    os.path.join(OUT, 'S25_inside_outside_verdict.csv'), index=False)
print('\n已写出 S25_disease_within_reference_stability.csv (%d 行)' % len(D))
print('已写出 S25_inside_outside_verdict.csv (%d 行)' % len(summary))
print('区间口径：生成 %d 个复本，区间由前 %d 个复本的 %d 对两两比较得到，'
      '这些对不相互独立，仅作经验分位数描述'
      % (B, R, R * (R - 1) // 2))
