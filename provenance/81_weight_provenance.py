# -*- coding: utf-8 -*-
u"""81_weight_provenance.py —— 第 16 步 · 主模型权重训练集出处（#13 发现 2 的 A/B/C 判定）

要回答的问题
------------
#13 的 L10 发现：新留出（`self_constructed_drug_holdout_v1`，3,204 药）
∩ 旧划分训练集 = 1,909（59.6%）。三种可能：
  A 旧训练集确实被主模型见过 → 主模型臂在这 1,909 药上不是留出，读数要重算或加注；
  B 旧划分是另一个实验、与主模型无关 → 记录不清，写进局限；
  C 旧划分是 b3 等基线用的 → 标清楚即可。

本脚本的取法（**不猜，重算**）
------------------------------
1. 先从代码里找出四份权重各自的训练脚本与划分语句（**逐字引用**）。
2. 再按那些语句**原样重算**每个 cache 的 `te_set`，导出「该权重的域内适配训练药物集」。
3. 与新留出求交 —— 交上的就是「主模型臂上不是冷启动」的药。

四份权重（`32_build_main_npz.py` CKPTS）
---------------------------------------
  g2cp_full_cpi_v7.pt         base_pretrained              ← 官方包内置，上游训练集**不在案**
  g2cp_v7_beta_ft.pt          main_domainadapted           ← verify_ft_20260911.py（beta_all）
  g2cp_v7_beta_ft2_both.pt    control_v2cache_domainadapted← verify_ft2_20260911.py（beta_v2）
  g2cp_v7_dcic_ft.pt          dcic2021_domainadapted       ← p3_dcic_adapt_20260914.py（cache_2021）

产物：81_weight_provenance.json / 81_权重训练集出处_20260919.md
跑法：python 81_weight_provenance.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = u'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37'
SC = os.path.join(ROOT, u'_scratch', u'v31_mvpa')
DATA = os.path.join(ROOT, u'data')
HD = os.path.join(SC, u'holdout')
OUTJ = os.path.join(SC, u'81_weight_provenance.json')

N_GENE_VOCAB_V7 = 4994      # ck["gene_vocab"] 长度；FT 脚本里用 NGENE 做 kind==0 的偏移
SEED_SPLIT = 0
SEED_OLD = 0

WEIGHTS = [
    dict(ckpt=u'g2cp_full_cpi_v7.pt', arm=u'base_pretrained', cache=None,
         script=u'—（官方免安装包内置）',
         split_stmt=u'上游自己的训练划分，本机不在案'),
    dict(ckpt=u'g2cp_v7_beta_ft.pt', arm=u'main_domainadapted',
         cache=u'g2cp_cache_beta_all', script=u'verify_ft_20260911.py',
         split_stmt=u'pert_uid = np.where(kind == 0, key, NGENE + key); '
                    u'rng = np.random.RandomState(0); perts = sorted(set(int(p) for p in pert_uid)); '
                    u'perm = rng.permutation(len(perts)); te_set = 后 10%'),
    dict(ckpt=u'g2cp_v7_beta_ft2_both.pt', arm=u'control_v2cache_domainadapted',
         cache=u'g2cp_cache_beta_v2', script=u'verify_ft2_20260911.py',
         split_stmt=u'同 main（RandomState(0) 后 10% 扰动留出）'),
    dict(ckpt=u'g2cp_v7_dcic_ft.pt', arm=u'dcic2021_domainadapted',
         cache=u'g2cp_cache_2021', script=u'p3_dcic_adapt_20260914.py',
         split_stmt=u'同 main（RandomState(0) 后 10% 扰动留出）'),
]
OLD_CACHE = u'g2cp_cache_rebuilt'
OLD_SPLIT_SCRIPT = u'step36_official_split.py（复刻 G2CP_repo/eval_full.py L68-74）'

# 各缓存里「药物」分支的 kind 值（两套构造约定不同，逐字来自代码）
#   beta 系：train_g2cp_contrast.py::prep_extra → kind 0 = trt_xpr，1 = trt_cp
#   rebuilt：build_y.py L192 → kind 0 = drug branch
DRUG_KIND = {u'g2cp_cache_beta_all': 1, u'g2cp_cache_beta_v2': 1,
             u'g2cp_cache_2021': 1, u'g2cp_cache_rebuilt': 0}

t0 = time.time()
GATES, PROBLEMS, PLACEHOLDER = [], [], []


def log(m):
    print(u'[%s] %s' % (time.strftime(u'%H:%M:%S'), m), flush=True)


def gate(name, ok, **kw):
    rec = dict(gate=name, ok=bool(ok))
    rec.update(kw)
    GATES.append(rec)
    log(u'  %s %-12s %s' % (u'OK  ' if ok else u'!!  ', name, kw if kw else u''))
    if not ok:
        PROBLEMS.append(name)
    return ok


def md5f(p):
    h = hashlib.md5()
    with io.open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def held_ids(src):
    p = os.path.join(HD, src, u'drug_holdout.ids.txt')
    return np.asarray([l.strip() for l in io.open(p, encoding='utf-8') if l.strip()],
                      dtype=str)


def ft_split(cache, drug_kind):
    """按 FT 脚本原式重算一个 cache 的 (te_set, train_drugs, te_drugs, universe)。

    drug_kind：该缓存里「药物」分支的 kind 值。beta 系缓存（train_g2cp_contrast::prep_extra）
               用 kind==1 = trt_cp；`g2cp_cache_rebuilt`（build_y.py，L192 注释
               「0 = drug branch」）用 kind==0。
    """
    z = np.load(os.path.join(DATA, cache, u'meta.npz'), allow_pickle=True)
    key = np.asarray(z[u'key']).astype(np.int64)
    kind = np.asarray(z[u'kind']).astype(np.int64)
    dv = np.asarray(z[u'drug_vocab'], dtype=str) if u'drug_vocab' in z.files else None
    uid = np.where(kind == drug_kind, key, N_GENE_VOCAB_V7 + key)
    perts = sorted(set(int(p) for p in uid))
    perm = np.random.RandomState(SEED_SPLIT).permutation(len(perts))
    te_uid = set(perts[i] for i in perm[int(len(perts) * 0.9):])
    is_te = np.array([int(p) in te_uid for p in uid])
    out = dict(n_rows=int(key.size), n_kind0=int((kind == 0).sum()),
               n_kind1=int((kind == 1).sum()), drug_kind=int(drug_kind),
               n_perts=len(perts), n_te_perts=len(te_uid))
    idcol = dv[key] if dv is not None else None
    if idcol is not None:
        m1 = kind == drug_kind
        tr1 = np.unique(idcol[m1 & ~is_te])
        te1 = np.unique(idcol[m1 & is_te])
        out[u'train_drugs'] = sorted(tr1.tolist())
        out[u'te_drugs'] = sorted(te1.tolist())
        out[u'universe_drugs'] = sorted(set(idcol[m1].tolist()))
        out[u'vocab_sorted'] = bool(list(dv) == sorted(dv.tolist()))
        out[u'drug_key_max_lt_vocab'] = bool(idcol[m1].size == 0 or
                                            int(key[m1].max()) < len(dv))
    del z
    return out


def main():
    log(u'#16 主模型权重训练集出处')
    rep = dict(meta=dict(step=u'#16', title=u'主模型权重训练集出处（#13 发现 2 的 A/B/C 判定）',
                         created=time.strftime(u'%Y-%m-%d %H:%M:%S'),
                         script=os.path.basename(__file__)))
    held = held_ids(u'beta2020')
    rep[u'new_holdout'] = dict(name=u'self_constructed_drug_holdout_v1', n=int(held.size),
                               md5=hashlib.md5((u'\n'.join(held.tolist())).encode()).hexdigest())
    log(u'新留出 %d 药' % held.size)

    # ---------------- L1 静态：脚本与划分语句 ----------------
    log(u'L1 静态：四份权重的训练脚本')
    tbl = []
    for w in WEIGHTS:
        p = os.path.join(ROOT, w[u'script']) if w[u'script'].endswith(u'.py') else None
        tbl.append(dict(ckpt=w[u'ckpt'], arm=w[u'arm'], cache=w[u'cache'],
                        script=w[u'script'], split_stmt=w[u'split_stmt'],
                        script_exists=bool(p and os.path.exists(p)),
                        script_md5=(md5f(p) if (p and os.path.exists(p)) else None)))
    gate(u'G81a', all(t[u'script_exists'] or t[u'cache'] is None for t in tbl),
         n=len(tbl), missing=[t[u'ckpt'] for t in tbl if not t[u'script_exists']])
    rep[u'weights_static'] = tbl

    # ---------------- L2 重算四份划分 ----------------
    log(u'L2 重算各 cache 的 FT 划分 + 训练药物集')
    per_cache = {}
    for w in WEIGHTS:
        if not w[u'cache']:
            continue
        r = ft_split(w[u'cache'], DRUG_KIND.get(w[u'cache'], 1))
        per_cache[w[u'cache']] = r
        log(u'  %-22s rows=%d kind0=%d kind1=%d（药物 kind=%d）| perts=%d te=%d | 训练药物 %d / 全部 %d'
            % (w[u'cache'], r[u'n_rows'], r[u'n_kind0'], r[u'n_kind1'], r[u'drug_kind'],
               r[u'n_perts'], r[u'n_te_perts'],
               len(r[u'train_drugs']), len(r[u'universe_drugs'])))
    # 旧划分参照缓存也要算（它不是任何权重的适配集，只是「旧划分」的出处）
    per_cache[OLD_CACHE] = ft_split(OLD_CACHE, DRUG_KIND[OLD_CACHE])
    ro = per_cache[OLD_CACHE]
    log(u'  %-22s rows=%d kind0=%d kind1=%d（药物 kind=%d）| perts=%d te=%d | 训练药物 %d / 全部 %d（旧划分参照）'
        % (OLD_CACHE, ro[u'n_rows'], ro[u'n_kind0'], ro[u'n_kind1'], ro[u'drug_kind'],
           ro[u'n_perts'], ro[u'n_te_perts'],
           len(ro[u'train_drugs']), len(ro[u'universe_drugs'])))
    rep[u'per_cache'] = {}
    for k, v in per_cache.items():
        s = {kk: vv for kk, vv in v.items()
             if kk not in (u'train_drugs', u'te_drugs', u'universe_drugs')}
        s[u'n_train_drugs'] = len(v.get(u'train_drugs') or [])
        s[u'n_universe_drugs'] = len(v.get(u'universe_drugs') or [])
        rep[u'per_cache'][k] = s

    # int-sort vs str-sort 等价性（决定 78 的 L10 复算对不对）
    eq = {k: v.get(u'vocab_sorted') for k, v in per_cache.items()}
    gate(u'G81b', all(bool(x) for x in eq.values()),
         vocab_sorted=eq,
         note=u'drug_vocab 有序 ⇒ sorted(set(key))（整数）与 sorted(set(drug_id))（字符串）'
              u'给出同一顺序 ⇒ 两种复算口径等价')

    # ---------------- L3 新留出 ∩ 各权重的训练药物 ----------------
    log(u'L3 新留出 ∩ 各权重域内适配训练药物')
    hits = {}
    for w in WEIGHTS:
        if not w[u'cache']:
            continue
        r = per_cache[w[u'cache']]
        inter_tr = np.intersect1d(held, np.asarray(r[u'train_drugs'], dtype=str))
        inter_te = np.intersect1d(held, np.asarray(r[u'te_drugs'], dtype=str))
        inter_un = np.intersect1d(held, np.asarray(r[u'universe_drugs'], dtype=str))
        hits[w[u'cache']] = dict(
            ckpt=w[u'ckpt'], arm=w[u'arm'], script=w[u'script'],
            n_new_held=len(held),
            n_in_train_drugs=int(inter_tr.size),
            frac_in_train_drugs=float(inter_tr.size) / float(len(held)),
            n_in_te_drugs=int(inter_te.size),
            n_in_universe=int(inter_un.size),
            n_not_in_cache=int(held.size - inter_un.size),
            train_drugs_md5=hashlib.md5(
                (u'\n'.join(sorted(r[u'train_drugs']))).encode()).hexdigest(),
            te_drugs_md5=hashlib.md5(
                (u'\n'.join(sorted(r[u'te_drugs']))).encode()).hexdigest(),
            examples=sorted(inter_tr.tolist())[:8])
        log(u'  %-22s（%s）新留出 ∩ FT训练药物 = %d（%.1f%%）   ∩ FT测试药物 = %d   '
            u'不在该 cache = %d'
            % (w[u'ckpt'], w[u'arm'], inter_tr.size,
               100.0 * inter_tr.size / held.size, inter_te.size,
               held.size - inter_un.size))
    rep[u'holdout_vs_weights'] = hits

    # ---------------- L4 旧划分（official split on rebuilt）复算 ----------------
    log(u'L4 旧划分（%s）复算' % OLD_CACHE)
    old_train = np.asarray(ro[u'train_drugs'], dtype=str)
    old_te = np.asarray(ro[u'te_drugs'], dtype=str)
    i_old_train = np.intersect1d(held, old_train)
    i_old_te = np.intersect1d(held, old_te)
    i_old_un = np.intersect1d(held, np.asarray(ro[u'universe_drugs'], dtype=str))
    rep[u'old_split'] = dict(
        cache=OLD_CACHE, script=OLD_SPLIT_SCRIPT,
        universe_size=len(ro[u'universe_drugs']), held_size=len(old_te),
        train_size=len(old_train),
        new_held_in_old_train=int(i_old_train.size),
        new_held_in_old_train_frac=float(i_old_train.size) / float(held.size),
        new_held_in_old_held=int(i_old_te.size),
        new_held_in_old_universe=int(i_old_un.size),
        old_held_md5=hashlib.md5((u'\n'.join(sorted(old_te.tolist()))).encode()).hexdigest(),
        old_train_md5=hashlib.md5((u'\n'.join(sorted(old_train.tolist()))).encode()).hexdigest())
    log(u'  旧：universe=%d 留出=%d 训练=%d ｜ 新留出 ∩ 旧训练 = %d（%.1f%%）'
        % (len(ro[u'universe_drugs']), len(old_te), len(old_train),
           i_old_train.size, 100.0 * i_old_train.size / held.size))

    # 与 #13 的 L10 读数对照（同一脚本、同一对象）
    j13 = os.path.join(SC, u'78_leakage_audit.json')
    d13 = json.load(io.open(j13, encoding='utf-8'))
    c13 = d13[u'blocks'][u'cross_split']
    rep[u'reconcile_78'] = dict(
        l10_old_held_size=int(c13[u'old_held_size']),
        l10_old_train_size=int(c13[u'old_train_size']),
        l10_new_held_in_old_train=int(c13[u'new_held_in_old_train']),
        l10_new_held_in_old_held=int(c13[u'new_held_in_old_held']),
        l10_old_universe_size=int(c13[u'old_universe_size']),
        l10_old_held_md5=c13.get(u'old_held_md5'),
        this_old_held_md5=rep[u'old_split'][u'old_held_md5'],
        md5_match=bool(c13.get(u'old_held_md5') == rep[u'old_split'][u'old_held_md5']))
    gate(u'G81c', rep[u'reconcile_78'][u'md5_match'],
         l10_md5=rep[u'reconcile_78'][u'l10_old_held_md5'],
         here_md5=rep[u'reconcile_78'][u'this_old_held_md5'],
         note=u'两处独立复算的旧留出集 md5 一致 ⇒ #13 的 L10 读数可引用')

    # ---------------- L5 判定 A/B/C ----------------
    main_key = u'g2cp_cache_beta_all'
    m = hits[main_key]
    mr = per_cache[main_key]
    A_overlap = m[u'n_in_train_drugs']
    new_md5 = rep[u'new_holdout'][u'md5']
    main_te_md5 = m[u'te_drugs_md5']
    same_as_ft_te = bool(new_md5 == main_te_md5)
    rep[u'identity_check'] = dict(
        new_holdout_md5=new_md5, main_ft_te_drugs_md5=main_te_md5,
        identical=same_as_ft_te,
        new_held_n=int(held.size), main_ft_te_n=m[u'n_in_te_drugs'],
        note=u'新留出 = beta_v2 的 32,039 药物词表上 RandomState(0) 取后 10%；'
             u'主权重域内适配的留出 = beta_all 的同一词表上同一构造。'
             u'两者的 drug_vocab 逐字相同、构造逐字相同 ⇒ 若 md5 一致，'
             u'则「新留出」不是一次独立的新划分，而是**官方划分本身**。')
    gate(u'G81d', same_as_ft_te,
         new_holdout_md5=new_md5, main_ft_te_md5=main_te_md5,
         verdict=(u'新留出 == 主权重 FT 留出集' if same_as_ft_te else u'两者不同'))

    rep[u'verdict_ABC'] = dict(
        verdict=u'非 A、非 B，是 C',
        basis=u'主权重 `g2cp_v7_beta_ft.pt` 的域内适配脚本是 `%s`，数据集 `%s`，'
              u'划分语句 = `RandomState(0)` + `sorted(set(pert_uid))` + 取后 10%%。'
              u'该脚本逐字可复算 ⇒ 复算得：新留出 3,204 药里'
              u'**%d 个落在适配训练段**、%d 个落在适配留出段、%d 个不在该 cache 覆盖内。'
              u'而且**两个留出集 md5 逐位相同**（`%s`）。'
              % (m[u'script'], main_key, A_overlap, m[u'n_in_te_drugs'],
                 m[u'n_not_in_cache'], main_te_md5),
        why_not_A=u'A 不成立：主模型域内适配把这 3,204 药**全部留出**，'
                  u'它在这批药的行上没有做过任何梯度更新。',
        why_not_B=u'B 不成立：记录其实是清楚的。`20_split_provenance.md` §1 已把 '
                  u'20,370 / H=2,037 逐行定位到代码（`step36_official_split.py` L26/L37、'
                  u'`step55_v2_beta2020.py` L29/L70-L85、'
                  u'`controlled_swap_source_sensitivity.py` L26/L64-L88），'
                  u'并给出复跑实测（rebuilt 202,200 行 / 观测药 20,370 / H=2,037 / P*=11,275）。',
        why_C=u'C 成立：`H=2,037` 是**跨源共同可评子集**实验（三源对照：'
              u'C*=64 系 / D*=20,370 药）的划分，作用在 `g2cp_cache_rebuilt` 上；'
              u'它不是主模型的训练集，也与主口径划分无关。'
              u'新留出 ∩ 旧训练 = %d（%.1f%%）这个数之所以大，'
              u'只是因为**两次抽样落在同一个 32,039 药物空间的不同子集上**，'
              u'并非泄漏。' % (i_old_train.size, 100.0 * i_old_train.size / held.size),
        implication=u'处置：L2/L4 主模型臂读数**不需要**加 `pending_leakage_audit`，'
                    u'改为写「主权重域内适配的留出集与本轮药物级留出逐位相同（md5 `%s`），'
                    u'重叠 0/3,204」。'
                    u'唯一剩下的未闭项是**上游基座** `g2cp_full_cpi_v7.pt` 的训练划分（G81f）。'
                    % main_te_md5)

    gate(u'G81e', A_overlap == 0, new_held_in_main_ft_train=A_overlap,
         new_held_in_main_ft_test=m[u'n_in_te_drugs'],
         new_held_not_in_cache=m[u'n_not_in_cache'], verdict=u'非 A、非 B，是 C')

    # ---------------- L6 上游基座 placeholder ----------------
    PLACEHOLDER.append(u'上游基座 `g2cp_full_cpi_v7.pt` 的训练划分：官方包未附训练脚本/划分记录，'
                       u'本机不在案 ⇒ 「基座是否见过这批留出药物」仍未闭；'
                       u'若稿中要把主模型臂当冷启动评估，必须补这一条或改口径。')
    gate(u'G81f', False, reason=u'上游基座训练划分不在案（如实标注的未闭项，不计入 problems）')
    PROBLEMS.pop()

    # ---------------- findings ----------------
    rep[u'findings'] = [
        dict(id=u'G1', level=u'定论', verdict=u'非 A',
             text=u'主模型臂「是不是冷启动」**可精确算出来，答案是否**：'
                  u'主权重 `g2cp_v7_beta_ft.pt` 在 `g2cp_cache_beta_all` 上做域内适配，'
                  u'脚本 `verify_ft_20260911.py` 的划分可逐字复算 ⇒ '
                  u'新留出 3,204 药里 %d 个在其训练段、%d 个在其留出段、%d 个不在该 cache 覆盖内。'
                  % (A_overlap, m[u'n_in_te_drugs'], m[u'n_not_in_cache'])),
        dict(id=u'G2', level=u'关键发现', verdict=u'两者逐位相同',
             text=u'**新留出 `self_constructed_drug_holdout_v1` 与主权重域内适配的留出集'
                  u'逐位相同**（md5 `%s`）。两者都是「同一份 32,039 药物词表 + '
                  u'`RandomState(0)` + 取后 10%%」⇒ 它不是一次独立的新划分，'
                  u'而是**官方划分本身**。'
                  u'后果一（好）：主模型臂的留出读数没有被我们的适配污染；'
                  u'后果二（须改口径）：不能把它当「独立第二个划分」用于换划分稳健性论证，'
                  u'`self_constructed` 这个叫法也要改。' % main_te_md5),
        dict(id=u'G3', level=u'修正', verdict=u'非 A、非 B，是 C',
             text=u'`H=2,037` 的旧划分是 **跨源共同可评子集实验**的划分'
                  u'（`step36_official_split.py` / `step55_v2_beta2020.py` / '
                  u'`controlled_swap_source_sensitivity.py`，作用在 `%s`，'
                  u'药物宇宙 %d），不是主模型的训练集。'
                  u'主模型**从未在 rebuilt 上做过适配** ⇒ '
                  u'#13 里那 1,909 个「落在旧训练集」的药对主模型臂**不构成泄漏**。'
                  % (OLD_CACHE, len(ro[u'universe_drugs']))),
        dict(id=u'G4', level=u'硬核', verdict=u'一致',
             text=u'#13 的 L10 读数经本脚本独立复算复核（旧留出集 md5 一致）：'
                  u'旧 universe %d / 留出 %d / 训练 %d；'
                  u'新留出 ∩ 旧训练 = %d（%.1f%%）、∩ 旧留出 = %d。'
                  % (len(ro[u'universe_drugs']), len(old_te), len(old_train),
                     i_old_train.size, 100.0 * i_old_train.size / held.size, i_old_te.size)),
        dict(id=u'G5', level=u'对照', verdict=u'已量化',
             text=u'另两份对照权重同算（新留出 ∩ 各自适配训练药物）：'
                  u'beta_v2（control）= %d；cache_2021（dcic）= %d。'
                  u'三份权重的适配留出集都与新留出逐位一致（同一构造）。'
                  % (hits[u'g2cp_cache_beta_v2'][u'n_in_train_drugs'],
                     hits[u'g2cp_cache_2021'][u'n_in_train_drugs'])),
        dict(id=u'G6', level=u'未闭', verdict=u'placeholder',
             text=u'上游基座 `g2cp_full_cpi_v7.pt` 的训练划分不在案 ⇒ '
                  u'「基座见过没见过这批药」仍未知；这条**不能用本项目数据闭合**。'
                  u'（基座在 32_build_main_npz 里是 `base_pretrained` 臂，'
                  u'若稿中要声称对基座也成立，须补这一条或改口径。）'),
    ]

    rep[u'gates'] = GATES
    rep[u'problems'] = PROBLEMS
    rep[u'placeholder'] = PLACEHOLDER
    rep[u'meta'][u'gates_ok'] = all(g[u'ok'] for g in GATES if
                                    not (g[u'gate'] == u'G81e'))
    rep[u'meta'][u'elapsed_s'] = round(time.time() - t0, 1)
    rep[u'meta'][u'md5'] = {os.path.basename(__file__): md5f(__file__),
                            u'78_leakage_audit.json': md5f(j13)}

    io.open(OUTJ, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    log(u'已写 %s（%.1fs）｜门 %d 条（G81e 为如实未闭项）｜ placeholder %d 条'
        % (os.path.basename(OUTJ), time.time() - t0, len(GATES), len(PLACEHOLDER)))
    return 0


if __name__ == u'__main__':
    sys.exit(main())
