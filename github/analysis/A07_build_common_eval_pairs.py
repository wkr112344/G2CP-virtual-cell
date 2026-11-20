# -*- coding: utf-8 -*-
"""A07 — recover the EXACT G2CP S25 common evaluation-pair list (11,275 pairs).

Faithful port of step 1 of p4c_run_A_2x2.py (the manuscript 2x2 runner):

    C*    = cells common to g2cp_cache_rebuilt and g2cp_cache_2021
    D*    = drugs common to the two products
    H     = 10% of D* held out at the DRUG level, RandomState(0) permutation
            -> 2,037 held-out drugs
    P*    = (cell, drug) pairs present in BOTH rebuilt and 2021,
            with cell in C* and drug in H
    assert (64, 20370, 2037, 11275) == (|C*|, |D*|, |H|, |P*|)

The (cell, drug) pair LIST is gene-axis agnostic; PRnet "full set A" uses the 969
axis but reuses this identical pair set so the eval is comparable to S25.

We then verify every P* pair is also present in the two EVAL caches
(g2cp_cache_beta_v2 for the beta eval arm, g2cp_cache_2021 for the dcic arm)
so downstream PRnet eval can actually look up an observed Y for each pair.

Output:
    data/common_eval_pairs.csv   (idx, cell, drug, in_beta_v2, in_2021)
    data/common_eval_pairs.json  (design counts + provenance)
"""
import json, os, time
import numpy as np

BASE = r'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37'
D_REB = os.path.join(BASE, 'data', 'g2cp_cache_rebuilt')     # design: pairs
D_B2  = os.path.join(BASE, 'data', 'g2cp_cache_beta_v2')     # beta eval arm
D_21  = os.path.join(BASE, 'data', 'g2cp_cache_2021')        # dcic eval arm
OUT   = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data'

def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)

def load_pairs(d):
    pm = np.load(os.path.join(d, 'pairs_meta.npz'), allow_pickle=True)
    cell = [str(x) for x in pm['cell_name']]
    drug = [str(x) for x in pm['drug_id']]
    return set(zip(cell, drug))

def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # ---- design set from rebuilt + 2021 (verbatim p4c step 1) ----
    pmB = np.load(os.path.join(D_21, 'pairs_meta.npz'), allow_pickle=True)
    b_cell = [str(x) for x in pmB['cell_name']]
    b_drug = [str(x) for x in pmB['drug_id']]
    mR = np.load(os.path.join(D_REB, 'meta.npz'), allow_pickle=True)
    r_cl = [str(x) for x in mR['cl_names']]
    r_dv = [str(x) for x in mR['drug_vocab']]
    r_cn = [r_cl[int(c)] for c in mR['cell'].astype(np.int64)]
    r_dn = [r_dv[int(k)] for k in mR['key'].astype(np.int64)]

    Cstar = sorted(set(r_cn) & set(b_cell))
    Dstar = sorted(set(r_dn) & set(b_drug))
    perts = sorted(Dstar)
    n_te = int(len(perts) * 0.9)
    perm = np.random.RandomState(0).permutation(len(perts))
    H = set(perts[i] for i in perm[n_te:])
    Cs_set = set(Cstar)
    r_pairs = set(zip(r_cn, r_dn))
    b_pairs = set(zip(b_cell, b_drug))
    Pstar = sorted(p for p in (r_pairs & b_pairs) if p[0] in Cs_set and p[1] in H)

    assert (len(Cstar), len(Dstar), len(H), len(Pstar)) == (64, 20370, 2037, 11275), \
        'design != archived S25 (64,20370,2037,11275); got %s' % (len(Cstar), len(Dstar), len(H), len(Pstar))
    log('design C*=%d D*=%d H=%d P*=%d (== S25)' % (len(Cstar), len(Dstar), len(H), len(Pstar)))

    # ---- verify each P* pair exists in BOTH eval caches ----
    b2_pairs = load_pairs(D_B2)
    d21_pairs = load_pairs(D_21)
    in_b2 = [p in b2_pairs for p in Pstar]
    in_21 = [p in d21_pairs for p in Pstar]
    n_in_b2 = sum(in_b2); n_in_21 = sum(in_21)
    log('eval-arm coverage: in beta_v2=%d/%d  in 2021=%d/%d' % (n_in_b2, len(Pstar), n_in_21, len(Pstar)))
    assert n_in_b2 == len(Pstar) and n_in_21 == len(Pstar), \
        'some P* pairs missing from an eval cache (beta_v2=%d, 2021=%d of %d)' % (n_in_b2, n_in_21, len(Pstar))

    # ---- write CSV + JSON ----
    csvp = os.path.join(OUT, 'common_eval_pairs.csv')
    with open(csvp, 'w', encoding='utf-8') as f:
        f.write('idx\tcell\tdrug\tin_beta_v2\tin_2021\n')
        for i, (c, dg) in enumerate(Pstar):
            f.write('%d\t%s\t%s\t%d\t%d\n' % (i, c, dg, 1 if in_b2[i] else 0, 1 if in_21[i] else 0))
    log('wrote %s (%d pairs)' % (csvp, len(Pstar)))

    meta = dict(
        schema='g2cp_S25_common_eval_pairs',
        recovered_from='p4c_run_A_2x2.py step 1 (manuscript 2x2 design)',
        seed=0, test_frac=0.10,
        n_cells_Cstar=len(Cstar), n_drugs_Dstar=len(Dstar),
        n_heldout_drugs_H=len(H), n_eval_pairs_Pstar=len(Pstar),
        eval_arm_beta='g2cp_cache_beta_v2', eval_arm_dcic='g2cp_cache_2021',
        coverage_in_beta_v2=n_in_b2, coverage_in_2021=n_in_21,
        note='gene-axis agnostic; PRnet full set A reuses this identical 11,275 (cell,drug) set on the 969 axis.',
        elapsed_s=round(time.time() - t0, 1),
    )
    json.dump(meta, open(os.path.join(OUT, 'common_eval_pairs.json'), 'w'), indent=1)
    log('wrote common_eval_pairs.json | %.1fs' % (time.time() - t0))

if __name__ == '__main__':
    main()
