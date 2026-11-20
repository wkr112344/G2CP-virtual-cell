# -*- coding: utf-8 -*-
"""A02 — define ONE common 969-gene axis for both PRnet_beta and PRnet_dcic.

From the alignment QC: 978 beta landmarks -> 959 exact-match symbols in the dcic2021
GCTX, +10 recoverable via HGNC renames (ADCK3->COQ8A, ...), 9 genuinely absent.
We adopt the 969-gene universe as the SINGLE gene axis used by BOTH models, ordered
by the beta h5ad var order, expressed in dcic symbol space. The beta h5ad is relabelled
(old symbols -> dcic symbols) so both training sets share the identical axis.
Output: data/common_969_genes.tsv (+ .json)
"""
import h5py, scanpy as sc, os, json

BETA = r'C:/Users/wkr20/Desktop/Lincs_L1000.h5ad'
DCIC = r'C:/Users/wkr20/Desktop/cp_coeff_mat.gctx'
OUT  = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data'

# beta(old symbol) -> dcic(current symbol), well-established HGNC renames
RENAME = {
    'ADCK3':'COQ8A', 'PRUNE':'PRUNE1', 'KIAA0196':'WDFY3', 'KIAA1033':'CEP135',
    'TOMM70A':'TOMM70', 'HN1L':'JPT1', 'PAPD7':'TENT4A', 'TMEM5':'XYLT2',
    'HDGFRP3':'HDGFL3', 'FAM63A':'TTC13',
}
RESIDUAL_MISSING = ['ATP5S','FAM69A','IKBKAP','KIAA0907','LRRC16A','NARFL',
                    'SQRDL','TMEM110','TMEM2']

def main():
    os.makedirs(OUT, exist_ok=True)
    a = sc.read(BETA)
    beta_genes = list(a.var_names)
    with h5py.File(DCIC, 'r') as h:
        dcic_genes = set((x.decode() if isinstance(x, bytes) else x)
                         for x in h['0/META/ROW']['id'][:])
    rows = []          # (gene_dcic, action, beta_original)
    for g in beta_genes:
        if g in dcic_genes:
            rows.append((g, 'exact', g))
        elif g in RENAME and RENAME[g] in dcic_genes:
            rows.append((RENAME[g], 'rename_from:' + g, g))
    # sanity: residual must be exactly the known 9
    got = {r[2] for r in rows}
    still = [g for g in beta_genes if g not in got]
    assert set(still) == set(RESIDUAL_MISSING), ('unexpected missing: %s' % still)
    assert len(rows) == 969, ('expected 969, got %d' % len(rows))
    tsv = os.path.join(OUT, 'common_969_genes.tsv')
    with open(tsv, 'w', encoding='utf-8') as f:
        f.write('idx\tgene_dcic\tbeta_original\taction\n')
        for i, (canon, action, betaorig) in enumerate(rows):
            f.write(f'{i}\t{canon}\t{betaorig}\t{action}\n')
    json.dump({'n': len(rows), 'rows': rows, 'residual_missing': RESIDUAL_MISSING},
              open(os.path.join(OUT, 'common_969_genes.json'), 'w'), indent=1)
    print('wrote', len(rows), 'genes ->', tsv)
    print('residual 9 dropped:', RESIDUAL_MISSING)

if __name__ == '__main__':
    main()
