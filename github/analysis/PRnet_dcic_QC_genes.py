# -*- coding: utf-8 -*-
"""Augment PRnet_dcic_alignment_QC.json with gene-symbol reconciliation.

The beta h5ad (978 landmarks) and the dcic2021 GCTX (12327 genes) use different
HGNC symbol snapshots: 19 of the 978 are missing as exact symbols in dcic.
10 are recoverable via known LINCS-era -> current renames; 9 are genuinely absent
(dcic processing gene-drop). Produce a single JOINT gene universe for both models.
"""
import h5py, scanpy as sc, json

BETA = r'C:/Users/wkr20/Desktop/Lincs_L1000.h5ad'
DCIC = r'C:/Users/wkr20/Desktop/cp_coeff_mat.gctx'
OUT  = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results/PRnet_dcic_alignment_QC.json'

# beta(old) -> dcic(current) where rename is well established
RENAME = {
    'ADCK3':'COQ8A', 'PRUNE':'PRUNE1', 'KIAA0196':'WDFY3', 'KIAA1033':'CEP135',
    'TOMM70A':'TOMM70', 'HN1L':'JPT1', 'PAPD7':'TENT4A', 'TMEM5':'XYLT2',
    'HDGFRP3':'HDGFL3', 'FAM63A':'TTC13',
}

a = sc.read(BETA)
beta_genes = list(a.var_names)
with h5py.File(DCIC, 'r') as h:
    dcic_genes = set((x.decode() if isinstance(x, bytes) else x)
                     for x in h['0/META/ROW']['id'][:])

# 1) exact intersection
exact = [g for g in beta_genes if g in dcic_genes]
# 2) recoverable via rename
recovered = [(old, RENAME[old]) for old in RENAME if RENAME[old] in dcic_genes]
recovered_set = {new for _, new in recovered}
# 3) residual missing
renamed_beta = set(RENAME.keys())
residual = [g for g in beta_genes
            if g not in dcic_genes and g not in renamed_beta]

joint = sorted(set(exact) | recovered_set)   # union, fixed deterministic order
# order joint universe by beta h5ad var order (so PRnet_beta keeps natural order)
joint_ordered = [g for g in beta_genes if g in set(joint)] + \
                [g for g in joint if g not in set(beta_genes)]

recon = {
    'exact_intersection_n': len(exact),
    'recovered_via_rename': recovered,
    'recovered_n': len(recovered),
    'residual_missing': sorted(residual),
    'residual_missing_n': len(residual),
    'joint_universe_n': len(joint_ordered),
    'recommended': 'Use joint_universe (sorted by beta var order) for BOTH PRnet_beta and '
                   'PRnet_dcic so the gene axis is identical; x_dimension=969 not 978. '
                   'Conservative alt: exact_intersection only (959 genes, no alias guesswork).',
    'dcic_gene_total': len(dcic_genes),
}

R = json.load(open(OUT))
R['gene_check']['exact_intersection'] = len(exact)
R['gene_check']['recovered_via_rename'] = recovered
R['gene_check']['residual_missing'] = sorted(residual)
R['gene_reconciliation'] = recon
R['qc_verdict']['gene_978_complete'] = False
R['qc_verdict']['gene_joint_universe_n'] = len(joint_ordered)
R['qc_verdict']['recommendation'] = (
    'INSTANCE alignment is viable: 125,765 matched (cell,pert_id,dose,time) instances '
    '(125,765/234,912 = 53.5% of beta trt_cp; 115,302 matched (cell,drug) pairs ~= 99% of beta pairs, '
    'covering essentially all G2CP S25 11,275 pairs). '
    'BUT two fixes are required before training: '
    '(a) GENE AXIS: only 959/978 landmarks match exactly; 10 recoverable via HGNC rename, '
    '9 genuinely absent in dcic -> define ONE joint 969-gene universe (or conservative 959) used identically for both models; '
    '(b) CONTROLS: dcic GCTX is treatment-only (0 controls) -> PRnet_dcic cannot be trained locally. '
    'Achievable now: train PRnet_beta on the joint universe and evaluate against BOTH beta2020-true and dcic2021-true '
    '(cells (beta,beta) and (beta,dcic)) to test whether the reference interaction is G2CP-specific. '
    'The full 4-cell 2x2 needs a dcic2021 control/baseline matrix (currently missing).')

json.dump(R, open(OUT, 'w'), indent=2, ensure_ascii=False)
print(json.dumps(recon, indent=1, ensure_ascii=False))
print('\njoint_universe_n =', len(joint_ordered))
print('updated', OUT)
