"""CD_consolidate_from_prior.py
Consolidate the C / D pair already computed by the existing pipeline (G2CP controlled reference swap) into
clean deliverables under experiments/results/, noting their origin (existing controlled_swap / r94 outputs, not retrained here).
- D = G2CP controlled "reference swap" 2x2: the same beta-trained model (g2cp_v7_beta_ft.pt) scored against
      A=level5beta2020 (cache_beta_v2) and B=dcic2021 (cache_2021) respectively.
- C = per-drug stability for the same design (2,037 held-out drugs).
The pair shares the same P* / C*=64 cell lines / D*=20,370 drugs / H=2,037 drugs / 11,275 pairs. Gene axis = 978 landmarks.
"""
import json, csv, os
BASE = r'C:/Users/wkr20/WorkBuddy/2026-09-06-00-53-37'
RES = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/results'
os.makedirs(RES, exist_ok=True)

# ---------- D ----------
d = json.load(open(f'{BASE}/results/controlled_swap_level5beta2020_20260919_g978.json'))
metrics = ['pcc', 'dir', 'd1', 'd5', 'd10', 'd20', 'd50']
A = d['A_beta2020_raw']; B = d['B_dcic_raw']
# paired drug-level diff (from D json)
pd = d.get('paired', {})
pd_key = 'A_beta2020_raw_minus_B_dcic_raw'
pdv = pd.get(pd_key, {})

with open(f'{RES}/D_g2cp_refswap_2x2.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['metric', 'A_beta2020_est', 'A_ci_low', 'A_ci_high',
                'B_dcic2021_est', 'B_ci_low', 'B_ci_high',
                'paired_diff_est', 'paired_diff_ci_low', 'paired_diff_ci_high'])
    for m in metrics:
        a = A[m]; b = B[m]
        p = pdv.get(m, [None, None, None])
        w.writerow([m, f'{a[0]:.4f}', f'{a[1]:.4f}', f'{a[2]:.4f}',
                    f'{b[0]:.4f}', f'{b[1]:.4f}', f'{b[2]:.4f}',
                    (f'{p[0]:.4f}' if p[0] is not None else ''),
                    (f'{p[1]:.4f}' if p[1] is not None else ''),
                    (f'{p[2]:.4f}' if p[2] is not None else '')])
    des = d.get('design', {})
    w.writerow(['design.Cstar_cells', des.get('Cstar_cells'), '', '', '', '', '', '', '', ''])
    w.writerow(['design.Dstar_drugs', des.get('Dstar_drugs'), '', '', '', '', '', '', '', ''])
    w.writerow(['design.heldout_drugs', des.get('heldout_drugs'), '', '', '', '', '', '', '', ''])
    w.writerow(['design.eval_pairs', des.get('eval_pairs'), '', '', '', '', '', '', '', ''])
    w.writerow(['design.ckpt', des.get('ckpt'), '', '', '', '', '', '', '', ''])
    w.writerow(['gene_axis', '978 landmark', '', '', '', '', '', '', '', ''])
# D summary
dsum = {
    'experiment': 'D_g2cp_controlled_source_swap_2x2',
    'model_ckpt': des.get('ckpt'),
    'gene_axis': '978 landmark',
    'Cstar_cells': des.get('Cstar_cells'), 'Dstar_drugs': des.get('Dstar_drugs'),
    'heldout_drugs': des.get('heldout_drugs'), 'eval_pairs': des.get('eval_pairs'),
    'A_beta2020_PCC': A['pcc'], 'B_dcic2021_PCC': B['pcc'],
    'paired_dPCC_A_minus_B': pdv.get('pcc', 'see D_g2cp_refswap_2x2.csv'),
    'note': 'same beta-trained G2CP evaluated vs two references; reproduces S14.1 fixed-output (0.3680 vs 0.0724)',
    'source_json': 'results/controlled_swap_level5beta2020_20260919_g978.json',
}
json.dump(dsum, open(f'{RES}/D_summary.json', 'w'), indent=2, ensure_ascii=False)

# ---------- C ----------
c = json.load(open(f'{BASE}/results/r94_swap_perdrug_ranking_20260919_g978.json'))
pdlist = c['per_drug']  # 2037 rows: [rank_A, drug, pcc_A_beta2020, pcc_B_dcic2021, in_A_top10, in_B_top10, in_B_top50]
with open(f'{RES}/C_perdrug_refswap.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['rank_A', 'drug', 'pcc_A_beta2020', 'pcc_B_dcic2021',
                'in_A_top10', 'in_B_top10', 'in_B_top50'])
    for row in pdlist:
        w.writerow(row)
head = c.get('headline', {})
csum = {
    'experiment': 'C_perdrug_reference_swap_stability',
    'n_drugs': head.get('n_drugs'),
    'mean_pcc_A_beta2020': head.get('mean_pcc_A'),
    'mean_pcc_B_dcic2021': head.get('mean_pcc_B'),
    'paired_dPCC': head.get('paired_dPCC'),
    'paired_dPCC_ci95': head.get('paired_dPCC_ci95'),
    'spearman_rho_crossproduct': head.get('spearman_rho'),
    'top10_overlap_pct': head.get('R1_top10_overlap_pct'),
    'jaccard_top10': head.get('jaccard_top10'),
    'A_top10_fallen_out_of_B_top50_pct': head.get('R2_A_top10_fallen_out_of_B_top50_pct'),
    'gene_axis': '978 landmark',
    'note': 'G2CP per-drug ranking/top-k stability under reference swap; same P* as D',
    'source_json': 'results/r94_swap_perdrug_ranking_20260919_g978.json',
}
json.dump(csum, open(f'{RES}/C_summary.json', 'w'), indent=2, ensure_ascii=False)
print('C rows:', len(pdlist))
print('C summary:', json.dumps(csum, indent=2, ensure_ascii=False))
print('D summary:', json.dumps(dsum, indent=2, ensure_ascii=False))
