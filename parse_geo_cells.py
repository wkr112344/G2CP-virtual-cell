# -*- coding: utf-8 -*-
"""parse_geo_cells.py —— 解析 GEO 多类型细胞数据(iPSC/心肌/HUVEC/内皮/胰岛)并入训练。

数据集:
- GSE17579: hESC/hiPSC 未分化 vs 心肌(CM) + 胎儿/成人心脏, 18 样本, GPL6947
- GSE33622: hiPSC + KY02111 心肌分化小分子, 5 样本, GPL6244
- GSE50378: HUVEC + PPAR激动剂/缺氧, 4 样本, GPL570
- GSE20986: HUVEC vs 眼血管内皮(iris/retina/choroid), 12 样本, GPL570
- GSE50397: 人类胰岛(89 供体), 89 样本, GPL6244
输出: data/geo_cells/cells_samples.json  [{cell, pert, expr[12328], n_samp, coverage}]
"""
import gzip, json, os, sys, time, re
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
HVD = os.path.join(BASE, "data", "g2cp_cache_fullgene", "hvg.json")
PROBE_CACHE = os.path.join(BASE, "data", "geo_immune", "probe2sym.json")

def parse_series_matrix(f):
    fields = {}
    expr = []
    in_expr = False
    first_data = True
    with gzip.open(f, 'rt', errors='ignore') as fh:
        for l in fh:
            if l.startswith('!Sample_'):
                parts = l.rstrip().split('\t')
                fields.setdefault(parts[0].replace('!Sample_', ''), [p.strip('"') for p in parts[1:]])
            elif l.startswith('!series_matrix_table_begin'):
                in_expr = True
            elif in_expr and l.startswith('!series_matrix_table_end'):
                break
            elif in_expr:
                if first_data:
                    first_data = False
                    continue
                parts = l.rstrip().split('\t')
                expr.append((parts[0].strip('"'), [float(x) for x in parts[1:]]))
    return fields, expr

def load_probe_map(probe_ids):
    import mygene
    known = {}
    if os.path.isfile(PROBE_CACHE):
        known = json.load(open(PROBE_CACHE))
    todo = [p for p in probe_ids if p not in known]
    print(f'探针 {len(probe_ids)}, 已缓存 {len(known)}, 待查 {len(todo)}', flush=True)
    if todo:
        mg = mygene.MyGeneInfo()
        for i in range(0, len(todo), 1000):
            batch = todo[i:i + 1000]
            out = mg.querymany(batch, scopes='symbol,reporter', fields='symbol', verbose=False, species='human')
            for r in out:
                q = r.get('query')
                if q is not None and r.get('symbol'):
                    known[q] = r['symbol']
            print(f'  查询 {i+len(batch)}/{len(todo)}', flush=True)
            time.sleep(1)
        json.dump(known, open(PROBE_CACHE, 'w'))
    return known

def norm_expr(vals):
    a = np.array(vals, dtype=np.float64)
    sd = a.std()
    if sd < 1e-8:
        return np.zeros_like(a)
    return (a - a.mean()) / sd

# ---------------- 数据集定义: (文件名, 分类函数) ----------------
def cls_gse17579(t):
    if t.startswith('hiPSC_undiff'): return 'iPSC', 'IMM_REST'
    if t.startswith('hESC_undiff'): return 'ESC', 'IMM_REST'
    if t.startswith('hiPSC_CM'): return 'iPSC-CM', 'IMM_REST'
    if t.startswith('hESC_CM'): return 'ESC-CM', 'IMM_REST'
    if 'fetal heart' in t: return 'Heart-Fetal', 'IMM_REST'
    if 'adult heart' in t: return 'Heart-Adult', 'IMM_REST'
    return None, None

def cls_gse33622(t):
    if '0h' in t or 'DMSO' in t: return 'iPSC', 'IMM_REST'
    if 'KY02111' in t: return 'iPSC', 'IMM_KY02111'
    return None, None

def cls_gse50378(t):
    hyp = 'hypoxia' in t
    ago = 'agonist' in t
    if not hyp and not ago: return 'HUVEC', 'IMM_REST'
    if hyp and not ago: return 'HUVEC', 'IMM_HUVEC_Hypoxia'
    if not hyp and ago: return 'HUVEC', 'IMM_HUVEC_PPARa'
    return 'HUVEC', 'IMM_HUVEC_Hypoxia_PPARa'

def cls_gse20986(t):
    if t.startswith('huvec'): return 'HUVEC', 'IMM_REST'
    if t.startswith('iris'): return 'Iris-EC', 'IMM_REST'
    if t.startswith('retina'): return 'Retina-EC', 'IMM_REST'
    if t.startswith('choroid'): return 'Choroid-EC', 'IMM_REST'
    return None, None

def cls_gse50397(t):
    return 'Islet', 'IMM_REST'

def cls_gse25941(t):
    # 骨骼肌活检(年轻/老年)
    return 'Skeletal-Muscle', 'IMM_REST'

def cls_gse11917(t):
    # 冠状动脉平滑肌 CASMC: 对照 → 静息; 含 Dex/AA/BGP(钙化诱导) → 钙化
    if t.startswith('Control') or 'Media alone' in t:
        return 'SMC', 'IMM_REST'
    return 'SMC', 'IMM_SMC_Calcify'

def cls_gse62914(t):
    # iPSC→软骨
    if 'hiPSC' in t:
        return 'iPSC', 'IMM_REST'
    return 'Chondrocyte', 'IMM_REST'

def cls_gse21413(t):
    if 'differentiating' in t:
        return 'Keratinocyte', 'IMM_Ker_Diff'
    return 'Keratinocyte', 'IMM_REST'

def cls_gse53751(t):
    tl = t.lower()
    if 'vehicle' in tl:
        return 'Keratinocyte', 'IMM_REST'
    if 'interferon' in tl:
        return 'Keratinocyte', 'IMM_Ker_IFNg'
    if 'interleukin-4' in tl:
        return 'Keratinocyte', 'IMM_Ker_IL4'
    if 'interleukin-17' in tl:
        return 'Keratinocyte', 'IMM_Ker_IL17A'
    if 'interleukin-22' in tl:
        return 'Keratinocyte', 'IMM_Ker_IL22'
    if 'interleukin-6' in tl:
        return 'Keratinocyte', 'IMM_Ker_IL6'
    return None, None

DATASETS = [
    ('GSE17579', 'data/geo_cells/GSE17579_series_matrix.txt.gz', cls_gse17579),
    ('GSE33622', 'data/geo_cells/GSE33622_series_matrix.txt.gz', cls_gse33622),
    ('GSE50378', 'data/geo_cells/GSE50378_series_matrix.txt.gz', cls_gse50378),
    ('GSE20986', 'data/geo_cells/GSE20986_series_matrix.txt.gz', cls_gse20986),
    ('GSE50397', 'data/geo_cells/GSE50397_series_matrix.txt.gz', cls_gse50397),
    ('GSE25941', 'data/geo_cells/GSE25941_series_matrix.txt.gz', cls_gse25941),
    ('GSE11917', 'data/geo_cells/GSE11917_series_matrix.txt.gz', cls_gse11917),
    ('GSE62914', 'data/geo_cells/GSE62914_series_matrix.txt.gz', cls_gse62914),
    ('GSE21413', 'data/geo_cells/GSE21413_series_matrix.txt.gz', cls_gse21413),
    ('GSE53751', 'data/geo_cells/GSE53751_series_matrix.txt.gz', cls_gse53751),
]

def main():
    hvg = json.load(open(HVD))
    print(f'输出基因对齐: {len(hvg)} 个', flush=True)

    all_probe_ids = set()
    loaded = []
    for acc, path, cls_fn in DATASETS:
        f = os.path.join(BASE, path)
        if not os.path.isfile(f):
            print(f'缺少 {acc}', flush=True)
            continue
        fields, expr = parse_series_matrix(f)
        n_samp = len(fields.get('title', []))
        meta = [{'title': fields.get('title', ['']*n_samp)[i]} for i in range(n_samp)]
        loaded.append({'acc': acc, 'meta': meta, 'expr': expr, 'cls': cls_fn})
        for pid, _ in expr:
            all_probe_ids.add(pid)
        print(f'{acc}: {n_samp} 样本, {len(expr)} 探针', flush=True)

    probe_map = load_probe_map(list(all_probe_ids))
    print(f'探针映射: {len(probe_map)}/{len(all_probe_ids)}', flush=True)

    # 每个数据集: gene -> array(n_samp)
    ds_gene = []
    for ds in loaded:
        n_s = len(ds['meta'])
        gx = {}
        for pid, vals in ds['expr']:
            sym = probe_map.get(pid)
            if not sym or len(vals) != n_s:
                continue
            arr = np.array(vals, dtype=np.float64)
            if sym in gx:
                gx[sym] = (gx[sym] + arr) / 2
            else:
                gx[sym] = arr
        ds_gene.append(gx)
        print(f'  {ds["acc"]}: {len(gx)} 基因', flush=True)

    # 分组 (cell, pert) → [(ds_idx, sample_idx)]
    groups = {}
    for di, ds in enumerate(loaded):
        for si, m in enumerate(ds['meta']):
            cell, pert = ds['cls'](m['title'])
            if cell is None:
                print(f'  无法分类: {ds["acc"]} | {m["title"]}', flush=True)
                continue
            groups.setdefault((cell, pert), []).append((di, si))

    print(f'\n(细胞, 扰动) 组合: {len(groups)} 个')
    samples_out = []
    for (cell, pert), pairs in sorted(groups.items()):
        vec = np.zeros(len(hvg), dtype=np.float32)
        cnt = 0
        for gpos, g in enumerate(hvg):
            vals = [ds_gene[di][g][si] for di, si in pairs if g in ds_gene[di]]
            if vals:
                vec[gpos] = np.mean(vals)
                cnt += 1
        if cnt < 3000:
            print(f'  丢弃 {cell}|{pert}: 只覆盖 {cnt} 基因', flush=True)
            continue
        vec = norm_expr(vec)
        samples_out.append({'cell': cell, 'pert': pert, 'expr': vec.tolist(),
                            'n_samp': len(pairs), 'coverage': cnt})
        print(f'  {cell} | {pert}: {len(pairs)} 样本, {cnt} 基因覆盖', flush=True)

    os.makedirs(os.path.join(BASE, 'data', 'geo_cells'), exist_ok=True)
    out = os.path.join(BASE, 'data', 'geo_cells', 'cells_samples.json')
    json.dump(samples_out, open(out, 'w'))
    print(f'\n保存 {out}: {len(samples_out)} 个代表样本', flush=True)
    cells = sorted(set(s['cell'] for s in samples_out))
    perts = sorted(set(s['pert'] for s in samples_out))
    print(f'细胞系: {len(cells)} | 扰动: {len(perts)}', flush=True)
    print('细胞系:', cells, flush=True)
    print('扰动:', perts, flush=True)

if __name__ == '__main__':
    main()
