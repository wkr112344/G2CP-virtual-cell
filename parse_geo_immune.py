# -*- coding: utf-8 -*-
"""parse_geo_immune.py —— 解析 GEO 免疫细胞数据(GSE22886 + GSE60235)并入全基因缓存。

- GSE22886: 12 种免疫细胞(CD8T/CD4T/B/DC/Mono/CD14/NK/Neutrophil/MemoryT/PlasmaCell), 228 样本, 双平台 GPL96(U133A)+GPL97(U133B)
- GSE60235: CD4+ T 细胞激活(anti-CD3/CD28/IFNb/Th17), 75 样本, GPL6244
- 输出: data/geo_immune/immune_samples.json
  [{cell: 'PBMC-CD8T', pert: 'IMM_REST' | 'IMM_DC_LPS' | ..., expr: [12328 维对齐到 hvg 顺序]}]
"""
import gzip, json, os, sys, time, re
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
HVD = os.path.join(BASE, "data", "g2cp_cache_fullgene", "hvg.json")

def parse_series_matrix(f):
    """返回 (samples_meta, expr_rows) —— expr_rows: list of (probe_id, [vals])"""
    with gzip.open(f, 'rt', errors='ignore') as fh:
        lines = fh.readlines()
    fields = {}
    expr = []
    in_expr = False
    first_data = True
    for l in lines:
        if l.startswith('!Sample_'):
            parts = l.rstrip().split('\t')
            fields.setdefault(parts[0].replace('!Sample_', ''), [p.strip('"') for p in parts[1:]])
        elif l.startswith('!series_matrix_table_begin'):
            in_expr = True
        elif in_expr and l.startswith('!series_matrix_table_end'):
            break
        elif in_expr:
            if first_data:  # 跳过表头行(ID_REF + GSM 号)
                first_data = False
                continue
            parts = l.rstrip().split('\t')
            expr.append((parts[0].strip('"'), [float(x) for x in parts[1:]]))
    return fields, expr

def load_probe_map(probe_ids, cache='data/geo_immune/probe2sym.json'):
    """探针 → 基因符号, 用 mygene 批量查询, 结果缓存"""
    import mygene
    os.makedirs('data/geo_immune', exist_ok=True)
    cache_p = os.path.join(BASE, cache)
    known = {}
    if os.path.isfile(cache_p):
        known = json.load(open(cache_p))
    todo = [p for p in probe_ids if p not in known]
    print(f'探针总数 {len(probe_ids)}, 已缓存 {len(known)}, 待查 {len(todo)}', flush=True)
    if todo:
        mg = mygene.MyGeneInfo()
        for i in range(0, len(todo), 1000):
            batch = todo[i:i+1000]
            out = mg.querymany(batch, scopes='symbol,reporter', fields='symbol', verbose=False, species='human')
            for r in out:
                q = r.get('query')
                if q is not None and r.get('symbol'):
                    known[q] = r['symbol']
            print(f'  查询 {i+len(batch)}/{len(todo)}', flush=True)
            time.sleep(1)
        json.dump(known, open(cache_p, 'w'))
    return known

def norm_expr(vals):
    """表达值 → 标准化(与 LINCS z-score 口径近似): per-sample zscore"""
    a = np.array(vals, dtype=np.float64)
    sd = a.std()
    if sd < 1e-8:
        return np.zeros_like(a)
    return (a - a.mean()) / sd

def main():
    hvg = json.load(open(HVD))
    hvg_set = set(hvg)
    print(f'输出基因对齐: {len(hvg)} 个 (hvg.json)', flush=True)

    # 按平台读入
    platforms = []   # {name, meta, expr}
    probe_ids = set()

    for plat, tag in [('GSE22886-GPL96_series_matrix.txt.gz', 'A'),
                      ('GSE22886-GPL97_series_matrix.txt.gz', 'B')]:
        f = os.path.join(BASE, 'data', 'geo_immune', plat)
        if not os.path.isfile(f):
            print(f'缺少 {plat}', flush=True)
            continue
        fields, expr = parse_series_matrix(f)
        n_samp = len(fields.get('title', []))
        meta = [{'title': fields.get('title', ['']*n_samp)[i],
                 'source': fields.get('source_name_ch1', ['']*n_samp)[i],
                 'platform': tag} for i in range(n_samp)]
        platforms.append({'name': plat, 'meta': meta, 'expr': expr})
        for pid, vals in expr:
            probe_ids.add(pid)
        print(f'{plat}: {n_samp} 样本, {len(expr)} 探针', flush=True)

    f = os.path.join(BASE, 'data', 'geo_immune', 'GSE60235_series_matrix.txt.gz')
    if os.path.isfile(f):
        fields, expr = parse_series_matrix(f)
        n_samp = len(fields.get('title', []))
        meta = [{'title': fields.get('title', ['']*n_samp)[i],
                 'source': 'CD4+ T cells from PBMC', 'platform': 'C'} for i in range(n_samp)]
        platforms.append({'name': 'GSE60235', 'meta': meta, 'expr': expr})
        for pid, vals in expr:
            probe_ids.add(pid)
        print(f'GSE60235: {n_samp} 样本, {len(expr)} 探针', flush=True)

    # 探针映射
    probe_map = load_probe_map(list(probe_ids))
    print(f'探针映射: {len(probe_map)}/{len(probe_ids)} 命中符号', flush=True)

    # 每个平台: gene -> np.array(该平台样本数)
    plat_gene = []
    for p in platforms:
        n_s = len(p['meta'])
        gx = {}
        for pid, vals in p['expr']:
            sym = probe_map.get(pid)
            if not sym or len(vals) != n_s:
                continue
            arr = np.array(vals, dtype=np.float64)
            if sym in gx:
                gx[sym] = (gx[sym] * 0.5 + arr * 0.5)  # 多探针取均值
            else:
                gx[sym] = arr
        plat_gene.append(gx)
        print(f'  {p["name"]}: {len(gx)} 基因', flush=True)

    # GSE22886: A+B 同批样本合并; GSE60235 独立
    sample_meta, sample_vec = [], []   # sample_vec: gene -> float
    # GSE22886: 前两个平台(同批 114 样本)
    n_ab = len(platforms[0]['meta'])
    for i in range(n_ab):
        gv = {}
        for gx in plat_gene[:2]:
            for g, arr in gx.items():
                if g in gv:
                    gv[g] = (gv[g] + arr[i]) / 2
                else:
                    gv[g] = arr[i]
        sample_meta.append(platforms[0]['meta'][i])
        sample_vec.append(gv)
    # GSE60235
    if len(platforms) > 2:
        n_c = len(platforms[2]['meta'])
        for i in range(n_c):
            gv = {g: arr[i] for g, arr in plat_gene[2].items()}
            sample_meta.append(platforms[2]['meta'][i])
            sample_vec.append(gv)

    print(f'合并后样本: {len(sample_meta)}', flush=True)

    def classify_title(t):
        """title → (cell, pert)"""
        t = t.split(' [')[0]
        if t.startswith('CD8Tcell'):
            return 'PBMC-CD8T', 'IMM_REST'
        if t.startswith('CD4Tcell-N0'):
            return 'PBMC-CD4T', 'IMM_REST'
        if t.startswith('CD4Tcell-Th1'):
            return 'PBMC-CD4T', 'IMM_CD4T_Th1'
        if t.startswith('CD4Tcell-Th2'):
            return 'PBMC-CD4T', 'IMM_CD4T_Th2'
        if t.startswith('Bcell'):
            return 'PBMC-Bcell', 'IMM_REST'
        if t.startswith('DendriticCell-Control'):
            return 'PBMC-DC', 'IMM_REST'
        if t.startswith('DendriticCell-LPS'):
            return 'PBMC-DC', 'IMM_DC_LPS'
        if t.startswith('MemoryTcell-RO-activated'):
            return 'PBMC-MemoryT', 'IMM_MemT_Act'
        if t.startswith('MemoryTcell-RO-unactivated'):
            return 'PBMC-MemoryT', 'IMM_REST'
        if t.startswith('Monocyte'):
            m = re.search(r'Day(\d+)', t)
            day = m.group(1) if m else '0'
            if day == '0':
                return 'PBMC-Mono', 'IMM_REST'
            return 'PBMC-Mono', f'IMM_Mono_Day{day}'
        if t.startswith('CD14'):
            return 'PBMC-CD14', 'IMM_REST'
        if t.startswith('NK'):
            return 'PBMC-NK', 'IMM_REST'
        if t.startswith('Neutrophil'):
            return 'PBMC-Neutrophil', 'IMM_REST'
        if t.startswith('PlasmaCell'):
            if 'BM' in t or 'bone' in t.lower():
                return 'BM-PlasmaCell', 'IMM_REST'
            return 'PBMC-PlasmaCell', 'IMM_REST'
        # GSE60235
        if t.startswith('IGTB'):
            if 'Unstimulated' in t:
                return 'PBMC-CD4T', 'IMM_REST'
            if 'Activated' in t:
                return 'PBMC-CD4T', 'IMM_CD4T_Act'
            if 'IFNb' in t:
                return 'PBMC-CD4T', 'IMM_CD4T_IFNb'
            if 'Th17' in t:
                return 'PBMC-CD4T', 'IMM_CD4T_Th17'
        return None, None

    # 分组: (cell, pert) → [样本索引]
    groups = {}
    for i, m in enumerate(sample_meta):
        cell, pert = classify_title(m['title'])
        if cell is None:
            print(f'  无法分类: {m["title"]}', flush=True)
            continue
        groups.setdefault((cell, pert), []).append(i)

    print(f'\n免疫 (细胞, 扰动) 组合: {len(groups)} 个')
    samples_out = []
    for (cell, pert), idxs in sorted(groups.items()):
        vec = np.zeros(len(hvg), dtype=np.float32)
        cnt = 0
        for gpos, g in enumerate(hvg):
            vals = [sample_vec[i][g] for i in idxs if g in sample_vec[i]]
            if vals:
                vec[gpos] = np.mean(vals)
                cnt += 1
        if cnt < 3000:
            print(f'  丢弃 {cell}|{pert}: 只覆盖 {cnt} 基因', flush=True)
            continue
        # per-sample zscore 标准化, 对齐 LINCS level5 口径
        vec = norm_expr(vec)
        samples_out.append({'cell': cell, 'pert': pert, 'expr': vec.tolist(),
                            'n_samp': len(idxs), 'coverage': cnt})
        print(f'  {cell} | {pert}: {len(idxs)} 样本, {cnt} 基因覆盖', flush=True)

    os.makedirs(os.path.join(BASE, 'data', 'geo_immune'), exist_ok=True)
    out = os.path.join(BASE, 'data', 'geo_immune', 'immune_samples.json')
    json.dump(samples_out, open(out, 'w'))
    print(f'\n保存 {out}: {len(samples_out)} 个免疫代表样本', flush=True)
    cells = sorted(set(s['cell'] for s in samples_out))
    perts = sorted(set(s['pert'] for s in samples_out))
    print(f'细胞系: {len(cells)} 个 | 扰动: {len(perts)} 个', flush=True)
    print('细胞系:', cells, flush=True)
    print('扰动:', perts, flush=True)

if __name__ == '__main__':
    main()
