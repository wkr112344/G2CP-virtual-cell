#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟细胞 · 真实转录组/遗传筛选数据下载器（B 方案：UniPert-G2CP 可靠工具）
- 全部公开免费（学术 CC/公共领域），无需付费
- 断点续传（HTTP Range），落盘到桌面统一目录
- 数据分类：DepMap(基因预训练标签) / LINCS(药物真实转录组) / sciPlex3(单细胞 perturb-seq)
- 仅用标准库，不依赖 torch（下载/解析阶段不需要 GPU）

注意：各数据源的确切文件名/体积以下载前 WebFetch/WebSearch 核对的稳定 URL 为准，
本脚本给出健壮的下载框架与已知稳定基址，执行 download_all() 前请先跑 verify_urls()。
"""
import os, sys, json, time, urllib.request, urllib.error, hashlib

ROOT = r"C:\Users\wkr20\Desktop\virtual_cell_real_data"
DIRS = {
    "DepMap":   os.path.join(ROOT, "DepMap"),
    "LINCS":    os.path.join(ROOT, "LINCS"),
    "sciPlex3": os.path.join(ROOT, "sciPlex3"),
}

# ---- 数据源定义（基址稳定；确切文件名下载前用 verify_urls 核对）----
# 1) LINCS L1000 (GEO GSE92742)：药物扰动真实转录组（978 landmark genes）
#    小文件先跑通：cell_info / gene_info / inst_info；大矩阵 Level3 gctx 后续按需下
LINCS_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
LINCS_FILES = [
    ("GSE92742_Broad_LINCS_gene_info.txt",        "LINCS",  "~0.3 MB"),
    ("GSE92742_Broad_LINCS_cell_info.txt",        "LINCS",  "~0.3 MB"),
    ("GSE92742_Broad_LINCS_inst_info.txt",        "LINCS",  "~20 MB"),
    # 大矩阵（按需，约 1.8-2.3 GB）：
    # ("GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx", "LINCS", "~2.3 GB"),
]

# 2) DepMap (Broad Institute)：CRISPR 基因必需性（基因预训练标签）
#    DepMap 公开在 AWS Open Data / portal download；确切 URL 以 verify_urls 核对
DEPMAP_BASE = "https://depmap.org/portal/download/api/download/"
DEPMAP_FILES = [
    ("Achilles_gene_effect.csv", "DepMap", "~30 MB"),   # CRISPR 基因效应（预训练标签）
    ("Achilles_gene_dependency.csv", "DepMap", "~30 MB"),
    ("sample_info.csv", "DepMap", "~5 MB"),             # 细胞系注释
]

# 3) sciPlex3 (Norman et al. 2021, Science)：单细胞 perturb-seq，3 细胞系
#    zenodo/figshare 公开 h5ad；确切 URL 以 verify_urls 核对
SCIPLEX_BASE = "https://zenodo.org/records/4550619/files/"
SCIPLEX_FILES = [
    ("sciplex3.h5ad", "sciPlex3", "~1 GB"),            # 合并版（如可用）
    # 或按细胞系拆分（各 94-285 MB）：
    # ("sciplex_293t.h5ad", "sciPlex3", "~285 MB"),
    # ("sciplex_k562.h5ad", "sciPlex3", "~120 MB"),
    # ("sciplex_jurkat.h5ad", "sciPlex3", "~94 MB"),
]

ALL = [("LINCS", LINCS_BASE, LINCS_FILES),
       ("DepMap", DEPMAP_BASE, DEPMAP_FILES),
       ("sciPlex3", SCIPLEX_BASE, SCIPLEX_FILES)]

def _ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

def _url_ok(url, timeout=15):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200, r.headers.get("Content-Length")
    except Exception as e:
        return False, str(e)

def verify_urls():
    """下载前核对每个文件的 URL 是否可达 + 真实体积（避免写死失效链接）。"""
    _ensure_dirs()
    print("=== 核对数据源 URL 可达性 ===")
    ok = 0; bad = 0
    for src, base, files in ALL:
        for fname, sub, size_hint in files:
            url = base + fname
            good, cl = _url_ok(url)
            tag = "OK " if good else "FAIL"
            real = f"real={cl}B" if good and cl else f"({cl})"
            print(f"  [{tag}] {src:9s} {fname:55s} hint={size_hint:10s} {real}")
            ok += 1 if good else 0; bad += 0 if good else 1
    print(f"=== 可达 {ok} / 失败 {bad} ===")
    return bad == 0

def _download(base, fname, sub, retry=3, timeout=60):
    """单文件断点续传下载到 DIRS[sub]。"""
    url = base + fname
    dest = os.path.join(DIRS[sub], fname)
    for attempt in range(1, retry+1):
        try:
            hdr = {}
            if os.path.exists(dest):
                hdr["Range"] = f"bytes={os.path.getsize(dest)}-"
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r, \
                 open(dest, "ab" if os.path.exists(dest) else "wb") as f:
                total = r.headers.get("Content-Length")
                total = int(total) if total else None
                got = os.path.getsize(dest) if os.path.exists(dest) else 0
                chunk = 1 << 20
                t0 = time.time()
                while True:
                    buf = r.read(chunk)
                    if not buf: break
                    f.write(buf); got += len(buf)
                    if total:
                        pct = got/total*100
                        sys.stdout.write(f"\r  {fname}: {got>>20}/{total>>20} MB ({pct:.1f}%)")
                        sys.stdout.flush()
                print(f"\n  完成 {fname} ({got>>20} MB)")
            return True
        except Exception as e:
            print(f"\n  尝试 {attempt}/{retry} 失败 {fname}: {e}")
            time.sleep(3*attempt)
    return False

def download_all(skip_large=True):
    """下载所有已定义的小文件；大矩阵用 skip_large 跳过（手动确认后再下）。"""
    _ensure_dirs()
    print("=== 开始下载真实数据（落盘桌面 virtual_cell_real_data）===")
    for src, base, files in ALL:
        for fname, sub, size_hint in files:
            big = any(k in size_hint for k in ("GB",))  # 粗略判断大文件
            if big and skip_large:
                print(f"  [跳过大文件] {fname} ({size_hint}) —— 先跑通管线，稍后单独下")
                continue
            print(f"-- {src}/{fname} --")
            _download(base, fname, sub)
    print("=== 下载完成（大矩阵需手动确认后放开）===")

if __name__ == "__main__":
    # 默认：先核对 URL 可达性（不下载），确认链接有效再 download_all()
    if len(sys.argv) > 1 and sys.argv[1] == "go":
        download_all()
    else:
        verify_urls()
        print("\n核对无误后运行：python fetch_real_data.py go")
