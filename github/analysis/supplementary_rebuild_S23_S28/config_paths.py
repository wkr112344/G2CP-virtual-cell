# -*- coding: utf-8 -*-
"""
统一路径配置 —— 所有复现脚本从这里取输入/输出目录，不再写死本机绝对路径。

用法（二选一）
------------
1) 环境变量
     set GIGASCIENCE_ROOT=D:\\gigascience_repro        (Windows cmd)
     export GIGASCIENCE_ROOT=/data/gigascience_repro   (bash)
   目录下应有：
     <ROOT>/source_data/          ← 从 Zenodo record 解出的 source_data/
     <ROOT>/external/             ← 需自备的第三方公共数据（见 README）
     <ROOT>/work/                 ← 脚本中间产物 / 本包输出

2) 直接改本文件里的 DEFAULTS，或复制一份 config_local.py

任何脚本都以 `from config_paths import P` 开头，P 暴露以下属性：
  P.root, P.source_data, P.external, P.work,
  P.depmap_bench  (Bai 等 drug-perturbation-benchmark 仓库根目录)
  P.h5ad          (LINCS SDST h5ad)
  P.genemap       (93_gene_map_12328_to_978.json)
  P.cache_modz, P.cache_cd  (两套 LINCS 参考的逐 (cell,drug) 缓存)
  P.msigdb_gmt    (MSigDB C2:CGP v2023.2 gmt，需自行按许可下载)
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 默认值
DEFAULTS = {
    # 项目根：可从环境变量覆盖
    "root": os.environ.get("GIGASCIENCE_ROOT", ""),
    # 第三方大文件目录（不经 Zenodo 分发，需按 README 自行下载）
    "external": os.environ.get("GIGASCIENCE_EXTERNAL", ""),
}


def _resolve():
    root_env = DEFAULTS["root"].strip()
    if root_env:
        root = Path(root_env).expanduser().resolve()
        source_data = root / "source_data"
        external = Path(DEFAULTS["external"]).expanduser().resolve() if DEFAULTS["external"].strip() else root / "external"
        work = root / "work"
    else:
        # 未设环境变量时的兜底：以本文件所在目录往上找
        here = Path(__file__).resolve()
        root = here.parents[3]          # .../GigaScience_reproducibility_release
        source_data = root / "zenodo" / "source_data"
        external = root / "external"
        work = root / "work"
    return root, source_data, external, work


ROOT, SOURCE_DATA, EXTERNAL, WORK = _resolve()


class _P:
    root = ROOT
    source_data = SOURCE_DATA
    external = EXTERNAL
    work = WORK

    # ---------- Zenodo source_data 内的固定相对路径 ----------
    @property
    def genemap(self) -> Path:
        return self.source_data / "lincs_index" / "93_gene_map_12328_to_978.json"

    # ---------- 第三方仓库 / 大文件（external 下）----------
    @property
    def depmap_bench(self) -> Path:
        """Bai et al. drug-perturbation-benchmark 仓库根目录。
        来源：https://github.com/baijinming97/drug-perturbation-benchmark
        预测结果归档：https://zenodo.org/records/20081274

        可用 GIGASCIENCE_BENCH 环境变量覆盖（例如解压后目录带 -main 后缀时）。
        """
        env = os.environ.get("GIGASCIENCE_BENCH", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return self.external / "drug-perturbation-benchmark"

    @property
    def h5ad(self) -> Path:
        """LINCS SDST 处理后表达矩阵（XPert 提供）。"""
        return self.depmap_bench / "data" / "XPert" / "processed_data" / "l1000_sdst_78453.h5ad"

    @property
    def cache_modz(self) -> Path:
        """beta2020 (MODZ) 逐 (cell, drug) 参考缓存：pairs_meta.npz + y.npy。
        可用 GIGASCIENCE_CACHE_MODZ 覆盖。
        """
        env = os.environ.get("GIGASCIENCE_CACHE_MODZ", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return self.external / "g2cp_cache_beta_v2"

    @property
    def cache_cd(self) -> Path:
        """dcic2021 (CD) 逐 (cell, drug) 参考缓存：pairs_meta.npz + y.npy。
        可用 GIGASCIENCE_CACHE_CD 覆盖。
        """
        env = os.environ.get("GIGASCIENCE_CACHE_CD", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return self.external / "g2cp_cache_2021"

    @property
    def msigdb_gmt(self) -> Path:
        """MSigDB C2:CGP v2023.2（Human, symbols）。需按 MSigDB 许可自行下载。"""
        env = os.environ.get("GIGASCIENCE_MSIGDB", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return self.external / "msigdb" / "c2.cgp.v2023.2.Hs.symbols.gmt"

    # ---------- 输出 ----------
    @property
    def out(self) -> Path:
        self.work.mkdir(parents=True, exist_ok=True)
        return self.work


P = _P()


def require(*paths):
    """检查输入是否存在，缺失时给出可执行的提示。"""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        msg = ["以下输入缺失，请按 README.md 的『数据从哪里来』一节准备："]
        msg += ["  - " + m for m in missing]
        msg.append("可用环境变量 GIGASCIENCE_ROOT / GIGASCIENCE_EXTERNAL 指向你的目录。")
        raise FileNotFoundError("\n".join(msg))


if __name__ == "__main__":
    for k in ("root", "source_data", "external", "work", "genemap", "depmap_bench",
              "h5ad", "cache_modz", "cache_cd", "msigdb_gmt"):
        print("%-14s %s" % (k, getattr(P, k)))
