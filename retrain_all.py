# -*- coding: utf-8 -*-
"""retrain_all.py —— 全细胞系数据重训一键入口。
用户下载的 level5 gctx 到位后，一条命令跑完整流程：
    python retrain_all.py --gctx "D:/下载/level5_....gctx" [--epochs 800] [--qc]

流程：parse gctx → 合并 pool → 预处理（可选 qc 过滤）→ 对比预训练+表型训练 → 三指标评测 → 提示切服务
"""
import sys, os, subprocess, argparse, json, time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LINCSEXTRA = os.path.join(BASE, "data", "lincs_extra", "all_cells.h5ad")
POOL = os.path.join(BASE, "data", "g2cp", "data", "LINCS", "pool", "pool_gene_chem_ctrl_adata.h5ad")


def log(m):
    print(f"[retrain] {m}", flush=True)


def run(cmd, tag):
    log(f"▶ {tag}: {cmd}")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=BASE)
    log(f"  完成 {tag}（{time.time()-t0:.0f}s, exit={r.returncode}）")
    if r.returncode != 0:
        log(f"❌ {tag} 失败，中止"); sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gctx", required=True, help="level5 .gctx 全细胞系数据路径")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--qc", action="store_true", help="只使用 qc_pass 高质量样本")
    ap.add_argument("--no_pool", action="store_true", help="不用原有 5 系 pool 数据（仅新数据）")
    args = ap.parse_args()

    log("========== 全细胞系重训流水线 ==========")
    log(f"gctx: {args.gctx} | epochs: {args.epochs} | qc 过滤: {args.qc}")

    # 1) 解析 gctx → h5ad
    run(f'"{PY}" parse_gctx.py --gctx "{args.gctx}"', "解析 gctx")

    # 2) 合并 pool + 新数据（同构 h5ad，按样本合并；细胞系字段扩展）
    if not args.no_pool:
        run(f'"{PY}" -c "import sys,os,anndata as ad;'
            f"sys.path.insert(0, os.getcwd());"
            f"import pandas as pd;"
            f"a=ad.read_h5ad('{POOL}'); b=ad.read_h5ad('{LINCSEXTRA}');"
            f"common=list(set(a.var_names)&set(b.var_names));"
            f"a=a[:,common]; b=b[:,common];"
            f"c=ad.concat([a,b], join='outer', merge='same');"
            f"os.makedirs('data/lincs_extra', exist_ok=True);"
            f"c.write_h5ad('data/lincs_extra/merged.h5ad');"
            f"print('合并:', c.shape, '细胞系:', c.obs['cell_line'].nunique())",
            "合并 pool+新数据")

    # 3) 预处理（训练脚本 prep，qc 过滤由环境变量控制）
    qc_env = "QC_ONLY=1 " if args.qc else ""
    run(f'cd "{BASE}" && {qc_env}"{PY}" -c "import sys,os;sys.path.insert(0,os.getcwd());from train_g2cp_contrast import prep_extra;prep_extra()"',
        "预处理（含 qc 过滤）")

    # 4) 训练（对比预训练 + 表型联合，进度实时可查）
    run(f'cd "{BASE}" && "{PY}" train_g2cp_contrast.py --phase joint --epochs {args.epochs} '
        f"--headw 2048 --pcc_w 1.5 --lam_nce 0.5 --lam_cls 0.3 --cls_bal 0.6 --tau 0.1 "
        f"--max_cell 20 --bpert 64 --save g2cp_fullcells.pt > train_fullcells.log 2>&1",
        "训练（进度: train_progress.json / train_fullcells.log）")

    # 5) 三指标评测
    run(f'cd "{BASE}" && "{PY}" eval_g2cp.py --ckpt g2cp_fullcells.pt 2>&1 | grep -E "PCC|留出" | tail -4'
        f' && "{PY}" eval_smd.py --ckpt g2cp_fullcells.pt 2>&1 | grep SMD | tail -1'
        f' && "{PY}" eval_ef.py --ckpt g2cp_fullcells.pt 2>&1 | grep EF@ | tail -3',
        "三指标评测")

    log("========== 全部完成 ==========")
    log("下一步：更新 8766 服务指向 g2cp_fullcells.pt：")
    log('  python -c "open(\'serve_g2cp.py\',\'a\')"  # 或改 CKPT 回退链，重启 serve_g2cp.py')
    log("也可直接：python serve_g2cp.py --port 8766")


if __name__ == "__main__":
    main()
