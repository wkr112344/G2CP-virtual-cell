# -*- coding: utf-8 -*-
"""level5 断点续传脚本：网络恢复即自动下载，断线自动退避重试。"""
import subprocess, time, os

URL = "https://lincs-dcic.s3.amazonaws.com/LINCS-sigs-2021/gctx/cd-coefficient/cp_coeff_mat.gctx"
TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lincs", "level5_cp.gctx")
TOTAL = 36084518760  # 33.6 GB（论文同款 cp_coeff_mat.gctx）

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log(f"续传脚本启动，目标 {TOTAL/1e9:.1f}GB -> {TARGET}")
while True:
    sz = os.path.getsize(TARGET) if os.path.exists(TARGET) else 0
    if sz >= TOTAL - 1000000:
        log(f"下载完成！{sz/1e9:.2f}GB")
        break
    log(f"当前 {sz/1e9:.2f}GB ({sz*100/TOTAL:.1f}%)，开始续传...")
    r = subprocess.run(["curl", "-s", "-L", "-C", "-", "-o", TARGET, URL,
                        "--retry", "3", "--retry-delay", "20", "--retry-all-errors",
                        "--max-time", "300"], capture_output=True, text=True)
    nsz = os.path.getsize(TARGET) if os.path.exists(TARGET) else 0
    got = nsz - sz
    if nsz >= TOTAL - 1000000:
        log(f"下载完成！{nsz/1e9:.2f}GB")
        break
    log(f"本轮新增 {got/1e6:.1f}MB (exit={r.returncode})，60s 后重试")
    time.sleep(60)
