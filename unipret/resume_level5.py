"""
LINCS level5 自动续传 + 合并 + 校验（凌晨低峰期运行）
=====================================================
目标：把 33.6GB 的 level5 化学扰动签名矩阵（论文 G2CP 同款数据）下完。
part 文件已存在（5.34GB），用 curl -C - 断点续传，直到 4 个分块全部完整。
完成后合并为 level5_full.gctx 并校验大小；随后调用解析流程。

用法：python unipret/resume_level5.py
幂等：重复运行安全（已完成的 part 会跳过）。
"""
import os
import sys
import time
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINCS = os.path.join(BASE, "data", "lincs")
URL = ("https://lincs-dcic.s3.amazonaws.com/LINCS-sigs-2021/"
       "gctx/cd-coefficient/cp_coeff_mat.gctx")
TOTAL = 36084518760
N_PARTS = 4
STEP = TOTAL // N_PARTS


def part_range(i):
    s = i * STEP
    e = TOTAL - 1 if i == N_PARTS - 1 else (i + 1) * STEP - 1
    return s, e


def part_path(i):
    return os.path.join(LINCS, f"part{i}.bin")


def part_done(i):
    s, e = part_range(i)
    need = e - s + 1
    p = part_path(i)
    if not os.path.isfile(p):
        return False
    # 严格相等：文件大于 need 说明内容错乱（重复追加），不算完成
    return os.path.getsize(p) == need


def resume_part(i, max_wait=3600):
    s0, e = part_range(i)
    need = e - s0 + 1
    p = part_path(i)
    if part_done(i):
        print(f"  part{i} 已完整", flush=True)
        return True
    sz = os.path.getsize(p) if os.path.isfile(p) else 0
    if sz > need:
        # 异常：文件超过目标大小（历史重复追加 bug 的产物），截断到目标大小
        print(f"  part{i} 大小异常 {sz} > {need}，截断修复", flush=True)
        with open(p, "r+b") as fo:
            fo.truncate(need)
        sz = need
    # 续传起点 = 分块起点 + 已下载字节数（part 文件内容从 s0 起连续）。
    # 注意：不能用 max(s0, sz) —— 当文件大小 < s0 时会从 s0 重复追加，损坏数据。
    s = s0 + sz
    print(f"  part{i}: {sz/2**30:.2f}GB -> 目标 {need/2**30:.2f}GB 续传 (range {s}-{e})...", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if s > e:
            break
        # 注意：curl 的 -C(续传) 与 -r(Range) 互斥，不能同时使用！
        # 改为：Range 起点 = 全局偏移（= s0 + 文件已下载字节数，动态计算），输出追加写入（ab 模式）
        with open(p, "ab") as fo:
            subprocess.run(
                ["curl", "-s", "-L", "-r", f"{s}-{e}", "-o", "-", URL,
                 "--retry", "5", "--retry-delay", "20", "--retry-all-errors",
                 "--max-time", "1800"],
                stdout=fo, stderr=subprocess.DEVNULL)
        if part_done(i):
            print(f"  part{i} 续传完成", flush=True)
            return True
        # 断线/限速：更新续传起点（= s0 + 文件已下载字节数，必须加 s0！），等 30s 重试
        s = s0 + (os.path.getsize(p) if os.path.isfile(p) else 0)
        time.sleep(30)
    return part_done(i)


def merge():
    out = os.path.join(LINCS, "level5_full.gctx")
    if os.path.isfile(out) and os.path.getsize(out) == TOTAL:
        print(f"  level5_full.gctx 已存在且完整 ({TOTAL})", flush=True)
        return True
    with open(out, "wb") as fo:
        for i in range(N_PARTS):
            p = part_path(i)
            if not part_done(i):
                print(f"  part{i} 未完整，不能合并", flush=True)
                return False
            with open(p, "rb") as fi:
                while True:
                    chunk = fi.read(16 * 2**20)
                    if not chunk:
                        break
                    fo.write(chunk)
            print(f"  合并 part{i} 完成", flush=True)
    ok = os.path.getsize(out) == TOTAL
    print(f"  合并完成: {os.path.getsize(out)} / {TOTAL} {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    print(f">>> LINCS level5 续传（当前时间 {time.strftime('%H:%M')}）", flush=True)
    os.makedirs(LINCS, exist_ok=True)
    for i in range(N_PARTS):
        resume_part(i)
    if not all(part_done(i) for i in range(N_PARTS)):
        print("  !! 仍有分块未完整，本次运行结束（下次自动重试）", flush=True)
        return 1
    if merge():
        print("✅ LINCS level5 全部就绪，可用于论文级数据训练", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
