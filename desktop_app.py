"""
虚拟细胞工作台 · 桌面版
=======================
双击运行 → 自动拉起本地模型服务（Flask 后台线程）→ 原生窗口（WebView2）打开工作台。
全部计算在本机完成，不依赖任何外部服务器、不依赖浏览器手动输入地址。

用法：
  python desktop_app.py          # 开发模式
  VirtualCellWorkbench.exe       # 打包后的桌面程序（双击即用）
"""
import os
import sys
import time
import socket
import threading
import urllib.request


def _meipass():
    """资源根目录：打包后 = PyInstaller _MEIPASS；开发模式 = 项目根。"""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE = _meipass()
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# Windows 控制台默认 GBK，强制 UTF-8 输出避免打印特殊字符崩溃
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_ready(url, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def self_test():
    """无窗口自检：起服务 → health → predict → custom_drug → 退出（打包验证用）。"""
    import json
    port = _free_port()
    print(">>> [self-test] 加载模型 ...", flush=True)
    from unipret.serve_api import create_app
    app = create_app(port=port)
    threading.Thread(target=lambda: app.run(
        host="127.0.0.1", port=port, threaded=True, use_reloader=False),
        daemon=True).start()
    if not _wait_ready(f"http://127.0.0.1:{port}/health", timeout=120):
        print("[ERR] [self-test] 服务未就绪", flush=True)
        return 1

    def _post(path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=data, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())

    d = _post("/predict", {"drug": "Ruxolitinib (INCB018424)", "cell": 0})
    print(f"[self-test] predict OK 靶点top3={[t['g'] for t in d['targets'][:3]]}", flush=True)
    d2 = _post("/custom_drug", {"smiles": "CC1CCC2CC3CC(C(=CC=CC=CC4=CC(=O)C5(CC(C(=C5C(=O)C4=O)OC)OC)C(=O)O3)OC)C(=O)C2C1", "cell": 0})
    print(f"[self-test] custom_drug OK 通路={d2.get('pathway_name')}", flush=True)
    print("[OK] [self-test] 全部通过", flush=True)
    return 0


def main():
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    port = _free_port()
    print(">>> 正在加载模型（stageB 预测 + UniPert CPI + ESM2-8M）...", flush=True)
    print("    首次启动需 10-40 秒（取决于磁盘与 GPU），请稍候。", flush=True)
    try:
        from unipret.serve_api import create_app
        app = create_app(port=port)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERR] 模型加载失败：{e}", flush=True)
        input("按回车退出...")
        return

    threading.Thread(target=lambda: app.run(
        host="127.0.0.1", port=port, threaded=True, use_reloader=False),
        daemon=True).start()

    if not _wait_ready(f"http://127.0.0.1:{port}/health"):
        print("[ERR] 模型服务启动失败，请查看上方错误信息", flush=True)
        input("按回车退出...")
        return

    import webview
    url = f"http://127.0.0.1:{port}/gui/workbench.html"
    try:
        webview.create_window("虚拟细胞工作台 · UniPert-G2CP", url,
                              width=1440, height=920, min_size=(1080, 700))
        webview.start()
    except Exception as e:
        print(f"[WARN] 原生窗口启动失败（{e}），自动改用默认浏览器打开：{url}", flush=True)
        import webbrowser
        webbrowser.open(url)
        input("按回车退出...")
    print("工作台已退出。", flush=True)


if __name__ == "__main__":
    main()
