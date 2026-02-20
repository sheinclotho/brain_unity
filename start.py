#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain — 一键启动
=====================
直接运行即可：python start.py

无需任何参数。自动检测模型文件，自动打开浏览器。
"""

import asyncio
import http.server
import logging
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── 配置（默认值适合大多数用户，高级用户可修改此处） ────────────────────────
WEB_PORT = 8766          # 浏览器访问端口
WS_PORT  = 8765          # WebSocket 端口
WS_HOST  = "127.0.0.1"  # 只监听本地，安全默认值


def find_model() -> Optional[Path]:
    """在项目目录内自动搜索训练好的模型文件。"""
    candidates = list(project_root.glob("**/hetero_gnn_trained.pt"))
    candidates += list(project_root.glob("**/best_model.pt"))
    candidates += list(project_root.glob("**/*.pt"))
    # 排除临时/测试目录
    candidates = [p for p in candidates if "__pycache__" not in str(p)]
    return candidates[0] if candidates else None


def start_web_server(web_dir: Path, port: int) -> None:
    """在单独线程中启动静态文件服务器。"""
    os.chdir(web_dir)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass  # 静默日志

    server = http.server.HTTPServer(("127.0.0.1", port), QuietHandler)
    server.serve_forever()


async def start_websocket_server(model_path: Optional[Path], output_dir: Path) -> None:
    """启动 WebSocket 后端服务。"""
    try:
        from unity_integration import BrainVisualizationServer, BrainStateExporter, StimulationSimulator

        state_dir = output_dir / "brain_data" / "model_output"
        state_dir.mkdir(parents=True, exist_ok=True)

        exporter   = BrainStateExporter(atlas_info=None)
        simulator  = StimulationSimulator(n_regions=200)

        server = BrainVisualizationServer(
            model_path=str(model_path) if model_path else None,
            exporter=exporter,
            simulator=simulator,
            output_dir=str(output_dir),
            host=WS_HOST,
            port=WS_PORT,
        )

        await server.start()

    except ImportError as exc:
        logger.error(f"依赖缺失: {exc}")
        logger.info("请运行: pip install -r requirements.txt")
        sys.exit(1)


def check_deps() -> bool:
    """快速检查必要依赖。"""
    missing = []
    for pkg in ("numpy", "websockets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(f"缺少依赖: {', '.join(missing)}")
        logger.info("请运行: pip install -r requirements.txt")
        return False
    return True


def parse_args():
    """轻量参数解析（不用 argparse，减少导入时间）。"""
    args = {"model": None}  # type: dict
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            args["model"] = Path(argv[i + 1])
    return args


def main():
    print()
    print("=" * 52)
    print("  🧠  TwinBrain  —  大脑可视化平台")
    print("=" * 52)
    print()

    if not check_deps():
        sys.exit(1)

    cli = parse_args()
    model_path = cli["model"] or find_model()

    if model_path:
        logger.info(f"✓ 找到模型: {model_path}")
    else:
        logger.info("ℹ  未找到模型文件，以演示模式运行（随机数据）")

    output_dir = project_root / "Unity_TwinBrain"
    output_dir.mkdir(exist_ok=True)

    # 启动静态 Web 服务器（用于前端界面）
    web_dir = project_root / "web_frontend"
    if web_dir.exists():
        t = threading.Thread(target=start_web_server, args=(web_dir, WEB_PORT), daemon=True)
        t.start()
        logger.info(f"✓ Web 界面: http://localhost:{WEB_PORT}")
    else:
        logger.warning("web_frontend 目录不存在，跳过浏览器界面")

    logger.info(f"✓ WebSocket 服务: ws://{WS_HOST}:{WS_PORT}")
    logger.info("")
    logger.info("按 Ctrl+C 停止服务")
    logger.info("")

    # 延迟 1 秒后自动打开浏览器
    def open_browser():
        time.sleep(1)
        webbrowser.open(f"http://localhost:{WEB_PORT}")

    if web_dir.exists():
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        asyncio.run(start_websocket_server(model_path, output_dir))
    except KeyboardInterrupt:
        print("\n\n👋 已停止\n")


if __name__ == "__main__":
    main()
