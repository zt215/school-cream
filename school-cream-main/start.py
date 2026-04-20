#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一体化启动入口。

支持模式：
- api: 仅启动 Flask API
- worker: 仅启动检测主流程
- all: 同时启动 API（后台线程）+ worker（前台）
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import threading
from typing import List

API_REQUIRED_MODULES = [
    ("flask", "flask"),
    ("flask_cors", "flask-cors"),
    ("reportlab", "reportlab"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
]

WORKER_REQUIRED_MODULES = [
    ("ultralytics", "ultralytics"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("requests", "requests"),
]


def _ensure_project_on_path() -> None:
    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _pip_install(package: str) -> None:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return
    except subprocess.CalledProcessError as exc:
        if os.name == "nt":
            print(f"[deps] 安装 {package} 失败，尝试使用 --user ...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
                return
            except subprocess.CalledProcessError:
                pass
        raise SystemExit(exc.returncode) from exc


def _ensure_python_module(module_name: str, package_name: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    print(f"[deps] 检测到缺失模块 {module_name}，开始安装 {package_name} ...")
    _pip_install(package_name)


def _ensure_mode_dependencies(mode: str) -> None:
    module_specs = API_REQUIRED_MODULES if mode == "api" else WORKER_REQUIRED_MODULES
    for module_name, package_name in module_specs:
        _ensure_python_module(module_name, package_name)
    print("[deps] 依赖检测完成。")


def _run_api() -> None:
    from api_server import run_api_server

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    run_api_server(host=host, port=port, debug=debug)


def _run_worker(worker_args: List[str]) -> None:
    from main import main as worker_main

    old_argv = sys.argv[:]
    try:
        sys.argv = ["main.py", *worker_args]
        worker_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    _ensure_project_on_path()

    parser = argparse.ArgumentParser(description="一体化启动入口")
    parser.add_argument(
        "--mode",
        choices=["api", "worker", "all"],
        default="api",
        help="启动模式，默认 api",
    )
    parser.add_argument(
        "--worker-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="传递给 main.py 的参数，例如 --worker-args --camera 0 --output out.mp4",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="跳过启动前依赖自动检测与安装",
    )
    args = parser.parse_args()
    if not args.skip_deps:
        target_mode = "worker" if args.mode in {"worker", "all"} else "api"
        _ensure_mode_dependencies(target_mode)

    if args.mode == "api":
        _run_api()
        return

    if args.mode == "worker":
        _run_worker(args.worker_args)
        return

    api_thread = threading.Thread(target=_run_api, daemon=True)
    api_thread.start()
    _run_worker(args.worker_args)


if __name__ == "__main__":
    main()
