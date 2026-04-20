#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一体化启动入口。

支持模式：
- api: 仅启动 Flask API
- worker: 仅启动检测主流程
- mock: 启动课程 mock 服务
- all: 同时启动 API（后台线程）+ worker（前台）
"""

import argparse
import os
import sys
import threading
from typing import List


def _ensure_src_on_path() -> None:
    src_root = os.path.abspath(os.path.dirname(__file__))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


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


def _run_mock() -> None:
    from scripts.mock_registry_server import run_server

    run_server()


def main() -> None:
    _ensure_src_on_path()

    parser = argparse.ArgumentParser(description="一体化启动入口")
    parser.add_argument(
        "--mode",
        choices=["api", "worker", "mock", "all"],
        default="api",
        help="启动模式",
    )
    parser.add_argument(
        "--worker-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="传递给 worker(main.py) 的参数，例如 --worker-args --camera 0 --output out.mp4",
    )
    args = parser.parse_args()

    if args.mode == "api":
        _run_api()
        return

    if args.mode == "worker":
        _run_worker(args.worker_args)
        return

    if args.mode == "mock":
        _run_mock()
        return

    api_thread = threading.Thread(target=_run_api, daemon=True)
    api_thread.start()
    _run_worker(args.worker_args)


if __name__ == "__main__":
    main()

