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
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
import threading
from typing import List


def _ensure_src_on_path() -> None:
    src_root = os.path.abspath(os.path.dirname(__file__))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def _normalize_pkg_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _extract_requirement_name(requirement_line: str) -> str:
    line = requirement_line.strip()
    if not line or line.startswith("#"):
        return ""

    # 去掉行内注释与环境标记，保留基础 requirement 片段
    line = line.split("#", 1)[0].strip()
    line = line.split(";", 1)[0].strip()
    if not line:
        return ""

    if line.startswith("-"):
        return ""

    name_part = line.split("[", 1)[0]
    match = re.match(r"^[A-Za-z0-9_.-]+", name_part)
    if not match:
        return ""

    return _normalize_pkg_name(match.group(0))


def _ensure_dependencies(requirements_path: str) -> None:
    if not os.path.exists(requirements_path):
        return

    installed = set()
    for dist in importlib.metadata.distributions():
        dist_name = dist.metadata.get("Name")
        if dist_name:
            installed.add(_normalize_pkg_name(dist_name))

    missing_specs: List[str] = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            requirement = raw_line.strip()
            pkg_name = _extract_requirement_name(requirement)
            if not pkg_name:
                continue
            if pkg_name not in installed:
                missing_specs.append(requirement.split("#", 1)[0].strip())

    if not missing_specs:
        print("[deps] 依赖检测通过，无需安装。")
        return

    print(f"[deps] 检测到缺失依赖 {len(missing_specs)} 个，开始自动安装...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_specs])
    except subprocess.CalledProcessError as exc:
        # Windows 下常见 WinError 5，无管理员权限时回退到 --user 安装。
        if os.name == "nt":
            print("[deps] 检测到安装失败，尝试使用 --user 继续安装...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", *missing_specs])
                print("[deps] 已通过 --user 安装缺失依赖。")
                return
            except subprocess.CalledProcessError:
                pass

        print("[deps] 自动安装失败，请手动执行：")
        print(f'  "{sys.executable}" -m pip install -r "{requirements_path}"')
        raise SystemExit(exc.returncode) from exc

    print("[deps] 缺失依赖安装完成。")


def _ensure_python_module(module_name: str, package_name: str | None = None) -> None:
    """确保某个 Python 模块可导入，不可导入时自动安装对应包。"""
    if importlib.util.find_spec(module_name) is not None:
        return

    pkg = package_name or module_name
    print(f"[deps] 检测到缺失模块 {module_name}，尝试安装 {pkg} ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    except subprocess.CalledProcessError as exc:
        if os.name == "nt":
            print("[deps] 检测到安装失败，尝试使用 --user 安装...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])
                return
            except subprocess.CalledProcessError:
                pass

        print("[deps] 自动安装失败，请手动执行：")
        print(f'  "{sys.executable}" -m pip install {pkg}')
        raise SystemExit(exc.returncode) from exc


def _run_api() -> None:
    # API 模式关键依赖兜底，避免全量依赖安装耗时过长导致启动失败
    _ensure_python_module("reportlab")
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
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="跳过启动前依赖自动检测与安装",
    )
    args = parser.parse_args()
    if not args.skip_deps:
        requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
        # 默认 api/mock 模式仅做关键依赖兜底，避免每次启动都拉取重型 ML 依赖。
        if args.mode in {"worker", "all"}:
            _ensure_dependencies(requirements_path)

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

