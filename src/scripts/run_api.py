#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Flask API service from standardized script path."""

import os
import sys


def _ensure_project_root_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def main() -> None:
    _ensure_project_root_on_path()
    from app.api.server import run_api_server

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    run_api_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

