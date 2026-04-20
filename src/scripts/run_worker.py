#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run detection worker from standardized script path."""

import os
import sys


def _ensure_project_root_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def main() -> None:
    _ensure_project_root_on_path()
    from main import main as legacy_main

    legacy_main()


if __name__ == "__main__":
    main()

