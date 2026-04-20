"""Standardized API entrypoint.

This module keeps backward compatibility by reusing the existing root
`api_server.py` implementation.
"""

from api_server import app, run_api_server

__all__ = ["app", "run_api_server"]

