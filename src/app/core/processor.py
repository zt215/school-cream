"""Standardized processing entrypoint.

This module re-exports core symbols from the existing root `main.py`
so external callers can gradually migrate to `src/app/core`.
"""

from main import (
    VideoProcessor,
    detect_phone_usage,
    fetch_json_from_registry,
    generate_output_filename,
    parse_json_data,
)

__all__ = [
    "VideoProcessor",
    "detect_phone_usage",
    "fetch_json_from_registry",
    "generate_output_filename",
    "parse_json_data",
]

