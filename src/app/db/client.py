"""Standardized DB client entrypoint.

Compatibility layer for the existing root `database.py`.
"""

from database import DatabaseManager, get_db

__all__ = ["DatabaseManager", "get_db"]

