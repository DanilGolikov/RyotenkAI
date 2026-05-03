"""``ryotenkai`` package entry point.

The CLI lives in :mod:`src.cli`. This module is kept as a thin re-export
so ``python -m src.main`` and the legacy ``ryotenkai = "src.main:app"``
``console_scripts`` entry in ``pyproject.toml`` keep resolving without
indirection.
"""

from __future__ import annotations

from src.cli.app import app

__all__ = ["app"]


if __name__ == "__main__":
    app()
