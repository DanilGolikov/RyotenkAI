"""Versioned JSON-Schema contracts for cross-package boundaries.

Every schema in this directory is generated from the production source
of truth via :mod:`tests._contracts._generators`. Hand-written schemas
drift from production; generated schemas don't. Run::

    .venv/bin/python -m tests._contracts._generators.regen_all

to regenerate after touching a model.

Each emitted schema is committed and includes ``$id`` + ``title`` +
``version`` (string).
"""

from __future__ import annotations

from pathlib import Path

CONTRACTS_DIR = Path(__file__).resolve().parent

__all__ = ["CONTRACTS_DIR"]
