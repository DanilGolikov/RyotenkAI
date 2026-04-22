"""Pytest conftest — bind the sibling ``plugin.py`` as ``plugin`` module
before tests in this directory import it.

This is required because community plugins share the file name ``plugin.py``,
so pytest's importlib mode would otherwise cache the first one seen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def pytest_collectstart(collector):
    _rebind_plugin_module(collector.path)


def _rebind_plugin_module(path) -> None:
    here = Path(path).parent if Path(path).is_file() else Path(path)
    plugin_py = here / "plugin.py"
    if not plugin_py.exists():
        return
    spec = importlib.util.spec_from_file_location("plugin", plugin_py)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["plugin"] = module
    spec.loader.exec_module(module)
