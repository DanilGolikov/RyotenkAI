"""Debug-bundle pytest plugin.

При падении теста на L5+ (e2e/stack/chaos/load/visual/replay) собирает
``tests/.debug_bundles/<slug>-<ts>.tar.gz`` с отчётом, логами и
``fake_state.json`` — последний берётся через
:data:`tests._harness.stack._context.current_stack` (Phase 2). Если ни
один Stack не активен, ``fake_state.json`` остаётся пустым словарём.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

_BUNDLED_LAYERS = frozenset({"e2e", "stack", "chaos", "load", "visual", "replay"})
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _layer_for(nodeid: str) -> str | None:
    parts = nodeid.replace("\\", "/").split("/")
    if not parts or parts[0] != "tests" or len(parts) < 2:
        return None
    return parts[1]


def _slug(nodeid: str) -> str:
    return _SLUG_RE.sub("-", nodeid).strip("-")[:80] or "test"


def _add_text(tar: tarfile.TarFile, name: str, content: str) -> None:
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _collect_fake_state() -> dict[str, Any]:
    """Best-effort sidecar state dump via the active Stack.

    The contextvar lookup deliberately tolerates absence — most L5+ tests
    that don't boot a Stack still want a debug bundle when they fail.
    """
    # Imported here to avoid an import cycle: orchestrator.py imports
    # current_stack which lives in tests._harness.stack.
    try:
        from tests._harness.stack._context import current_stack
    except Exception:
        return {}

    stack = current_stack.get()
    if stack is None:
        return {}

    async def _do_dump() -> dict[str, Any]:
        return await stack.state_dump()

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_do_dump())
        # If we're inside a running loop (rare from a pytest hook but
        # possible in async tests), spin a small thread-safe runner.
        import threading

        result: dict[str, dict[str, Any]] = {}
        err: list[BaseException] = []

        def _runner() -> None:
            try:
                result.update(asyncio.run(_do_dump()))
            except BaseException as exc:
                err.append(exc)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(timeout=5.0)
        if err:
            return {"error": repr(err[0])}
        return result
    except BaseException as exc:
        return {"error": repr(exc)}


class DebugBundlePlugin:
    def __init__(self, output_dir: Path) -> None:
        self._dir = output_dir

    @property
    def output_dir(self) -> Path:
        return self._dir

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            return
        if report.outcome != "failed":
            return
        layer = _layer_for(report.nodeid)
        if layer not in _BUNDLED_LAYERS:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        bundle = self._dir / f"{_slug(report.nodeid)}-{ts}.tar.gz"
        with tarfile.open(bundle, "w:gz") as tar:
            _add_text(tar, "report.txt", str(report.longrepr) if report.longrepr else "")
            _add_text(tar, "logs/captured.txt", report.caplog or "")
            _add_text(tar, "logs/stdout.txt", report.capstdout or "")
            _add_text(tar, "logs/stderr.txt", report.capstderr or "")
            with contextlib.suppress(Exception):
                fake_state = _collect_fake_state()
                _add_text(tar, "fake_state.json", json.dumps(fake_state, default=str, indent=2))
            _add_text(tar, "journal.txt", "")


def register(config: pytest.Config, output_dir: Path | None = None) -> DebugBundlePlugin:
    target = output_dir or Path(config.rootpath) / "tests" / ".debug_bundles"
    plugin = DebugBundlePlugin(target)
    config.pluginmanager.register(plugin, name="ryotenkai-debug-bundle")
    return plugin
