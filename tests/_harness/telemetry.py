"""Test-telemetry pytest plugin.

Один JSONL-файл на pytest-сессию. Каждая строка — одна (test, phase=call)
запись. Phase-фильтр на ``call`` нужен, чтобы setup/teardown не дублировали
запись и не "съедали" outcome (pytest шлёт три отчёта на тест).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

_LAYERS = (
    "unit",
    "component",
    "contract",
    "integration",
    "e2e",
    "stack",
    "property",
    "golden",
    "chaos",
    "load",
    "visual",
    "replay",
)


def _layer_for(nodeid: str) -> str | None:
    parts = nodeid.replace("\\", "/").split("/")
    if not parts or parts[0] != "tests":
        return None
    if len(parts) < 2:
        return None
    layer = parts[1]
    return layer if layer in _LAYERS else None


def _marker_args(item: pytest.Item, name: str) -> list[str]:
    out: list[str] = []
    for marker in item.iter_markers(name=name):
        out.extend(str(a) for a in marker.args)
    return out


class TelemetryPlugin:
    def __init__(self, output_dir: Path) -> None:
        self._dir = output_dir
        self._path: Path | None = None
        self._fh: Any = None

    @property
    def path(self) -> Path | None:
        return self._path

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        del session
        self._dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._path = self._dir / f"run-{stamp}-{os.getpid()}.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        del session, exitstatus
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # Только phase=call; setup/teardown не должны затирать outcome.
        if report.when != "call":
            return
        if self._fh is None:
            return
        item_id = report.nodeid
        # Markers не доступны на TestReport напрямую; user_properties передаются
        # из конфтеста в pytest_runtest_makereport (см. ниже).
        markers: dict[str, list[str]] = {"fakes_used": [], "protocols_exercised": []}
        for key, value in report.user_properties:
            if key in markers and isinstance(value, list):
                markers[key] = list(value)
        record: dict[str, Any] = {
            "test_id": item_id,
            "layer": _layer_for(item_id),
            "fakes_used": markers["fakes_used"],
            "protocols_exercised": markers["protocols_exercised"],
            "scenario": None,
            "duration_ms": round(report.duration * 1000.0, 3),
            "outcome": report.outcome,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if report.outcome == "failed":
            record["error"] = str(report.longrepr) if report.longrepr is not None else ""
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    # Прокидываем маркеры в user_properties до того, как plugin увидит report.
    if call.when != "call":
        return
    fakes = _marker_args(item, "uses_fake")
    protocols = _marker_args(item, "exercises_protocol")
    item.user_properties.append(("fakes_used", fakes))
    item.user_properties.append(("protocols_exercised", protocols))


def register(config: pytest.Config, output_dir: Path | None = None) -> TelemetryPlugin:
    target = output_dir or Path(config.rootpath) / "tests" / ".telemetry"
    plugin = TelemetryPlugin(target)
    config.pluginmanager.register(plugin, name="ryotenkai-telemetry")
    return plugin
