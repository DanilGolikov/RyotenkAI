"""Golden-snapshot тест для ``ReportResponse``.

Стабилизированный (path/timestamp scrubbing) snapshot маленького отчёта,
который мы сериализуем в API. Любое изменение схемы или добавление полей
приведёт к diff'у в .ambr — это и есть наш ранний alarm.

Тест ставит ставку на ``syrupy`` (уже в окружении). Перегенерация
snapshots: ``pytest tests/golden/ --snapshot-update``.
"""

from __future__ import annotations

import re

import pytest

from ryotenkai_control.api.schemas.report import ReportResponse

pytestmark = pytest.mark.golden


# --- scrubbers --------------------------------------------------------------

_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")
_RUN_ID_RE = re.compile(r"r-\d{4,}")


def _scrub(payload: dict) -> dict:
    """Маскируем нестабильные поля до сравнения с baseline."""
    cleaned: dict = {}
    for key, value in payload.items():
        if isinstance(value, str):
            value = _TS_RE.sub("<TIMESTAMP>", value)
            value = _RUN_ID_RE.sub("<RUN_ID>", value)
        cleaned[key] = value
    return cleaned


# --- tests ------------------------------------------------------------------


def test_report_response_canonical_shape(snapshot) -> None:
    """Минимальный happy-path snapshot — фиксирует поля и порядок."""
    report = ReportResponse(
        path="reports/run-001/index.md",
        markdown="# Run summary\n\n- exit: success",
        generated_at="2026-05-12T12:00:00Z",
        regenerated=False,
    )
    payload = _scrub(report.model_dump())
    assert payload == snapshot


def test_report_response_with_regenerated_flag(snapshot) -> None:
    """Regenerated-режим даёт другой snapshot — мы хотим видеть, что
    флаг попадает в payload в правильной позиции."""
    report = ReportResponse(
        path="reports/run-002/index.md",
        markdown="# Regen\n\nUpdated.",
        generated_at="2026-05-12T13:00:00Z",
        regenerated=True,
    )
    payload = _scrub(report.model_dump())
    assert payload == snapshot
