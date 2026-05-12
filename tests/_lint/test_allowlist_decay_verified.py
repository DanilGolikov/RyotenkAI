"""Meta-test: verify the 365-day allowlist decay mechanism actually fires.

The decay test in ``test_no_protocol_mocking.py::test_allowlist_entries_renewed_within_365_days``
walks the REAL allowlist and asserts every entry has been renewed within
365 days. That guards against stale entries accumulating.

This meta-test verifies the DECAY LOGIC itself: given a synthetic
allowlist entry with ``renewed`` 400 days in the past, the decay test
SHOULD fail (i.e. the staleness check works). If the decay logic ever
breaks (e.g. someone changes 365 → 36500, or removes the check), this
meta-test catches it.

Approach: don't modify the real allowlist. Instead, construct a fake
``AllowlistEntry`` list, monkey-patch the entry source, and invoke the
real check function. We assert that it raises.
"""

from __future__ import annotations

import importlib
from datetime import date, timedelta

import pytest


def test_decay_fires_on_stale_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400-day-old `renewed` date must make the decay check fail.

    If this test passes (decay correctly raises), we know the production
    sentinel will catch a future stale entry. If it FAILS (decay didn't
    raise), then the decay mechanism is broken.
    """
    sentinel_mod = importlib.import_module("tests._lint.test_no_protocol_mocking")
    allowlist_mod = importlib.import_module("tests._lint._mock_allowlist")

    stale_renewed = (date.today() - timedelta(days=400)).isoformat()
    fake_entry = allowlist_mod.AllowlistEntry(
        path="tests/synthetic_decay_probe.py",
        line=1,
        pattern="async_mock",
        reason="synthetic — verifies decay mechanism",
        added=stale_renewed,
        renewed=stale_renewed,
        owner="decay-meta-test",
    )

    monkeypatch.setattr(sentinel_mod, "_allowlist_entries", lambda: [fake_entry])

    with pytest.raises(AssertionError) as exc_info:
        sentinel_mod.test_allowlist_entries_renewed_within_365_days()

    msg = str(exc_info.value)
    assert "Stale allowlist entries" in msg
    assert fake_entry.path in msg
    assert "decay-meta-test" not in msg or "400d" in msg  # age noted in some form


def test_decay_passes_on_fresh_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A renewed-today entry must NOT trigger the decay check."""
    sentinel_mod = importlib.import_module("tests._lint.test_no_protocol_mocking")
    allowlist_mod = importlib.import_module("tests._lint._mock_allowlist")

    today_iso = date.today().isoformat()
    fresh_entry = allowlist_mod.AllowlistEntry(
        path="tests/synthetic_fresh_probe.py",
        line=1,
        pattern="async_mock",
        reason="synthetic — verifies decay does NOT false-positive",
        added=today_iso,
        renewed=today_iso,
        owner="decay-meta-test",
    )

    monkeypatch.setattr(sentinel_mod, "_allowlist_entries", lambda: [fresh_entry])

    # Must NOT raise — fresh entry is within window.
    sentinel_mod.test_allowlist_entries_renewed_within_365_days()
