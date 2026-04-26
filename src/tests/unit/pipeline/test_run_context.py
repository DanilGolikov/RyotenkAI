from __future__ import annotations

from datetime import datetime, timezone

import src.pipeline.state.run_context as rc
import src.utils.run_naming as rn


def test_generate_run_name_format_and_utc(monkeypatch) -> None:
    now = datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc)

    seq = iter("abc12")
    monkeypatch.setattr(rn.secrets, "choice", lambda _alphabet: next(seq))

    name, created_at = rn.generate_run_name(now_utc=now, id_length=5)

    assert name == "run_20260120_123456_abc12"
    assert created_at == now
    assert created_at.tzinfo == timezone.utc


def test_run_context_create_uses_generator(monkeypatch) -> None:
    now = datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(rc, "generate_run_name", lambda **_kw: ("run_20260120_000000_zz999", now))

    run = rc.RunContext.create()
    assert run.name == "run_20260120_000000_zz999"
    assert run.created_at_utc == now


def test_build_run_directory_uses_canonical_run_name(monkeypatch) -> None:
    now = datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(rn, "generate_run_name", lambda **_kw: ("run_20260120_000000_zz999", now))

    run_dir, created_at = rn.build_run_directory(base_dir=rn.Path("/tmp/runs"))

    assert run_dir == rn.Path("/tmp/runs/run_20260120_000000_zz999")
    assert created_at == now

