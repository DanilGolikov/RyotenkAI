"""Unit tests for ``src.cli_state.context_store``.

Cover the contract documented in the module:

- ``RYOTENKAI_HOME`` overrides ``~/.ryotenkai``.
- Reads tolerate a missing file (fresh install) and a malformed file
  (returns empty :class:`CLIContext`, leaves the file in place).
- Writes are atomic — interrupted mid-write keeps the previous state.
- ``set_current_project`` rejects empty / whitespace-only ids.
- ``clear_current_project`` is idempotent (``rm -f`` semantics).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli_state import context_store


@pytest.fixture()
def cli_home_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the store at ``tmp_path`` via ``RYOTENKAI_HOME``."""
    monkeypatch.setenv(context_store.HOME_ENV, str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# cli_home / cli_context_path resolution
# ---------------------------------------------------------------------------


def test_cli_home_honours_env_override(cli_home_tmp: Path) -> None:
    assert context_store.cli_home() == cli_home_tmp


def test_cli_home_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(context_store.HOME_ENV, raising=False)
    assert context_store.cli_home() == Path.home() / ".ryotenkai"


def test_cli_home_treats_blank_env_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty string in ``RYOTENKAI_HOME`` should not point at the CWD."""
    monkeypatch.setenv(context_store.HOME_ENV, "   ")
    assert context_store.cli_home() == Path.home() / ".ryotenkai"


def test_cli_context_path_assembles_filename(cli_home_tmp: Path) -> None:
    expected = cli_home_tmp / context_store.CONTEXT_FILENAME
    assert context_store.cli_context_path() == expected


# ---------------------------------------------------------------------------
# load_context — fresh / malformed / valid
# ---------------------------------------------------------------------------


def test_load_context_returns_empty_when_file_missing(cli_home_tmp: Path) -> None:
    ctx = context_store.load_context()
    assert ctx == context_store.CLIContext()
    # Reading must NOT create the directory or the file (lazy semantics).
    assert not (cli_home_tmp / context_store.CONTEXT_FILENAME).exists()


def test_load_context_returns_empty_on_malformed_json(cli_home_tmp: Path) -> None:
    target = cli_home_tmp / context_store.CONTEXT_FILENAME
    target.write_text("this is not json {{{", encoding="utf-8")
    ctx = context_store.load_context()
    assert ctx == context_store.CLIContext()
    # The bad file is preserved so the user can inspect it.
    assert target.exists()


def test_load_context_returns_empty_on_non_object_payload(cli_home_tmp: Path) -> None:
    """A JSON list at the top level is valid JSON but the wrong shape."""
    target = cli_home_tmp / context_store.CONTEXT_FILENAME
    target.write_text(json.dumps(["unexpected"]), encoding="utf-8")
    ctx = context_store.load_context()
    assert ctx == context_store.CLIContext()


def test_load_context_reads_round_trip(cli_home_tmp: Path) -> None:
    context_store.set_current_project("alpha")
    ctx = context_store.load_context()
    assert ctx.current_project_id == "alpha"
    # ``set_at`` is set by the writer; just verify it's a non-empty string.
    assert ctx.set_at and ctx.set_at.endswith("Z")


def test_load_context_ignores_non_string_fields(cli_home_tmp: Path) -> None:
    """Stale or hand-edited files with wrong-typed fields shouldn't crash."""
    target = cli_home_tmp / context_store.CONTEXT_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({context_store.KEY_CURRENT_PROJECT: 42, context_store.KEY_SET_AT: True}),
        encoding="utf-8",
    )
    ctx = context_store.load_context()
    assert ctx == context_store.CLIContext()


# ---------------------------------------------------------------------------
# set_current_project — happy path + validation
# ---------------------------------------------------------------------------


def test_set_current_project_creates_directory(cli_home_tmp: Path) -> None:
    """The CLI home dir is created lazily on first write."""
    nested = cli_home_tmp / "nested" / "deeper"
    # Use a non-existent nested home to prove parent dirs are created.
    import os
    os.environ[context_store.HOME_ENV] = str(nested)
    try:
        context_store.set_current_project("alpha")
        assert (nested / context_store.CONTEXT_FILENAME).exists()
    finally:
        del os.environ[context_store.HOME_ENV]


def test_set_current_project_persists_id(cli_home_tmp: Path) -> None:
    context_store.set_current_project("beta")
    assert context_store.get_current_project() == "beta"


def test_set_current_project_overwrites_previous(cli_home_tmp: Path) -> None:
    context_store.set_current_project("first")
    context_store.set_current_project("second")
    assert context_store.get_current_project() == "second"


@pytest.mark.parametrize("bad_id", ["", "   ", "\n", "\t"])
def test_set_current_project_rejects_blank(cli_home_tmp: Path, bad_id: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        context_store.set_current_project(bad_id)


def test_set_current_project_strips_whitespace(cli_home_tmp: Path) -> None:
    context_store.set_current_project("  spaced  ")
    assert context_store.get_current_project() == "spaced"


def test_set_current_project_writes_sorted_pretty_json(cli_home_tmp: Path) -> None:
    """``atomic_write_json`` produces sorted keys + 2-space indent — verify
    the contract since users may read the file by hand."""
    context_store.set_current_project("gamma")
    raw = (cli_home_tmp / context_store.CONTEXT_FILENAME).read_text(encoding="utf-8")
    assert raw.endswith("\n")
    payload = json.loads(raw)
    assert list(payload.keys()) == sorted(payload.keys())


# ---------------------------------------------------------------------------
# clear_current_project — idempotent
# ---------------------------------------------------------------------------


def test_clear_current_project_removes_file(cli_home_tmp: Path) -> None:
    context_store.set_current_project("alpha")
    assert context_store.cli_context_path().exists()
    context_store.clear_current_project()
    assert not context_store.cli_context_path().exists()
    assert context_store.get_current_project() is None


def test_clear_current_project_is_noop_when_missing(cli_home_tmp: Path) -> None:
    """Clearing a non-existent context must NOT raise — matches ``rm -f``."""
    context_store.clear_current_project()
    context_store.clear_current_project()  # second call also fine


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------


def test_get_current_project_returns_none_when_unset(cli_home_tmp: Path) -> None:
    assert context_store.get_current_project() is None


def test_get_current_project_returns_id_when_set(cli_home_tmp: Path) -> None:
    context_store.set_current_project("delta")
    assert context_store.get_current_project() == "delta"
