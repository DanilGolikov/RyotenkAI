"""Sentinel: :class:`ErrorCode` enum never loses members.

Per RFC 9457 + project policy, every emitted error code is a stable,
client-pinned identifier. Frontend dashboards group on it, the CLI
``ryotenkai`` switches on it for exit-code mapping, third-party
scripts pin on it. Silently deleting an enum member breaks all of
those at runtime — the deletion would only surface when a real failure
of that kind is hit in production.

This sentinel maintains a snapshot of every ``ErrorCode`` value that
has ever existed in
:file:`tests/_lint/error_code_history.yaml`. Two complementary checks:

1. :func:`test_no_error_code_was_removed` — block PRs that delete
   members. Snapshot is the source of truth; current enum must be a
   superset.
2. :func:`test_history_includes_every_current_code` — block PRs that
   add a new member without registering it in the history file in the
   same PR.

To deprecate a code without breaking the contract: leave the enum
member in place, mark it deprecated in the docstring, and stop the
production code from raising it. The history file never loses entries.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ryotenkai_shared.contracts.problem_details import ErrorCode


_HISTORY_FILE = Path(__file__).parent / "error_code_history.yaml"


def _load_history() -> set[str]:
    data = yaml.safe_load(_HISTORY_FILE.read_text(encoding="utf-8"))
    return set(data["codes"])


def test_no_error_code_was_removed() -> None:
    """Block PRs that delete ``ErrorCode`` members.

    Clients (frontend, CLI, scripts) pin on these values. Removing a
    code silently breaks them. To deprecate without removing, leave
    the enum member in place and mark it deprecated in the docstring.
    """
    expected = _load_history()
    actual = {c.value for c in ErrorCode}
    missing = expected - actual
    assert not missing, (
        f"ErrorCode members were deleted: {sorted(missing)}. "
        f"This breaks clients pinning on these codes. "
        f"To deprecate without removing, leave the enum member in "
        f"place and mark it deprecated in the docstring."
    )


def test_history_includes_every_current_code() -> None:
    """Block adding a new :class:`ErrorCode` without updating the history.

    The history file is the canonical record of every code that has
    ever been part of the public surface. A new enum member must be
    registered in the same PR so a future deletion fails this
    sentinel rather than slipping through unnoticed.
    """
    expected = _load_history()
    actual = {c.value for c in ErrorCode}
    new = actual - expected
    assert not new, (
        f"New ErrorCode members not registered in error_code_history.yaml: "
        f"{sorted(new)}. Add them to the history file in this PR so a "
        f"future deletion would fail test_no_error_code_was_removed."
    )


def test_history_file_yaml_is_well_formed() -> None:
    """The history YAML parses, has the expected key, and contains strings only.

    Defensive check so a hand-edit that breaks YAML syntax or wraps the
    entries in the wrong key fails this sentinel directly rather than
    masking as "no codes were removed".
    """
    raw = _HISTORY_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), f"history root must be a dict, got {type(data)!r}"
    assert "codes" in data, "history must contain top-level 'codes' key"
    codes = data["codes"]
    assert isinstance(codes, list), f"'codes' must be a list, got {type(codes)!r}"
    for code in codes:
        assert isinstance(code, str), f"every history entry must be a str, got {code!r}"
        assert code.isupper() or "_" in code, (
            f"history entry {code!r} does not look like UPPER_SNAKE_CASE"
        )


def test_history_entries_are_unique() -> None:
    """No duplicates in the history file.

    Defensive: a hand-edit that pastes the same code twice would mask
    a deletion (set-comparison short-circuits duplicates). Catch the
    typo before it can mask anything.
    """
    raw = _HISTORY_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    codes_list = data["codes"]
    duplicates = [c for c in set(codes_list) if codes_list.count(c) > 1]
    assert not duplicates, f"duplicate entries in history file: {sorted(duplicates)}"


def test_history_matches_current_enum_exactly() -> None:
    """At a clean snapshot (no pending add/remove), history == enum.

    This is a stronger check that catches mid-PR drift: a PR that adds
    a new code WITHOUT updating the history fails
    :func:`test_history_includes_every_current_code`. A PR that removes
    a code fails :func:`test_no_error_code_was_removed`. This test
    rolls both into a single equality assertion, giving the clearest
    diff when CI fails. The two single-direction tests above stay for
    targeted error messages.
    """
    expected = _load_history()
    actual = {c.value for c in ErrorCode}
    only_in_history = sorted(expected - actual)
    only_in_enum = sorted(actual - expected)
    assert expected == actual, (
        f"ErrorCode enum drifted from history snapshot.\n"
        f"  Deleted from enum (must be restored): {only_in_history}\n"
        f"  Added to enum (must be added to history): {only_in_enum}"
    )
