"""Tests for ``PatchDictEnvironToMonkeypatchCodemod``.

Each test loads a ``before.py`` / ``after.py`` pair from
``scripts/codemods/test_cases/patchdict_environ_<scenario>/`` and asserts
the codemod transforms one into the other.  ``before.py`` and ``after.py``
are the *ground truth* — fixtures live with the codemod, not in the test
file, so TDD on new scenarios is "add a directory + assertion".
"""

from __future__ import annotations

from pathlib import Path

from libcst.codemod import CodemodTest

from scripts.codemods.patchdict_environ_to_monkeypatch import (
    PatchDictEnvironToMonkeypatchCodemod,
)


_CASES_DIR = Path(__file__).resolve().parent.parent / "test_cases"


def _load(name: str) -> tuple[str, str]:
    before = (_CASES_DIR / name / "before.py").read_text()
    after = (_CASES_DIR / name / "after.py").read_text()
    return before, after


class TestPatchDictEnvironToMonkeypatch(CodemodTest):
    TRANSFORM = PatchDictEnvironToMonkeypatchCodemod

    # --- conversions ----------------------------------------------------

    def test_simple_context_manager(self) -> None:
        """``with patch.dict("os.environ", {"K": "v"}): body`` → setenv + body."""

        before, after = _load("patchdict_environ_simple_context_manager")
        self.assertCodemod(before, after)

    def test_decorator_style(self) -> None:
        """``@patch.dict(...)`` decorator becomes setenv at top of body."""

        before, after = _load("patchdict_environ_decorator_style")
        self.assertCodemod(before, after)

    def test_multiple_envs_in_one_dict(self) -> None:
        """A dict with multiple keys emits one setenv per key."""

        before, after = _load("patchdict_environ_multiple_envs_in_one_dict")
        self.assertCodemod(before, after)

    def test_nested_patches(self) -> None:
        """Compound ``with A, B:`` with two patch.dict items → two setenvs."""

        before, after = _load("patchdict_environ_nested_patches")
        self.assertCodemod(before, after)

    def test_existing_monkeypatch_fixture(self) -> None:
        """Function already has ``monkeypatch`` param — don't double-add."""

        before, after = _load("patchdict_environ_existing_monkeypatch_fixture")
        self.assertCodemod(before, after)

    def test_removes_unused_patch_import(self) -> None:
        """``from unittest.mock import patch`` is dropped if patch is unused."""

        before, after = _load("patchdict_environ_removes_unused_patch_import")
        self.assertCodemod(before, after)

    def test_keeps_patch_import_if_other_patches_remain(self) -> None:
        """Keep ``patch`` import if non-dict patch usage remains."""

        before, after = _load(
            "patchdict_environ_keeps_patch_import_if_other_patches_remain",
        )
        self.assertCodemod(before, after)

    # --- skips ----------------------------------------------------------

    def test_skip_clear_true(self) -> None:
        """``clear=True`` cannot be mechanically converted; annotate TODO."""

        before, after = _load("patchdict_environ_skip_clear_true")
        self.assertCodemod(before, after)

    def test_skip_with_as_binding(self) -> None:
        """``with patch.dict(...) as env:`` binds a name; annotate TODO."""

        before, after = _load("patchdict_environ_skip_with_as_binding")
        self.assertCodemod(before, after)
