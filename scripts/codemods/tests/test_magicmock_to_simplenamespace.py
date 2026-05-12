"""Tests for ``MagicMockToSimpleNamespaceCodemod``.

Each test loads a ``before.py`` / ``after.py`` pair from
``scripts/codemods/test_cases/<scenario>/`` and asserts the codemod
transforms one into the other.  ``before.py`` and ``after.py`` are the
*ground truth* — fixtures live with the codemod, not in the test
file, so TDD on new scenarios is "add a directory + assertion".
"""

from __future__ import annotations

from pathlib import Path

from libcst.codemod import CodemodTest

from scripts.codemods.magicmock_to_simplenamespace import (
    MagicMockToSimpleNamespaceCodemod,
)


_CASES_DIR = Path(__file__).resolve().parent.parent / "test_cases"


def _load(name: str) -> tuple[str, str]:
    before = (_CASES_DIR / name / "before.py").read_text()
    after = (_CASES_DIR / name / "after.py").read_text()
    return before, after


class TestMagicMockToSimpleNamespace(CodemodTest):
    TRANSFORM = MagicMockToSimpleNamespaceCodemod

    # --- conversions ----------------------------------------------------

    def test_simple_data_carrier(self) -> None:
        before, after = _load("simple_data_carrier")
        self.assertCodemod(before, after)

    def test_multi_attr_data_carrier(self) -> None:
        before, after = _load("multi_attr_data_carrier")
        self.assertCodemod(before, after)

    def test_data_carrier_with_callable_attrs(self) -> None:
        """``runner = MagicMock(); runner.get_status = AsyncMock(...)`` pattern.

        This is the workhorse case for the control-plane router tests.
        """
        before, after = _load("data_carrier_with_callable_attrs")
        self.assertCodemod(before, after)

    def test_delete_unused_magicmock_import(self) -> None:
        """After the only ``MagicMock()`` is converted, drop the import."""
        before, after = _load("delete_unused_magicmock_import")
        self.assertCodemod(before, after)

    def test_reassignment_after_use(self) -> None:
        """Aliasing (``use(obj)``) disqualifies the candidate entirely."""
        before, after = _load("reassignment_after_use")
        self.assertCodemod(before, after)

    def test_post_use_attribute_mutation(self) -> None:
        """Read + post-use attribute write is safe (no aliasing)."""
        before, after = _load("post_use_attribute_mutation")
        self.assertCodemod(before, after)

    def test_skip_monkeypatch_setattr_target(self) -> None:
        """Candidate passed to ``monkeypatch.setattr`` → keep MagicMock."""
        before, after = _load("skip_monkeypatch_setattr_target")
        self.assertCodemod(before, after)

    def test_skip_interleaved_statement(self) -> None:
        """Interleaved statement closes the absorption window."""
        before, after = _load("skip_interleaved_statement")
        self.assertCodemod(before, after)

    # --- skips ----------------------------------------------------------

    def test_skip_callable(self) -> None:
        """``m()`` direct call → keep MagicMock semantics."""
        before, after = _load("skip_callable")
        self.assertCodemod(before, after)

    def test_skip_with_spec(self) -> None:
        """``MagicMock(spec=X)`` is out of Phase 2A scope."""
        before, after = _load("skip_with_spec")
        self.assertCodemod(before, after)

    def test_skip_with_return_value(self) -> None:
        """``m.return_value = ...`` indicates spy semantics → keep."""
        before, after = _load("skip_with_return_value")
        self.assertCodemod(before, after)

    def test_skip_with_assert_called(self) -> None:
        """``m.assert_called_with(...)`` is an interaction test → keep."""
        before, after = _load("skip_with_assert_called")
        self.assertCodemod(before, after)

    def test_skip_kwargs_construct(self) -> None:
        """``MagicMock(return_value=X)`` is mock-only kwargs → keep."""
        before, after = _load("skip_kwargs_construct")
        self.assertCodemod(before, after)

    def test_skip_chained_attr(self) -> None:
        """``m.foo.return_value = ...`` — chained mock-only attr → keep."""
        before, after = _load("skip_chained_attr")
        self.assertCodemod(before, after)

    def test_nested_attr_assignment(self) -> None:
        """``m.foo.bar = X`` — nested assignment cannot fold into kwargs.

        Conservative: leave the assignment as a MagicMock.
        """
        before, after = _load("nested_attr_assignment")
        self.assertCodemod(before, after)

    def test_inline_argument(self) -> None:
        """Inline ``MagicMock()`` (no assignment) is left untouched."""
        before, after = _load("inline_argument")
        self.assertCodemod(before, after)

    # --- additional invariants -----------------------------------------

    def test_idempotent(self) -> None:
        """Running the codemod twice yields the same result."""
        before, _ = _load("multi_attr_data_carrier")
        once = self._transform(before)
        twice = self._transform(once)
        assert once == twice

    def test_preserves_unrelated_code(self) -> None:
        """A file with no MagicMock is unchanged."""
        src = (
            "from __future__ import annotations\n\n"
            "def test_noop() -> None:\n"
            "    assert 1 == 1\n"
        )
        assert self._transform(src) == src

    # --- helper ---------------------------------------------------------

    def _transform(self, src: str) -> str:
        import libcst as cst
        from libcst.codemod import CodemodContext

        codemod = MagicMockToSimpleNamespaceCodemod(CodemodContext())
        tree = cst.parse_module(src)
        return codemod.transform_module_impl(tree).code
