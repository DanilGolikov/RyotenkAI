"""Candidate passed to ``monkeypatch.setattr`` / ``patch.object`` etc.

The candidate is now bound to ``target.attr``.  Production code in the
system under test may use Mock-only semantics on ``target.attr``
(``.return_value``, ``.assert_called_with``).  Since the codemod can't
see across files, we conservatively keep MagicMock semantics.
"""

from unittest.mock import MagicMock

import pytest


def test_skip_setattr(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.value = 1
    monkeypatch.setattr("module.thing", fake)
    assert fake.value == 1
