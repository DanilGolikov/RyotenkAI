"""``MagicMock(return_value=...)`` — mock-only kwarg, must be preserved."""

from unittest.mock import MagicMock


def test_skip_kwargs_construct() -> None:
    fn = MagicMock(return_value=42)
    assert fn() == 42
