from unittest.mock import patch


def test_clear_true() -> None:
    with patch.dict("os.environ", {"FOO": "bar"}, clear=True):
        result = "ok"
        assert result == "ok"
