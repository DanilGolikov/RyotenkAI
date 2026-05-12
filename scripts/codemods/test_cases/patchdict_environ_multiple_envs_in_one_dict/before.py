from unittest.mock import patch


def test_multi() -> None:
    with patch.dict("os.environ", {"FOO": "bar", "BAZ": "qux"}):
        result = "ok"
        assert result == "ok"
