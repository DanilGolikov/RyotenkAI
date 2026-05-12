from unittest.mock import patch


def test_has_mp(monkeypatch) -> None:
    monkeypatch.setattr("some.module.value", 42)
    with patch.dict("os.environ", {"FOO": "bar"}):
        result = "ok"
        assert result == "ok"
