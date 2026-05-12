from unittest.mock import patch


def test_mixed(monkeypatch) -> None:
    with patch("some.module.target") as mock_target:
        monkeypatch.setenv("FOO", "bar")
        mock_target.return_value = 1
        assert True
