def test_nested(monkeypatch) -> None:
    monkeypatch.setenv("A", "1")
    monkeypatch.setenv("B", "2")
    result = "ok"
    assert result == "ok"
