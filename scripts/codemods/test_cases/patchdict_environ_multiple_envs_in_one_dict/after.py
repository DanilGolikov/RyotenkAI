def test_multi(monkeypatch) -> None:
    monkeypatch.setenv("FOO", "bar")
    monkeypatch.setenv("BAZ", "qux")
    result = "ok"
    assert result == "ok"
