def test_something(monkeypatch) -> None:
    monkeypatch.setenv("FOO", "bar")
    result = "ok"
    assert result == "ok"
