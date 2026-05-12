def test_has_mp(monkeypatch) -> None:
    monkeypatch.setattr("some.module.value", 42)
    monkeypatch.setenv("FOO", "bar")
    result = "ok"
    assert result == "ok"
