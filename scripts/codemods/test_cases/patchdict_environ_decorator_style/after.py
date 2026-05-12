def test_decorated(monkeypatch) -> None:
    monkeypatch.setenv("FOO", "bar")
    assert True
