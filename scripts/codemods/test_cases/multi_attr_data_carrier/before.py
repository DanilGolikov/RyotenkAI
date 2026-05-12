from unittest.mock import MagicMock


def test_multi_attr() -> None:
    cfg = MagicMock()
    cfg.name = "alpha"
    cfg.count = 3
    cfg.enabled = True
    assert cfg.name == "alpha"
    assert cfg.count == 3
    assert cfg.enabled is True
