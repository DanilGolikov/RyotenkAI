from unittest.mock import MagicMock


class Service:
    def do(self) -> int:
        return 1


def test_skip_with_spec() -> None:
    svc = MagicMock(spec=Service)
    assert svc is not None
