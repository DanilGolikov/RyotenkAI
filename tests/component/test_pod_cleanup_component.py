"""L2 component test для :class:`RunPodCleanupManager`.

SUT: :class:`RunPodCleanupManager` — terminate-with-bounded-retry поверх
RunPod API. Все коллабораторы — fakes:

* ``api_client`` — узкий stub поверх :class:`FakeRunPodAPI`, который
  отвечает за ``terminate_pod`` (Result-стиль).
* ``sleep_fn`` — заменяется на счётчик, так что тесты не блокируются
  реальным sleep и могут проверять backoff-схему.

Три сценария: happy / retry-then-success / exhaust-retries.
"""

from __future__ import annotations

import pytest

from ryotenkai_providers.runpod.training.cleanup_manager import RunPodCleanupManager
from ryotenkai_shared.utils.result import Err, Ok, ProviderError
from tests._fakes.runpod import FakeRunPodAPI

pytestmark = pytest.mark.component


# --- адаптер: SUT ожидает sync метод ``terminate_pod(pod_id) -> Result`` -----


class _SyncRunPodAdapter:
    """SUT — синхронный, FakeRunPodAPI — async. Адаптер хранит
    предзаписанные результаты и крутит их по очереди."""

    def __init__(self, results: list[object]) -> None:
        # каждый результат — либо Ok(None), либо Err(ProviderError(...))
        self._results = list(results)
        self.calls: list[str] = []

    def terminate_pod(self, pod_id: str) -> object:
        self.calls.append(pod_id)
        if not self._results:
            raise AssertionError("test exhausted prepared results without expectation")
        return self._results.pop(0)


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def sleep_recorder() -> tuple[list[float], object]:
    """Record sleeps без реального ожидания."""
    recorded: list[float] = []

    def sleep_fn(seconds: float) -> None:
        recorded.append(seconds)

    return recorded, sleep_fn


# --- happy path --------------------------------------------------------------


class TestPositive:
    def test_first_attempt_succeeds(
        self, sleep_recorder: tuple[list[float], object],
    ) -> None:
        recorded, sleep_fn = sleep_recorder
        adapter = _SyncRunPodAdapter([Ok(None)])
        mgr = RunPodCleanupManager(adapter, sleep_fn=sleep_fn)  # type: ignore[arg-type]

        result = mgr.cleanup_pod("pod-1")

        assert result.is_ok()
        assert adapter.calls == ["pod-1"]
        # Sleep не должен звучать — успех с первой попытки.
        assert recorded == []


# --- retry then success ------------------------------------------------------


class TestRetryThenSuccess:
    def test_retry_after_transient_failure(
        self, sleep_recorder: tuple[list[float], object],
    ) -> None:
        recorded, sleep_fn = sleep_recorder
        adapter = _SyncRunPodAdapter(
            [
                Err(ProviderError(message="5xx blip", code="RUNPOD_TRANSIENT")),
                Ok(None),
            ],
        )
        mgr = RunPodCleanupManager(
            adapter,  # type: ignore[arg-type]
            initial_backoff_s=2.0,
            sleep_fn=sleep_fn,
        )

        result = mgr.cleanup_pod("pod-2")

        assert result.is_ok()
        assert adapter.calls == ["pod-2", "pod-2"]
        # Один sleep между попытками — initial_backoff.
        assert recorded == [2.0]


# --- exhausted retries — last error surfaces --------------------------------


class TestRetryExhausted:
    def test_all_attempts_fail_returns_last_error(
        self, sleep_recorder: tuple[list[float], object],
    ) -> None:
        recorded, sleep_fn = sleep_recorder
        adapter = _SyncRunPodAdapter(
            [
                Err(ProviderError(message="first", code="RUNPOD_TRANSIENT")),
                Err(ProviderError(message="second", code="RUNPOD_TRANSIENT")),
                Err(ProviderError(message="third", code="RUNPOD_AUTH")),
            ],
        )
        mgr = RunPodCleanupManager(
            adapter,  # type: ignore[arg-type]
            max_attempts=3,
            initial_backoff_s=1.0,
            sleep_fn=sleep_fn,
        )

        result = mgr.cleanup_pod("pod-3")

        assert result.is_err()
        # Последний error — из третьей попытки, не из первой.
        err = result.unwrap_err()
        assert err.code == "RUNPOD_AUTH"
        assert "third" in err.message
        # 3 попытки + 2 sleep'а между ними (exponential backoff: 1s, 2s).
        assert len(adapter.calls) == 3
        assert recorded == [1.0, 2.0]


def test_fake_runpod_smoke() -> None:
    """Sanity check, что FakeRunPodAPI остаётся импортируемой —
    в более глубоких component-тестах FakeRunPodAPI выступает SUT-fake
    для async-кода, и мы хотим, чтобы импорт оставался дёшев."""
    api = FakeRunPodAPI()
    api.upsert_pod("pod-x")
    assert "pod-x" in api.snapshot()["pods"]
