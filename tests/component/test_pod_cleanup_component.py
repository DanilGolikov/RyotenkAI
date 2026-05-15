"""L2 component test для :class:`RunPodCleanupManager`.

SUT: :class:`RunPodCleanupManager` — terminate-with-bounded-retry поверх
RunPod API. Все коллабораторы — fakes:

* ``api_client`` — узкий stub поверх :class:`FakeRunPodAPI`, который
  отвечает за ``terminate_pod`` (raise-based: успех = None; ошибка =
  типизированный :class:`RyotenkAIError`).
* ``sleep_fn`` — заменяется на счётчик, так что тесты не блокируются
  реальным sleep и могут проверять backoff-схему.

Три сценария: happy / retry-then-success / exhaust-retries.

Phase A2 finale (2026-05-16): converted from legacy ``Result[None,
ProviderError]`` to raise-based contract. ``terminate_pod`` now returns
``None`` on success and raises a typed exception on failure;
``cleanup_pod`` re-raises the last typed exception on exhaustion.
"""

from __future__ import annotations

import pytest

from ryotenkai_providers.runpod.training.cleanup_manager import RunPodCleanupManager
from ryotenkai_shared.errors import ProviderUnavailableError
from tests._fakes.runpod import FakeRunPodAPI

pytestmark = pytest.mark.component


# --- адаптер: SUT ожидает sync метод ``terminate_pod(pod_id) -> None`` -------


class _SyncRunPodAdapter:
    """SUT — синхронный, FakeRunPodAPI — async. Адаптер хранит
    предзаписанные результаты и крутит их по очереди. ``None`` означает
    успех; экземпляр :class:`Exception` — ошибку (raise on call)."""

    def __init__(self, results: list[object]) -> None:
        # каждый результат — либо None (успех), либо исключение для raise
        self._results = list(results)
        self.calls: list[str] = []

    def terminate_pod(self, pod_id: str) -> None:
        self.calls.append(pod_id)
        if not self._results:
            raise AssertionError("test exhausted prepared results without expectation")
        nxt = self._results.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return None


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
        adapter = _SyncRunPodAdapter([None])
        mgr = RunPodCleanupManager(adapter, sleep_fn=sleep_fn)  # type: ignore[arg-type]

        out = mgr.cleanup_pod("pod-1")

        assert out is None
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
                ProviderUnavailableError(
                    detail="5xx blip", context={"code": "RUNPOD_TRANSIENT"}
                ),
                None,
            ],
        )
        mgr = RunPodCleanupManager(
            adapter,  # type: ignore[arg-type]
            initial_backoff_s=2.0,
            sleep_fn=sleep_fn,
        )

        out = mgr.cleanup_pod("pod-2")

        assert out is None
        assert adapter.calls == ["pod-2", "pod-2"]
        # Один sleep между попытками — initial_backoff.
        assert recorded == [2.0]


# --- exhausted retries — last error surfaces --------------------------------


class TestRetryExhausted:
    def test_all_attempts_fail_raises_last_error(
        self, sleep_recorder: tuple[list[float], object],
    ) -> None:
        recorded, sleep_fn = sleep_recorder
        adapter = _SyncRunPodAdapter(
            [
                ProviderUnavailableError(
                    detail="first", context={"code": "RUNPOD_TRANSIENT"}
                ),
                ProviderUnavailableError(
                    detail="second", context={"code": "RUNPOD_TRANSIENT"}
                ),
                ProviderUnavailableError(
                    detail="third", context={"code": "RUNPOD_AUTH"}
                ),
            ],
        )
        mgr = RunPodCleanupManager(
            adapter,  # type: ignore[arg-type]
            max_attempts=3,
            initial_backoff_s=1.0,
            sleep_fn=sleep_fn,
        )

        with pytest.raises(ProviderUnavailableError) as exc_info:
            mgr.cleanup_pod("pod-3")

        # Последний error — из третьей попытки, не из первой.
        assert exc_info.value.context.get("code") == "RUNPOD_AUTH"
        assert "third" in (exc_info.value.detail or "")
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
