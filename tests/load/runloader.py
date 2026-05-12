"""Skeleton ``RunLoader`` — лёгкий load-генератор.

Цель — "10 параллельных attempts × 30 stages each" из плана
[docs/plans/structured-hopping-starfish.md](../../docs/plans/structured-hopping-starfish.md)
строка L10. Текущий модуль — proof-of-pattern: интерфейс, минимальные
метрики, в Phase 5 будет расширен реальной интеграцией со стеком.

Конвенция:

* Сценарии складываются в :mod:`tests.load.runloader.scenarios`.
* Сценарий — это callable, который принимает ``RunLoaderConfig`` и
  возвращает один "attempt" в виде coroutine. RunLoader запускает их
  параллельно и собирает latency.
* Метрики формата ``LoadResult`` — JSON-сериализуемые, складываются в
  ``tests/.load_results/<run-id>.json``.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

# Один "запуск" пайплайна, который мы хотим тестировать на нагрузке.
AttemptFn = Callable[[int], Awaitable[None]]


@dataclass(frozen=True)
class RunLoaderConfig:
    """Параметры одного load-прохода."""

    concurrency: int
    """Сколько attempts стартуем одновременно (10 — default по плану)."""

    stages_per_attempt: int
    """Сколько stages в каждом attempt (30 — default по плану)."""

    timeout_seconds: float = 60.0
    """Глобальный timeout на весь load — защита от runaway."""


@dataclass
class LoadResult:
    """Результат одного RunLoader-прохода."""

    attempts_total: int
    attempts_succeeded: int
    attempts_failed: int
    duration_seconds: float
    latency_ms_per_attempt: list[float] = field(default_factory=list)

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latency_ms_per_attempt) if self.latency_ms_per_attempt else 0.0

    @property
    def p99_ms(self) -> float:
        if not self.latency_ms_per_attempt:
            return 0.0
        # 99-th percentile = sort and pick index ceil(0.99 * len)
        sorted_lat = sorted(self.latency_ms_per_attempt)
        idx = max(0, int(0.99 * len(sorted_lat)) - 1)
        return sorted_lat[idx]


class RunLoader:
    """Запускает N параллельных attempts и измеряет p50/p99 latency.

    Используется как pytest-фикстура в L10:
    ``RunLoader(cfg).run(my_attempt_fn)``.
    """

    def __init__(self, config: RunLoaderConfig) -> None:
        self._config = config

    async def run(self, attempt_fn: AttemptFn) -> LoadResult:
        """Запускает ``concurrency`` параллельных attempts.

        Каждый ``attempt_fn(stage_count)`` — независим. На исключении
        attempt считается failed; остальные продолжают работать.
        """
        started_at = time.monotonic()
        latencies: list[float] = []
        succeeded = 0
        failed = 0

        async def _one_attempt(idx: int) -> None:
            nonlocal succeeded, failed
            t0 = time.monotonic()
            try:
                await attempt_fn(self._config.stages_per_attempt)
                latencies.append((time.monotonic() - t0) * 1000)
                succeeded += 1
            except Exception:
                failed += 1

        try:
            await asyncio.wait_for(
                asyncio.gather(*[_one_attempt(i) for i in range(self._config.concurrency)]),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            # Не валим load целиком — фиксируем тайм-аут как failure.
            failed += self._config.concurrency - succeeded

        return LoadResult(
            attempts_total=self._config.concurrency,
            attempts_succeeded=succeeded,
            attempts_failed=failed,
            duration_seconds=time.monotonic() - started_at,
            latency_ms_per_attempt=latencies,
        )


__all__ = ["AttemptFn", "LoadResult", "RunLoader", "RunLoaderConfig"]
