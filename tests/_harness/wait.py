"""Eventually/Consistently — async polling helpers for L4+ tests.

Никогда не вызываем ``time.monotonic()`` напрямую: всё время идёт через
инжектированный ``Clock``. Это даёт детерминизм через ``ManualClock`` и
позволяет тестам контролировать прогресс fake-стека.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from tests._harness.clock import Clock, RealClock

_DEFAULT_CLOCK: Clock = RealClock()

ConditionResult = bool | Awaitable[bool]
Condition = Callable[[], ConditionResult]


async def _evaluate(condition: Condition) -> bool:
    result = condition()
    if inspect.isawaitable(result):
        return bool(await result)
    return bool(result)


async def Eventually(  # noqa: N802 -- spec API uses the capitalized name
    condition: Condition,
    *,
    timeout: float = 30.0,
    poll: float = 0.1,
    message: str | None = None,
    clock: Clock | None = None,
) -> None:
    """Опрашивает ``condition`` до truthy или таймаута.

    Returns silently on success. Raises ``TimeoutError`` on failure.
    """
    c = clock or _DEFAULT_CLOCK
    deadline = c.now() + timeout
    while True:
        if await _evaluate(condition):
            return
        if c.now() >= deadline:
            raise TimeoutError(message or f"Eventually timed out after {timeout}s")
        await c.sleep(poll)


async def Consistently(  # noqa: N802 -- spec API uses the capitalized name
    condition: Condition,
    *,
    duration: float = 2.0,
    poll: float = 0.1,
    message: str | None = None,
    clock: Clock | None = None,
) -> None:
    """Проверяет, что ``condition`` остаётся truthy ``duration`` секунд.

    Raises ``AssertionError`` на первой falsy проверке.
    """
    c = clock or _DEFAULT_CLOCK
    deadline = c.now() + duration
    while c.now() < deadline:
        if not await _evaluate(condition):
            raise AssertionError(message or f"Consistently flipped to false within {duration}s")
        await c.sleep(poll)
