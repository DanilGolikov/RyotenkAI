"""Property test: рандомные последовательности операций над FakePodLifecycleClient.

Свойства, которые валидируем:
1. **TERMINATED — поглощающее состояние**: после ``terminate`` любой
   последующий ``terminate``/``pause``/``resume`` детерминированно
   возвращает ``already_terminated`` / ``skipped`` / ``failed``.
2. **Идемпотентность ``terminate``**: повторный ``terminate`` всегда
   ``already_terminated``, никаких "case-sensitive" глюков.
3. **Нет NULL transitions**: каждое действие отдаёт ``LifecycleActionResult``
   с непустым ``outcome``.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings, strategies as st

from ryotenkai_shared.infrastructure.lifecycle import PodTerminalOutcome
from tests._fakes.lifecycle import FakePodLifecycleClient, PodState

pytestmark = pytest.mark.property


_actions = st.sampled_from(["terminate", "pause", "resume"])
_action_sequences = st.lists(_actions, min_size=1, max_size=10)


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.new_event_loop().run_until_complete(coro)


@given(actions=_action_sequences)
@settings(max_examples=50)
def test_terminated_is_absorbing(actions: list[str]) -> None:
    """Как только pod в ``TERMINATED``, никакое действие не возвращает
    pod в ``RUNNING``/``STOPPED``."""

    async def _scenario() -> None:
        client = FakePodLifecycleClient()
        client.register_pod("pod-1", state=PodState.RUNNING)
        # Принудительно загоняем в TERMINATED
        await client.terminate(resource_id="pod-1")
        assert client.get_pod_state("pod-1") == PodState.TERMINATED

        for action in actions:
            method = getattr(client, action)
            await method(resource_id="pod-1")
            # Состояние не сходит с TERMINATED
            assert client.get_pod_state("pod-1") == PodState.TERMINATED

    _run(_scenario())


@given(repeats=st.integers(min_value=1, max_value=5))
@settings(max_examples=20)
def test_terminate_idempotent(repeats: int) -> None:
    """Повторный ``terminate`` всегда отдаёт ``already_terminated``."""

    async def _scenario() -> None:
        client = FakePodLifecycleClient()
        client.register_pod("pod-x", state=PodState.RUNNING)

        first = await client.terminate(resource_id="pod-x")
        assert first.outcome == PodTerminalOutcome.TERMINATED

        for _ in range(repeats):
            result = await client.terminate(resource_id="pod-x")
            assert result.outcome == PodTerminalOutcome.ALREADY_TERMINATED

    _run(_scenario())


@given(actions=_action_sequences)
@settings(max_examples=50)
def test_outcomes_are_always_non_empty(actions: list[str]) -> None:
    """Никогда не получаем пустой ``outcome`` — это нарушает контракт
    ``LifecycleActionResult``."""

    async def _scenario() -> None:
        client = FakePodLifecycleClient()
        client.register_pod("pod-y", state=PodState.RUNNING)

        for action in actions:
            method = getattr(client, action)
            result = await method(resource_id="pod-y")
            assert isinstance(result.outcome, str)
            assert result.outcome, f"empty outcome for action={action!r}"

    _run(_scenario())
