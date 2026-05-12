# L2 — Component tests

Один production-класс — System-Under-Test, ВСЕ коллабораторы заменены
каноническими **fakes** (никогда `unittest.mock`). Цель: проверить
поведение класса в изоляции, но с реалистичными коллабораторами,
которые ведут state machine (а не возвращают `MagicMock`).

## Конвенция

```python
import pytest
from tests._fakes.mlflow import FakeMLflowManager
from ryotenkai_control.<module> import <Class>

pytestmark = pytest.mark.component


def test_happy_path() -> None:
    mlflow = FakeMLflowManager()
    mlflow.setup()
    sut = <Class>(mlflow_manager=mlflow)
    ...
```

## Что отличает L2 от L1 (unit)?

| Аспект | L1 (unit) | L2 (component) |
|---|---|---|
| Класс | один метод/функция | один класс целиком |
| Коллабораторы | минимальные mocks | canonical fakes |
| State | без состояния | state machine fake |
| Цель | логика метода | контракт класса |
| Время | <100ms | <500ms |

## Канонические fakes

| Fake | Protocol | Файл |
|---|---|---|
| `FakeMLflowManager` | `IMLflowManager` | [`tests/_fakes/mlflow.py`](../_fakes/mlflow.py) |
| `FakePodLifecycleClient` | `IPodLifecycleClient` | [`tests/_fakes/lifecycle.py`](../_fakes/lifecycle.py) |
| `FakeRunPodAPI` | `IRunPodAPI` | [`tests/_fakes/runpod.py`](../_fakes/runpod.py) |
| `FakeSSH` | `ISSHClient` | [`tests/_fakes/ssh.py`](../_fakes/ssh.py) |
| `FakeHFHub` | `IHFHubClient` | [`tests/_fakes/hf_hub.py`](../_fakes/hf_hub.py) |
| `FakeTrainer` | `ITrainerSpawner` | [`tests/_fakes/trainer.py`](../_fakes/trainer.py) |
| `FakeJobClient` | `IJobClient` | [`tests/_fakes/job_client.py`](../_fakes/job_client.py) |
| `FakeProviderContext` | `IProviderContext` | [`tests/_fakes/provider_context.py`](../_fakes/provider_context.py) |

Все fakes принимают опциональный `clock=ManualClock()` — детерминированное
время через [`tests/_harness/clock.py`](../_harness/clock.py).

## Категории тестов в каждом файле

Минимально — три класса:

```python
class TestPositive:
    """Happy path: всё работает как ожидается."""

class TestNegative:
    """Сбой коллаборатора, ошибка — SUT обрабатывает корректно."""

class TestBoundary:
    """Пустой ввод, граничный случай, отсутствующее состояние."""
```

Опционально:

* `TestRegression` — для воспроизведения конкретных багов.
* `TestInvariants` — для проверки property-like инвариантов.

## Когда адаптер нужен

Иногда SUT ожидает узкий Protocol (e.g. ``log_metric`` only), а
canonical fake выдаёт более широкий surface (``IMLflowManager``). В
этом случае напишите тонкий адаптер в тесте (см.
[`test_metrics_replay_component.py`](test_metrics_replay_component.py)
— `_MlflowLogMetricAdapter`) — это нормально. Адаптер
переиспользовать между файлами **нельзя**: он специфичен для
SUT-семантики.

## Текущее наполнение

| Файл | SUT | Fakes |
|---|---|---|
| `test_metrics_replay_component.py` | `BufferedMetricsReplay` | `FakeMLflowManager` (через адаптер) |
| `test_pod_cleanup_component.py` | `RunPodCleanupManager` | `FakeRunPodAPI` + adapter-stub |

## TODO

* `test_summary_reporter_component.py` — `ExecutionSummaryReporter`
  с `FakeMLflowManager`.
* `test_pod_terminator_component.py` — `PodTerminator` с
  `FakePodLifecycleClient`.
* `test_idle_detector_component.py` — `IdleDetector` с canonical fakes.
* `test_attempt_controller_component.py` — `AttemptController` с
  лестницей fakes.

## Hard rules

* `unittest.mock.patch` запрещён.
* `MagicMock()` запрещён в качестве коллаборатора.
* `time.sleep` запрещён — используйте `ManualClock` + `clock.sleep`.
* Тест помечен `pytestmark = pytest.mark.component`.
