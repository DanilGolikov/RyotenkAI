# L7 — Property-based tests (Hypothesis)

Свойственные (property-based) тесты на инварианты парсеров, валидаторов,
schema-сериализаторов. Для коробки фактов используем
[Hypothesis](https://hypothesis.readthedocs.io).

## Когда писать property вместо unit?

Используй **property test**, если для функции/класса можно сформулировать
инвариант формата "для любого валидного входа X выполняется Y", и
генерация X не комбинаторно дорога. Типичные кандидаты:

* **Round-trip сериализации** — ``dict → Model → model_dump`` идентично.
* **Парсеры**: вход ∈ грамматика ⇒ выход в форме A ⇒ обратно собирается.
* **Идемпотентность**: f(f(x)) == f(x).
* **State machine invariants**: терминальные состояния поглощают,
  переходы — детерминированы.

Используй **обычный unit test**, если:

* Свойство специфично для одного входа («при пустом списке кидаем X»).
* Hypothesis не справится с генерацией без сложных предусловий.
* Регрессия — фиксированный кейс, поведение на других входах не важно.

## Профили Hypothesis

Профили зарегистрированы в [`tests/conftest.py`](../conftest.py):

| Профиль | `max_examples` | Где запускается |
|---|---|---|
| `ci` (default) | 50 | `presubmit-blocking` (PR) |
| `nightly` | 5000 | `.github/workflows/nightly.yml` |

Переключение: `HYPOTHESIS_PROFILE=nightly pytest tests/property/`.

## Конвенции

* Все файлы помечены `pytestmark = pytest.mark.property`.
* Стратегии (`st.builds`, `st.composite`) — на уровне модуля, по
  возможности переиспользуемые между тестами.
* **Дет-сидинг**: `RYOTENKAI_TEST_SEED` (default `0`) глобально пробрасывается
  в `pytest_configure` (см. [tests/conftest.py](../conftest.py)).
* **Async-страт**: оборачивать в обычный sync, внутри звать
  `asyncio.new_event_loop().run_until_complete(coro)` — Hypothesis
  + pytest-asyncio не дружат на функциях `@given` напрямую.
* Падение на seed → копировать failing example в обычный unit test
  для быстрой регрессии.

## Текущее наполнение

| Файл | Покрывает |
|---|---|
| `test_runpod_info_invariants.py` | dataclass `RunPodInfo` round-trip, диапазон портов, hashability |
| `test_manifest_schema_roundtrip.py` | `PluginManifest` round-trip, schema version, default name |
| `test_lifecycle_state_machine.py` | `FakePodLifecycleClient` — terminated absorbing, idempotency |

## TODO

* `test_journal_event_roundtrip.py` — JSONL events / журналирование.
* `test_pipeline_config_normalization.py` — нормализация pipeline-конфигов
  идемпотентна.
* `test_pod_snapshot_parse.py` — `_normalize_ports` устойчив к любому
  валидному JSON-payload'у RunPod.
