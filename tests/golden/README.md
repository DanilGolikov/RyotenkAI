# L8 — Golden / snapshot tests

Snapshot-тесты на стабильность вывода: API-схем, отчётов, JSON-Schema
для FE, нормализованных pipeline-конфигов. Любой diff в snapshot ⇒
ручной review при merge — потому что либо это намеренное изменение
(коммит обновлённого baseline), либо регрессия.

## Инструмент: `syrupy`

Уже в окружении (`syrupy==5.1.0`). Каждый тест получает фикстуру
``snapshot``, и ``assert payload == snapshot`` либо сравнивает с
сохранённым `.ambr` в `__snapshots__/`, либо при `--snapshot-update`
перегенерирует его.

## Когда использовать snapshot vs обычный assertion?

Используй **snapshot**, если:

* Вывод имеет много полей и каждое из них одинаково важно (e.g. API
  response).
* Структура может расширяться (новые ключи) — diff на новый ключ
  заметим сразу.
* Reproducibility важна: regen происходит явно через флаг.

Используй **обычный assertion**, если:

* Проверяется одно конкретное поле/инвариант.
* Snapshot был бы пустой/тривиальный.

## Scrubber pattern

Snapshot'ы должны быть детерминированы. Любые time-, path-, run-id-
зависимые поля **обязательно** маскируем перед сравнением:

```python
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T...")
_RUN_ID_RE = re.compile(r"r-\d{4,}")

def _scrub(payload: dict) -> dict:
    cleaned = {}
    for key, value in payload.items():
        if isinstance(value, str):
            value = _TS_RE.sub("<TIMESTAMP>", value)
            value = _RUN_ID_RE.sub("<RUN_ID>", value)
        cleaned[key] = value
    return cleaned
```

Без scrubber'а snapshot flake'ает на каждую секунду.

## Регенерация workflow

```bash
# 1. Изменили production-код — snapshot падает.
pytest tests/golden/  # видим diff

# 2. Если diff намеренный — регенерируем baseline.
pytest tests/golden/ --snapshot-update

# 3. Внимательно проверяем diff в __snapshots__/*.ambr
git diff tests/golden/__snapshots__/

# 4. Commit. Reviewer обязательно смотрит .ambr diff.
```

В PR review обязательно вешать `pytest tests/golden/` в матрицу:
изменения `.ambr` без сопроводительного объяснения ⇒ блок merge.

## Текущее наполнение

| Файл | Покрывает |
|---|---|
| `test_run_report_snapshot.py` | `ReportResponse` API schema, regenerated-flag |
| `test_plugin_manifest_snapshot.py` | минимальный PluginManifest v5 dump |

## TODO

* `test_openapi_drift.py` — snapshot `openapi.json` для control API
  (важно для FE-zod-генерации).
* `test_pipeline_config_normalization.py` — несколько канонических
  pipeline-конфигов после прохождения через нормализатор.
* `test_journal_events_schema.py` — JSON-Schema event'ов журнала.
* `test_ws_events_schema.py` — control WebSocket events schema.

## Конвенции

* Все snapshot-тесты помечены `pytestmark = pytest.mark.golden`.
* Snapshot-каталог — `__snapshots__/` рядом с test-файлом (default syrupy).
* **Не коммитить `.ambr` с timestamp'ами / UUID'ами** — обязательно
  scrub'ать.
* **Один тест = один snapshot** для максимально читаемого diff.
