# План дочистки архитектуры TUI

## Цель
- Довести текущее расслоение до состояния, где `src/tui/screens/*` и TUI-side adapters не зависят от pipeline presentation/helper логики.
- Оставить в `pipeline` только нейтральные state/application/use-case контракты.
- Разделить:
  - `pipeline facts / use-cases`
  - `TUI presentation`
  - `CLI presentation`

## Что уже сделано
- `delete` вынесен в TUI-side backend и больше не зависит от `src/pipeline.run_deletion`.
- `RunsList`, `RunDetail`, `AttemptDetail`, `LaunchModal`, `apps.py`, `launch.py`, `launch_state.py` переведены на `src/tui/adapters/*`.
- В `pipeline_state.json` добавлены optional-поля:
  - `mlflow_runtime_tracking_uri`
  - `mlflow_ca_bundle_path`
- Для старых `pipeline_state.json` удаление не падает: папка удаляется, а отсутствие MLflow URI показывается как warning.

## Что еще не идеально
- `src/pipeline/run_inspector.py` все еще смешивает:
  - чтение state/logs
  - diff helpers
  - summary rows
  - CLI rendering
  - presentation helpers вроде иконок/длительностей
- `src/tui/adapters/launch_backend.py` все еще зависит от:
  - `src.pipeline.restart_points`
  - `src.utils.config.load_config`
  - `compute_config_hashes()`

## Финальная целевая граница
- `pipeline` должен отдавать только:
  - загруженный state
  - результаты query/use-case
  - restart/resume decisions
  - нейтральные данные без UI-оформления
- `TUI` должен:
  - читать только TUI adapters
  - сам строить presentation
  - не знать про внутренние helper-модули pipeline
- `CLI` должен:
  - использовать pipeline queries/use-cases
  - иметь свой rendering слой

## Этап 1. Разрезать `run_inspector`
- Выделить из [`src/pipeline/run_inspector.py`](src/pipeline/run_inspector.py) нейтральный query-layer:
  - загрузка `pipeline_state.json`
  - чтение log tail
  - diff между попытками
  - scan runs summary
- Вынести CLI rendering в отдельный модуль, например:
  - `src/cli/run_rendering.py`
  - или `src/pipeline/run_cli_rendering.py`
- Убрать из query-layer presentation helpers:
  - иконки статусов
  - цвета
  - форматирование для Rich/TUI

## Этап 2. Дочистить `main.py`
- Перевести команды CLI с прямой работы через старый `run_inspector` на новую границу:
  - `inspect-run`
  - `runs-list`
  - `run-diff`
  - `run-status`
- Сделать так, чтобы `main.py` использовал:
  - query/use-case API
  - отдельный CLI renderer
- Не держать в `main.py` UI-oriented helper’ы из pipeline.

## Этап 3. Сделать нормальную launch/restart границу
- Выделить pipeline-side use-case/query модуль для launch/restart решений.
- Этот слой должен отвечать на вопросы:
  - какой default mode выбрать
  - можно ли `resume`
  - какие restart points доступны
  - почему что-то заблокировано
  - какой config path должен использоваться
- TUI должен вызывать только этот use-case, а не собирать решение через helper-функции и `load_config`.

## Этап 4. Упростить TUI launch adapters
- После появления pipeline-side use-case упростить [`src/tui/adapters/launch_backend.py`](src/tui/adapters/launch_backend.py):
  - убрать из него знание о `compute_config_hashes`
  - убрать из него прямую работу с `list_restart_points`
  - убрать из него прямой `load_config`, если получится закрыть это use-case слоем
- Оставить там только TUI-facing адаптацию результата для модалки.

## Этап 5. Финальная зачистка
- Проверить, что в `src/tui/screens/*` нет прямых импортов из `src.pipeline.*`.
- Проверить, что в `pipeline` не осталось кода, существующего только ради TUI presentation/read-model.
- Убедиться, что:
  - TUI presentation живет в `src/tui/*`
  - CLI presentation живет в CLI-specific слое
  - pipeline остается источником state и use-case результатов

## Критерий готовности
- `src/pipeline/run_inspector.py` больше не содержит UI/presentation helper логики.
- `main.py` использует query/use-case + CLI renderer, а не смешанную boundary-функцию.
- `src/tui/adapters/launch_backend.py` не содержит pipeline business logic, а только адаптацию результата.
- TUI и CLI не зависят от pipeline presentation helpers.
- В `pipeline` остаются только domain/application/state/use-case контракты.

## Порядок выполнения
1. Разделить `run_inspector` на query-layer и CLI rendering.
2. Перевести CLI-команды на новую границу.
3. Выделить pipeline-side launch/restart use-case.
4. Упростить TUI launch adapter.
5. Добить остаточные зависимости и прогнать тесты.
