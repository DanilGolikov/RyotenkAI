# Custom metrics dashboard on top of MLflow API

> **Status:** parked / future work. Captured here so it's not forgotten.
> Sibling plan: `robust-rolling-dolphin.md` (system-metrics callback cleanup).

## Context

При обсуждении системы метрик всплыла **главная боль** пользователя: **MLflow UI плохо масштабируется на длинных run-ах** (12-48h+). Конкретно:

- График с десятками тысяч точек тормозит / зависает.
- Нет downsampling на уровне UI.
- Pan/zoom медленный.
- Compare runs side-by-side плохо работает на больших объёмах.

Это **архитектурное** ограничение MLflow — он рос из «эксперимент = таблица params/metrics», не из «time-series viewer».

Мы рассмотрели альтернативы и **отвергли** их:

- **VictoriaMetrics + Grafana** для системных метрик — решает только ~20% боли (system dashboards), не помогает с loss-curves в MLflow. Создаёт два UI и дубли.
- **Заменить MLflow на WandB / Aim** — переписать `MLflowManager`, `ResilientMLflowTransport`, `MetricsBuffer`, всю обвязку. Месяцы работы, потеря model registry/artifacts.
- **TensorBoard parallel** — два UI, дубли storage, не решает UX.

**Выбранное направление:** оставить MLflow как single source of truth, построить **свой backend middleware** с server-side downsampling + **кастомный React dashboard** во фронте, заточенный под наш ML-UX.

## Цель

Один источник правды (MLflow), быстрый собственный UI на длинных run-ах, control over UX (compare runs с конфигами под графиками, live training feed через WebSocket, click-through к артефактам).

---

## Approach

### Архитектура

```
┌─────────────────────────────────────────────────────┐
│  HF Trainer                                         │
│    └─→ mlflow.log_metrics                           │
│           └─→ ResilientMLflowTransport              │
│                  └─→ MLflow server (как сейчас)     │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  Our FastAPI backend (новое)                        │
│    /api/runs/:id/metrics?downsample=N&from=...&to=  │
│      ─→ MLflow REST client (читает raw)             │
│      ─→ LTTB downsampling в Python                  │
│      ─→ Redis/in-memory cache                       │
│      ─→ JSON для фронта                             │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  React frontend (расширение существующего)          │
│    /experiments/:id/metrics — кастомные графики     │
│      Recharts/Visx с pan/zoom                       │
│      Compare runs UI                                │
│      Live feed via WebSocket для активных run-ов   │
└─────────────────────────────────────────────────────┘
```

### 1. Backend middleware (FastAPI route)

**Файл:** `src/api/routes/metrics.py` (новый, рядом с существующими роутами в `src/api/`)

**Эндпоинт:** `GET /api/runs/{run_id}/metrics`

Параметры:
- `keys`: список ключей метрик (`?keys=train/loss,eval/loss`)
- `downsample`: целевое число точек на ключ (default 1000)
- `from`, `to`: optional time range filter
- `agg`: `lttb` (default) | `bucket_avg` | `none`

Логика:
1. Через `MlflowClient.get_metric_history(run_id, key)` тянет raw точки.
2. Если `len(points) > downsample` — применяет **LTTB** (Largest Triangle Three Buckets — стандарт downsampling time-series, сохраняет визуальную форму).
3. Кэширует результат:
   - На активный run (последний log < 60 сек назад): TTL 30 сек.
   - На завершённый run: TTL «forever» (key includes run_id, метрики иммутабельны после `end_run`).
4. Возвращает JSON: `{key: [{step, value, timestamp}, ...]}`.

**Зависимости:** `lttb` (Python lib, ~30 строк pure-Python если делать руками; или `pip install lttb`).

**Cache backend:** для MVP — `functools.lru_cache` per-process. Потом — Redis (если будет multi-worker FastAPI).

### 2. Compare-runs endpoint

**Эндпоинт:** `GET /api/runs/compare?run_ids=A,B,C&keys=train/loss`

Возвращает downsampled метрики для нескольких run-ов в одном response, с метаданными (run name, params snapshot) для отображения под графиком.

### 3. Live feed WebSocket

**Эндпоинт:** `WS /api/runs/{run_id}/live`

Уже есть `RunnerEventCallback` (`src/training/callbacks/runner_event_callback.py`) который шлёт события на pod-runner → WebSocket. Расширить:
- В `on_log` HF Trainer hook — пушить метрики прямо во WebSocket (минуя MLflow round-trip для UI).
- На фронте: график получает «реалтайм» точку каждый шаг, дополняя downsampled history из middleware.
- Это даёт **под-секундный latency** на активном trainе (vs. polling MLflow API раз в 5 сек).

### 4. Frontend dashboards

**Файлы:** `web/src/pages/experiments/MetricsView.tsx` (новый), `web/src/components/charts/` (новые компоненты).

Компоненты:
- `<MetricChart>` — single-key time-series, обёртка над Recharts/Visx.
- `<CompareRuns>` — сетка графиков по нескольким run-ам с conf-snippets под каждым.
- `<LiveTrainingFeed>` — подключается к WebSocket, апдейтит график инкрементально.
- `<MetricsExplorer>` — навигатор по ключам метрик (`train/*`, `eval/*`, `gpu/*`, `cpu/*`), пресеты дашбордов.

Поведение:
- Pan/zoom — встроено в Recharts (брать timestamp range, дёргать middleware с новым `from`/`to`, динамический re-fetch).
- Click on point → drawer с raw value, step, run params snapshot.
- Click on run name → переход в MLflow UI на тот run (для артефактов / model registry / params).

### 5. Routing на фронте

В `web/src/App.tsx`:
- `/experiments/:id/metrics` — новый дашборд (default).
- `/experiments/:id/metrics/compare?runs=...` — compare view.
- Кнопка «Open in MLflow UI» — fallback для случаев, когда наш UI чего-то не покрывает.

---

## Critical files to touch

### Новое
- `src/api/routes/metrics.py` — middleware route + LTTB
- `src/api/services/mlflow_metrics_cache.py` — кэш-слой
- `web/src/pages/experiments/MetricsView.tsx`
- `web/src/components/charts/MetricChart.tsx`
- `web/src/components/charts/CompareRuns.tsx`
- `web/src/components/charts/LiveTrainingFeed.tsx`
- `web/src/api/metrics.ts` — typed клиент

### Расширения существующего
- `src/api/main.py` — зарегистрировать новый router
- `src/training/callbacks/runner_event_callback.py` — добавить push метрик в WebSocket на `on_log`
- `web/src/App.tsx` — новые routes

## Reusable patterns

- **MLflow REST client** — уже используется в `src/training/managers/mlflow_manager/` (через `MlflowClient`).
- **WebSocket infrastructure** — `RunnerEventCallback` + pod-runner pipe (см. `src/training/callbacks/runner_event_callback.py`).
- **FastAPI router pattern** — `src/api/routes/run.py`, `src/api/schemas/run.py`.
- **React data-fetching** — посмотреть `web/src/api/openapi.json` (auto-gen) и `web/src/components/ConfigBuilder/` для существующего паттерна.
- **LTTB algorithm** — pure-Python либа `lttb` или 30 строк reference impl.

---

## Verification

1. Backend:
   - Unit-тесты на LTTB (в т.ч. edge cases: < `downsample` точек → возврат as-is; пустой ряд; одна точка).
   - Integration test: создать run с 100k точек метрики через MLflow API, дёрнуть `/api/runs/.../metrics?downsample=1000`, проверить что вернулось ≤1000 точек и форма curve визуально похожа (можно через image-similarity на рендере).
   - Кэш-тесты: повторный запрос обслуживается из кэша без обращения к MLflow.
2. Frontend:
   - Storybook entries на каждый chart-компонент.
   - Manual smoke: открыть длинный run в нашем UI, замерить time-to-interactive vs MLflow UI на том же run-е.
3. End-to-end:
   - Запустить тренировку 12+ часов (или симулировать через скрипт массового логирования), убедиться что наш UI не лагает на pan/zoom.
   - WebSocket live feed: запустить короткий trainе, открыть `/metrics` → видеть метрики в реальном времени без F5.
4. Performance budgets:
   - First metric render < 500 ms (with cache miss).
   - Pan/zoom interaction < 100 ms (cache hit).
   - Compare 5 runs × 10 metrics < 2 seconds total.

---

## Risks & open questions

- **Cardinality**: если ключей метрик много (100+ per run), `keys=*` запрос может разрослсясь. Лимит: max 20 ключей за запрос, фронт делает paginated requests при необходимости.
- **MLflow REST API rate limits**: на собственном tracking-server лимитов нет, но при scale-up на managed MLflow придётся учесть.
- **Кэш-инвалидация для активных run-ов**: TTL 30 сек = 30-сек задержка свежих точек в графике для не-WebSocket клиентов. WebSocket для активных компенсирует.
- **LTTB для не-numeric values?** Метрики MLflow только float, не проблема.
- **Open question**: использовать ли SSE вместо WebSocket для live feed? SSE проще на бэке (no protocol upgrade), но односторонний (нам не нужна двусторонняя коммуникация для отображения). SSE — кандидат.

## Sequencing

Можно делать **после** robust-rolling-dolphin (system metrics callback cleanup) и **параллельно** с любыми другими фичами — изолированный новый код, не пересекается со steady-state pipeline.

Оценка: ~2 недели на MVP (backend + базовые charts + один дашборд), ~1 месяц на production-grade с compare view, WebSocket live feed, storybook, e2e тестами.
