# Monorepo Packagization: разделение `src/` на 5 uv-workspace пакетов

**Date:** 2026-05-03
**Author:** Claude (под руководством daniil)
**Status:** DRAFT — ждёт финального GO от пользователя
**Branch:** `claude/dazzling-rosalind-b482fa`
**Worktree:** `.claude/worktrees/dazzling-rosalind-b482fa`

> ⚠️ **Это план, а не диф.** Здесь зафиксированы цели, фазы, риски и
> закреплённые решения. Никаких изменений до явного «GO».

> **Финальные решения (по итогам §13):** Flat naming (`ryotenkai_pod`,
> `ryotenkai_shared`, `ryotenkai_community`, `ryotenkai_providers`,
> `ryotenkai_control`). Один pod-пакет (runner + trainer вместе).
> Community — отдельный видимый workspace member (не растворяется в
> shared). **src-layout** для каждого пакета
> (`packages/<pkg>/src/ryotenkai_<pkg>/`). Phase A blocking перед
> Phase B. Phase B = 1 PR из 5 коммитов.

> **Для следующего агента:** §18 «Plan Audit» (закрытые дыры) +
> §19 «Handover» в самом конце документа — там собраны все ключевые
> команды, ссылки на файлы, gotchas и emergency procedures.

---

## 0. TL;DR

Разделить монолитный `src/` (≈93k LOC прод-кода + 117k LOC тестов, 16 SCC,
hotspot-плотность 100% в верхушке) на **5 физически независимых uv-workspace
пакетов**, чтобы:

1. Зафиксировать архитектурную границу control plane (Mac) ↔ runtime (pod)
   на уровне `pip install` (impossible-to-violate).
2. Уничтожить класс багов "16-crash chain" вида `src/providers →
   src/pipeline` (см. PR-history `5293bbf`, `9d29e6c`): в пакете
   `ryotenkai-providers` модуль `ryotenkai_pipeline` физически отсутствует.
3. Сделать pod-образ **легче на ~80%** (в pod больше нет paramiko,
   runpod-sdk, fastapi-control-plane, web-deps).
4. Подготовить почву для предстоящего "Transport Unification" плана (см.
   §11) и снять зависимость от importlinter как единственного барьера.

**Цена:** один большой PR с переносом ~1500 файлов и rewrite каждой строки
`from src.X` → `from ryotenkai_<pkg>.X`. Pre-cleanup (Phase A, ~6 PRs)
обязателен — иначе циклы заблокируют `uv sync`.

---

## 1. Контекст и мотивация

### 1.1 Почему сейчас

Из последнего сообщения пользователя:

> «давай займемся прям сейчас, будет больно, но чем дольше тянем, тем
> более будет, т.к. неправильная архитектура растет и закрепляется»

Эмпирические подтверждения архитектурного гниения (фактура, не мнение):

| Сигнал | Метрика | Источник |
|---|---|---|
| Циклов в графе модулей | **16 SCC** | repowise overview |
| Hotspot-плотность top-5 файлов | 99-100 percentile | repowise risk |
| Bus factor по 1380 файлам | **1** (sole owner: daniil) | repowise overview |
| Churn `pipeline/orchestrator.py` за 90д | **+3584/-3020** | repowise risk |
| Bug-prone `training_monitor.py` | **trend: increasing** | repowise risk |
| Контрольный 16-crash incident | run_20260502_113553_r8rul | git log `9d29e6c` |
| Last big refactor (Phase 0 этой ветки) | 3 PR за день, 164 теста зелёные | git log |

Архитектурная граница «Mac orchestrator vs pod runtime» уже существует
**идейно** (см. CLAUDE.md, ADR-серии), но не enforced. Она поддерживается
конвенциями (`README.md`, code review, импортлинтером в плане
`2026-05-03-transport-unification-http-only-runtime.md` §3.2). Каждый раз,
когда новый transitive импорт пересекает границу, она дрейфует на
несколько метров. Pre-2026-05-02 был whitelist в CodeSyncer — он
дрейфовал 16 раз подряд за один inference-run.

### 1.2 Почему именно uv-workspaces

Альтернативы и почему отвергнуты:

| Вариант | Вердикт | Почему |
|---|---|---|
| **Status quo + importlinter** | Маловато | Code-only барьер, не блокирует `pip install`; запланирован отдельно как «слабая» страховка. |
| **Pants/Bazel** | Overkill | Build orchestrator industrial-grade, требует переписывания CI и Dockerfile. Не оправдано для 93k LOC. |
| **poetry workspaces** | Outdated | poetry ≠ uv по скорости (10-100x); индустрия 2025-2026 переходит на uv (Apache Airflow, Pydantic, FastAPI core). |
| **Hatch projects** | Не workspace-first | Hatch — проектный менеджер, не монорепа-tool. |
| **pip-only multiple `setup.cfg`** | Legacy | Нарушает PEP 621/735, нет lock-файла, ручное управление совместимостью версий. |
| **uv workspaces** | ✅ ВЫБРАНО | Astral, single `uv.lock`, single `.venv`, native в нашем pyproject.toml уже (см. §2.5). Apache Airflow в проде с 122 distributions ([FOSDEM 2026](https://fosdem.org/2026/schedule/event/WE7NHM-modern-python-monorepo-apache-airflow/)). |

### 1.3 Что НЕ цель

- ❌ **Public PyPI publishing** — пакеты остаются internal (private repo).
- ❌ **Independent versioning** — все пакеты держат единую версию `1.0.0`
  (синхронный bump). Расхождение версий — вопрос будущего.
- ❌ **Frontend (`web/`) refactor** — фронт деприоритизирован
  (явно по сообщению пользователя). Только update импортов в Makefile-цели
  `make openapi`.
- ❌ **Замена FastAPI / SQLAlchemy / Pydantic** и прочие библиотечные
  замены — out of scope.

---

## 2. Текущее состояние (фактура, измеримое)

### 2.1 Инвентарь `src/` (15 прод-папок)

LOC взяты `wc -l` от 2026-05-03 (включая комментарии/пробелы):

| Папка | LOC | Сторона | Роль | Куда уйдёт |
|---|---:|---|---|---|
| `pipeline` | 18,990 | Mac | Mac orchestrator (stages, deployment, state, launch, mlflow_attempt) | `ryotenkai-control` |
| `training` | 15,593 | Pod | Trainer subprocess: orchestrator, callbacks, strategies, managers, mlflow, models, trainers, reward_plugins, templates | `ryotenkai-pod` (`ryotenkai_pod.trainer`) |
| `providers` | 10,604 | Mac | Provider clients: runpod (lifecycle, training, runtime, inference) + single_node | `ryotenkai-providers` |
| `api` | 8,027 | Mac | FastAPI control-plane server (routers, services, ws, presentation) | `ryotenkai-control` |
| `runner` | 6,615 | Pod | uvicorn FastAPI in-pod server (api/control, api/events, api/internal, runtime/fsm, runtime/event_bus) | `ryotenkai-pod` (`ryotenkai_pod.runner`) |
| `community` | 6,289 | Both | Plugin loader/catalog/manifest/sync framework | `ryotenkai-community` (own pkg) |
| `utils` | 5,462 | Both | logger, result, file_utils, container.py (god object — см. §3) | `ryotenkai-shared` |
| `config` | 5,314 | Both | Pydantic config schemas (PipelineConfig, ModelConfig, …) | `ryotenkai-shared` |
| `cli` | 5,144 | Mac | Typer CLI (commands/run, server, config, integrations, …) | `ryotenkai-control` |
| `reports` | 4,845 | Mac | Markdown report generation (renderers, document, plugins, models) | `ryotenkai-control` |
| `data` | 1,986 | Mac | Dataset loaders + preview (hf_loader, json_loader, multi_source_loader, factory) | `ryotenkai-control` |
| `workspace` | 1,788 | Mac | UX-layer: projects/providers registry (CLI persisted state) | `ryotenkai-control` |
| `evaluation` | 1,398 | Mac | Eval orchestration (calls plugins) | `ryotenkai-control` |
| `infrastructure` | 462 | Mac | MLflow gateway HTTP client + URI resolver | `ryotenkai-control` |
| `cli_state` | 194 | Mac | CLI cache (context_store) | `ryotenkai-control` |
| `inference` | 124 | Both | Engine constants (`__about__`) | `ryotenkai-shared` |
| `tests` | 116,867 | — | Per-package: distributed по новым пакетам (см. §6.4) | per-package |

**Итого прод:** ~93,000 LOC. **Тесты:** ~117,000 LOC.

### 2.2 Граф зависимостей (целевой, на 2026-05-03)

```
                    ┌─────────────────────┐
                    │  ryotenkai-shared   │  ← листовой
                    │ utils + config +    │
                    │ constants +         │
                    │ inference +         │
                    │ infrastructure      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ ryotenkai-community │  ← plugin framework
                    │ (loader/catalog/    │
                    │  manifest/sync)     │
                    └─────┬───────┬───────┘
                          │       │
            ┌─────────────┘       │
            │                     │
     ┌──────▼────┐         ┌──────▼─────┐         ┌─────────────┐
     │ryotenkai- │         │ ryotenkai- │         │ ryotenkai-  │
     │   pod     │         │  control   │◄────────┤ providers   │
     │ (runner + │         │ (pipeline+ │         │ (RunPod +   │
     │  trainer) │         │  api+cli+…)│         │ single_node)│
     └───────────┘         └────────────┘         └──────┬──────┘
        (pod)                  (Mac)                     │
                                                         │
                                              shared ◄───┘
                                              (providers → shared
                                               only, NO community)
```

Стрелки (после Phase A):
- shared: листовой, никуда не зависит.
- community → shared.
- pod → shared, community (trainer использует reward_plugins).
- providers → shared (НЕ зависит от community).
- control → shared, community, providers.

Запрещено навсегда (importlinter): control → pod, pod → control, pod →
providers, providers → control, providers → pod.

### 2.3 16 циклов: симптомы

Из repowise `search_codebase("circular dependency")`:

- `scc-160` — **god object** `src/utils/container.py` (718 LOC), DI-контейнер
  для тренировки. Импортируется только из `src/training/*`. **Решение:**
  переехать `src/utils/container.py` → `src/training/container.py` целиком.
- `scc-208`, `scc-219` — *utility ↔ training orchestration* coupling вокруг
  того же container.py. Уйдут вместе с переездом контейнера.
- `scc-273` — `pipeline/stages/dataset_validator/plugin_runner.py` ↔
  `dataset_validator/__init__.py`. Локальный, разруливается reorder
  импортов внутри пакета.
- `scc-267` — `runpod/inference/__init__.py` ↔ `runpod/inference/provider.py`.
  Локальный.
- `scc-20` — config schemas ↔ validation logic. Локальный, не пересекает
  будущие границы пакетов.

Из 16 SCC — **до границ пакетов доходят 4** (см. §2.4). Остальные —
internal, разрулятся либо при миграции, либо позже.

### 2.4 Cross-package violations (детально, файл:строка → символ)

#### A. `api ↔ pipeline` — двунаправленный цикл (КРУПНЫЙ)

**pipeline → api (4 импорта):**
```
src/pipeline/stages/training_monitor.py:34
    from src.api.clients.job_client import (...)
src/pipeline/stages/training_monitor.py:68-69
    from src.api.services.tunnel_service import SSHTunnelManager
src/pipeline/stages/managers/deployment/training_launcher.py:46-48
    from src.api.clients.job_client import JobClient, JobClientError
    from src.api.services.control_plane_heartbeat import ControlPlaneHeartbeat
    from src.api.services.tunnel_service import (...)
```

**api → pipeline (10+ импортов):**
```
src/api/state_cache.py:39        from src.pipeline.state.store import PipelineStateLoadError, PipelineStateStore
src/api/exceptions.py:6          from src.pipeline.state import PipelineStateLoadError, PipelineStateLockError
src/api/dependencies.py:16       from src.pipeline.state import PipelineStateStore
src/api/services/log_service.py  from src.pipeline.state import PipelineStateStore
src/api/services/run_service.py  from src.pipeline.launch import is_process_alive, read_lock_pid
src/api/services/run_service.py  from src.pipeline.run_queries import (...)
src/api/ws/log_stream.py:12      from src.pipeline.run_queries import effective_pipeline_status
src/api/ws/log_stream.py:13      from src.pipeline.state import PipelineStateLoadError, PipelineStateStore
src/api/presentation/formatters.py from src.pipeline.state.models import StageRunState
src/api/presentation/icons.py    from src.pipeline.state.models import StageRunState
```

**Анализ направления связи:**
- `api → pipeline.state` — api **читает** state файлы пайплайна. Это
  legitimate: api — это "тонкий слой view" поверх state-storage.
  **Решение:** state не уезжает. После split api и pipeline в одном
  пакете `ryotenkai-control` — цикл становится внутрипакетным и
  допустимым (если вообще остаётся; см. ниже).
- `pipeline → api.clients/services` — `JobClient`, `ControlPlaneHeartbeat`,
  `SSHTunnelManager` — это **HTTP-клиенты к api**, размещённые в `src/api`
  по ошибке. `JobClient` стучится в `runner` (а не в `api`), а
  `ControlPlaneHeartbeat` стучится в `api` извне. Они должны жить либо в
  shared (HTTP utility layer), либо в `pipeline` (потребитель). **Решение
  Phase A:** перенести их из `src/api/clients`/`src/api/services` в
  целевые подпакеты.

#### B. `providers → pipeline` — однонаправленные нарушения (3)

```
src/providers/runpod/lifecycle/pod_ssh_waiter.py:45
    from src.pipeline.cancellation import sleep_cancellable
src/providers/runpod/training/provider.py:14
    from src.pipeline.cancellation import PipelineCancelled
src/providers/single_node/inference/provider.py:23
    from src.pipeline.inference.vllm import VLLMEngine
```

Это корень 16-crash chain (run_20260502_113553_r8rul). **Решение Phase A:**
- `pipeline.cancellation` (`PipelineCancelled`, `sleep_cancellable`) —
  переехать в `src/utils/cancellation.py` (general primitives). Pipeline
  переэкспортирует для backward compat внутри ветки до Phase B.
- `pipeline.inference.vllm.VLLMEngine` — переехать в
  `src/providers/inference/vllm/engine.py`. vLLM — это provider-side
  inference engine; неправильно лежит в pipeline.

#### C. `pipeline → runner` — одна константа

```
src/pipeline/stages/managers/deployment/dependency_installer.py:25
    from src.runner.__about__ import RUNTIME_IMAGE
```

**Решение Phase A:** перенести `RUNTIME_IMAGE` в shared
`src/constants/runtime.py`. Тривиально.

#### D. `pipeline → training` — один менеджер

```
src/pipeline/mlflow_attempt/manager.py:29
    from src.training.managers.mlflow_manager import MLflowManager
```

Pipeline (Mac) импортирует тип из training (pod). Это самое неприятное:
оба пакета идут на разные машины и разные python-окружения. **Решение
Phase A:** извлечь `IMLflowManager` Protocol в shared
(`src/utils/mlflow_protocol.py` или `src/infrastructure/mlflow/protocol.py`),
и pipeline импортит **только** Protocol. Training продолжает
имплементировать его в своём `MLflowManager`.

#### E. `utils → config` — facade

```
src/utils/config.py:8
    from src.config import (... 30+ symbols ...)
```

Это устаревший backward-compat facade (комментарий в файле:
"Source of truth lives in `src/config/`"). Используется в 10+ местах.

**Решение Phase A:** удалить facade, переписать импорты через codemod
(`from src.utils.config import X` → `from src.config import X`).

### 2.5 Текущий `pyproject.toml` (что есть)

- Single root `pyproject.toml` (10.8 KB).
- Один distribution: `name = "ryotenkai"`, version `1.0.0`.
- `uv.lock` уже **существует и используется** (Astral uv детектится).
- `setuptools.packages.find` ищет `src*` — ёжина из эпохи 2024.
- Один `console_script`: `ryotenkai = "src.main:app"`.
- Single dependency list — 35 prod пакетов, 13 dev пакетов.
- Pre-commit, ruff, mypy, pytest, coverage, bandit — все настроены глобально.

**Что важно:** мы УЖЕ на uv. Workspace — это конфигурационное расширение,
а не миграция инструмента.

---

## 3. Целевое состояние

### 3.1 Layout (high-level)

```
RyotenkAI/
├── pyproject.toml            ← root: workspace declaration only
├── uv.lock                   ← single lock для всех пакетов
├── Makefile, README.md, CONTRIBUTING.md, LICENSE
├── .pre-commit-config.yaml, .gitignore, .dockerignore, .env.example
├── pyrightconfig.json, pytest.ini, setup.sh, run.sh
│
├── packages/                 ← ВСЕ workspace members живут здесь
│   ├── shared/               ← ryotenkai-shared (листовой, обе стороны)
│   ├── community/            ← ryotenkai-community (plugin framework)
│   ├── pod/                  ← ryotenkai-pod (runner + trainer)
│   ├── providers/            ← ryotenkai-providers (Mac)
│   └── control/              ← ryotenkai-control (Mac)
│
├── community/                ← плагины-данные (НЕ workspace member)
│   ├── evaluation/  reward/  validation/  presets/  libs/  reports/
│   └── (третьесторонние plugin.py + manifest.toml)
│
├── docker/                   ← Dockerfiles (training/inference/mlflow)
├── docs/                     ← планы, ADR
├── examples/                 ← примеры конфигов
└── web/                      ← frontend (деприоритизирован)
```

**Итого: 5 пакетов**, каждый — independent installable distribution.

> ⚠️ **Терминологический disclaimer:** в репо две сущности с похожими
> именами:
> - `packages/community/` (NEW, workspace member) — Python-код
>   plugin loader/catalog/manifest framework. Старое имя: `src/community/`.
> - `community/` в корне репо — конфиги и Python-файлы конкретных
>   плагинов (helixql/, eval-плагины, reward-плагины). НЕ workspace member,
>   просто данные. После Phase B их `.py` файлы импортят
>   `ryotenkai_community.X` и `ryotenkai_*.<other>` вместо `src.X`.

**Почему runner и trainer в одном пакете `ryotenkai-pod`** (решено
пользователем): они всегда деплоятся вместе (один Docker image), runner
spawns trainer как subprocess, dep surface всё равно общий (torch +
fastapi нужны оба в pod-image). Разделять академически чище, но
практически даёт ноль выгоды. Внутри `ryotenkai-pod` сохраняется чёткое
разделение subpackages: `runner/` (HTTP-сервер) и `trainer/` (subprocess).

**Почему community вынесена в отдельный пакет** (решено пользователем):
плагин-фреймворк концептуально отделён от обычных utils. Чтобы новый
разработчик с первого взгляда видел эту границу — она получает свой
`packages/community/` рядом с pod, providers, control. С точки зрения
зависимостей: community → только shared; никто (кроме pod и control)
не зависит от community → внутри pod лишь trainer-сабпакет (для reward
plugins), runner-сабпакет community не использует.

### 3.2 Layout (file-level, полное дерево после Phase B)

Сокращённо (только ключевые файлы и каталоги; полный список — в
актуальном `tree -L 4 packages/` после миграции):

```
packages/
├── shared/
│   ├── pyproject.toml                    ← name="ryotenkai-shared"
│   ├── README.md
│   ├── src/ryotenkai_shared/
│   │   ├── __init__.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── logger.py                 (← src/utils/logger.py)
│   │   │   ├── result.py                 (← src/utils/result.py)
│   │   │   ├── memory_manager.py         (← src/utils/memory_manager.py)
│   │   │   ├── text_utils.py             (← src/utils/text_utils.py)
│   │   │   ├── cancellation.py           (← перенесён из src/pipeline/cancellation.py в Phase A.1)
│   │   │   └── clients/
│   │   │       ├── job_client.py         (← из src/api/clients/ в Phase A.2)
│   │   │       └── ssh_tunnel.py         (← из src/api/services/ в Phase A.2)
│   │   ├── config/                       (← src/config/* целиком)
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py
│   │   │   ├── training/
│   │   │   ├── providers/
│   │   │   └── …
│   │   ├── constants/
│   │   │   ├── __init__.py
│   │   │   ├── runtime.py                (← RUNTIME_IMAGE из src/runner/__about__.py в Phase A.4)
│   │   │   └── …
│   │   ├── inference/                    (← src/inference/, constants only)
│   │   └── infrastructure/               (← src/infrastructure/)
│   │       └── mlflow/
│   │           ├── gateway.py
│   │           ├── uri_resolver.py
│   │           ├── environment.py
│   │           └── protocol.py           (← NEW: IMLflowManager Protocol из Phase A.5)
│   └── tests/unit/...
│
├── community/
│   ├── pyproject.toml                    ← name="ryotenkai-community", deps=["ryotenkai-shared"]
│   ├── README.md
│   ├── src/ryotenkai_community/
│   │   ├── __init__.py
│   │   ├── catalog.py                    (← src/community/catalog.py)
│   │   ├── pack.py
│   │   ├── stale_plugins.py
│   │   ├── sync.py
│   │   ├── registry_base.py
│   │   ├── manifest.py
│   │   ├── toml_writer.py
│   │   ├── constants.py
│   │   ├── instance_validator.py
│   │   ├── scaffold_template.py
│   │   ├── inference.py
│   │   └── libs.py                       (helper для preload_community_libs)
│   └── tests/unit/...
│
├── pod/
│   ├── pyproject.toml                    ← name="ryotenkai-pod", deps=["ryotenkai-shared","ryotenkai-community"]
│   ├── README.md
│   ├── src/ryotenkai_pod/
│   │   ├── __init__.py
│   │   ├── runner/                       (← src/runner/*)
│   │   │   ├── __init__.py
│   │   │   ├── __about__.py              (без RUNTIME_IMAGE — он переехал в shared)
│   │   │   ├── main.py                   ← uvicorn entry: ryotenkai_pod.runner.main:app
│   │   │   ├── api/
│   │   │   │   ├── control.py
│   │   │   │   ├── events.py
│   │   │   │   └── internal.py
│   │   │   └── runtime/
│   │   │       ├── fsm.py
│   │   │       ├── event_bus.py
│   │   │       └── log_archiver.py
│   │   └── trainer/                      (← src/training/*)
│   │       ├── __init__.py
│   │       ├── run_training.py           ← console_script: ryotenkai-trainer-run
│   │       ├── container.py              ← из src/utils/container.py (Phase A.7)
│   │       ├── orchestrator/
│   │       │   ├── strategy_orchestrator.py
│   │       │   ├── chain_runner.py
│   │       │   └── phase_executor/
│   │       ├── strategies/
│   │       ├── managers/
│   │       │   ├── mlflow_manager/       ← реализует ryotenkai_shared.infrastructure.mlflow.protocol.IMLflowManager
│   │       │   └── data_buffer/
│   │       ├── callbacks/
│   │       ├── mlflow/
│   │       ├── models/
│   │       ├── trainers/
│   │       ├── reward_plugins/           (использует ryotenkai_community)
│   │       ├── templates/
│   │       └── constants.py
│   └── tests/unit/{runner,trainer}/...
│
├── providers/
│   ├── pyproject.toml                    ← name="ryotenkai-providers", deps=["ryotenkai-shared"]
│   ├── README.md
│   ├── src/ryotenkai_providers/
│   │   ├── __init__.py
│   │   ├── runpod/
│   │   │   ├── lifecycle/
│   │   │   ├── training/
│   │   │   ├── runtime/
│   │   │   └── inference/
│   │   ├── single_node/
│   │   │   ├── training/
│   │   │   ├── runtime/
│   │   │   └── inference/
│   │   └── inference/
│   │       └── vllm/
│   │           └── engine.py             (← из src/pipeline/inference/vllm.py в Phase A.3)
│   └── tests/unit/...
│
└── control/
    ├── pyproject.toml                    ← name="ryotenkai-control",
    │                                       deps=["ryotenkai-shared","ryotenkai-community","ryotenkai-providers"]
    ├── README.md
    ├── src/ryotenkai_control/
    │   ├── __init__.py
    │   ├── main.py                       ← console_script: ryotenkai = "ryotenkai_control.cli.app:app"
    │   ├── pipeline/                     (← src/pipeline/*)
    │   │   ├── orchestrator.py
    │   │   ├── stages/
    │   │   ├── state/
    │   │   ├── launch/
    │   │   ├── bootstrap/
    │   │   ├── reporting/
    │   │   ├── artifacts/
    │   │   ├── context/
    │   │   ├── execution/
    │   │   ├── mlflow_attempt/
    │   │   ├── config_drift/
    │   │   ├── run_queries.py
    │   │   ├── deletion.py
    │   │   ├── constants.py
    │   │   └── heartbeat/                (← из src/api/services/control_plane_heartbeat.py в Phase A.2)
    │   ├── api/                          (← src/api/*, без clients/services которые переехали)
    │   │   ├── main.py
    │   │   ├── routers/
    │   │   ├── ws/
    │   │   ├── presentation/
    │   │   └── openapi_dump.py           (target для `make openapi`)
    │   ├── cli/                          (← src/cli/*)
    │   │   ├── app.py
    │   │   └── commands/
    │   ├── data/                         (← src/data/*)
    │   ├── evaluation/                   (← src/evaluation/*)
    │   ├── reports/                      (← src/reports/*)
    │   ├── workspace/                    (← src/workspace/*)
    │   └── cli_state/                    (← src/cli_state/*)
    └── tests/{unit,integration,contract}/...
```

**Размер диффа Phase B:** ~1500 файлов перемещены, ~30k LOC import
statements переписаны, ~200 файлов вне `src/` патчатся (Makefile,
docker, web/openapi-generator, root community/ plugin .py, docs).

### 3.3 Naming convention

| Element | Convention | Пример |
|---|---|---|
| Distribution name (PyPI) | kebab-case с префиксом | `ryotenkai-pod` |
| Import name (Python module) | snake_case с префиксом | `ryotenkai_pod` |
| Workspace folder | kebab-case без префикса | `packages/runner/` |
| Console scripts | kebab-case с префиксом | `ryotenkai-trainer-run` |

Префикс `ryotenkai-` — обязателен. Защищает от:
- Конфликтов с stdlib / third-party (`runner` коллизия с pytest-runner;
  `pipeline` коллизия с многими ML-libs; `control` и `shared` — общие).
- Будущей публикации на private PyPI / Artifactory.
- Грепабельности (`grep -r ryotenkai_` находит ВСЁ наше).

Альтернатива (см. §13 Open Questions): namespace package
(`ryotenkai.runner`, `ryotenkai.trainer`, …) как Apache Airflow
(`airflow.providers.amazon`). Дороже в Phase B (нужны pkgutil-style
`__init__.py`), но красивее долгосрочно.

### 3.4 Зависимости каждого пакета

**`ryotenkai-shared`** (листовой):
```toml
dependencies = [
  "pydantic>=2.10.0,<3.0.0",
  "pydantic-settings>=2.6.0,<3.0.0",
  "python-dotenv>=1.0.0,<2.0.0",
  "pyyaml>=6.0.0,<7.0.0",
  "loguru>=0.7.0,<1.0.0",
  "rich>=14.2.0,<15.0.0",
  "colorlog>=6.9.0",
  "requests>=2.32.0,<3.0.0",
  "mlflow>=3.11.1,<4.0.0",  # для infrastructure.mlflow.gateway
]
```

**`ryotenkai-community`** (plugin framework):
```toml
dependencies = [
  "ryotenkai-shared",  # workspace
  # минимум: только tomllib (stdlib в py3.12+) и pyyaml/pydantic из shared
]
```

**`ryotenkai-pod`** (pod, runner + trainer):
```toml
dependencies = [
  "ryotenkai-shared",     # workspace
  "ryotenkai-community",  # workspace (для trainer.reward_plugins)
  # — Runner side (HTTP server) —
  "fastapi>=0.115.0,<1.0.0",
  "uvicorn[standard]>=0.32.0,<1.0.0",
  "websockets>=13.0.0,<14.0.0",
  "httpx>=0.27.0",
  # — Trainer side (ML stack) —
  "torch>=2.5.0",
  "transformers>=4.46.0,<5.0.0",
  "peft>=0.13.0,<1.0.0",
  "accelerate>=1.0.0",
  "bitsandbytes>=0.42.0; sys_platform != 'darwin'",
  "sentencepiece>=0.2.0",
  "datasets>=4.4.1",
  "trl>=0.12.0",
  "huggingface-hub>=0.26.0,<1.0.0",
  "pynvml>=11.5.0",
  "tensorboard>=2.18.0",
  "scipy>=1.13.0",
  "scikit-learn>=1.5.0",
  "pandas>=2.2.0",
  "protobuf>=5.28.0",
  "dill>=0.4.0",
]
```

**`ryotenkai-providers`** (Mac):
```toml
dependencies = [
  "ryotenkai-shared",  # workspace
  "runpod>=1.7.0,<2.0.0",
  "paramiko>=3.5.0",
  "cryptography>=41.0.0",
  "httpx>=0.27.0",
]
```

**`ryotenkai-control`** (Mac, аггрегатор):
```toml
dependencies = [
  "ryotenkai-shared",
  "ryotenkai-community",
  "ryotenkai-providers",
  "fastapi>=0.115.0,<1.0.0",
  "uvicorn[standard]>=0.32.0,<1.0.0",
  "websockets>=13.0.0,<14.0.0",
  "typer>=0.12.0",
  "prefect>=3.1.0,<4.0.0",
  "dvc>=3.58.0,<4.0.0",
  "httpx>=0.27.0",
]
```

**Root `pyproject.toml`** (workspace declaration, dev tools):
```toml
[project]
name = "ryotenkai-monorepo"
version = "1.0.0"
requires-python = ">=3.12"

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
ryotenkai-shared    = { workspace = true }
ryotenkai-community = { workspace = true }
ryotenkai-pod       = { workspace = true }
ryotenkai-providers = { workspace = true }
ryotenkai-control   = { workspace = true }

[project.optional-dependencies]
dev = [
  "pytest>=8.3.0,<9.0.0",
  "pytest-cov>=6.0.0,<7.0.0",
  "pytest-mock>=3.14.0,<4.0.0",
  "pytest-asyncio>=0.24.0,<1.0.0",
  "ruff>=0.8.0,<1.0.0",
  "black>=24.10.0,<25.0.0",
  "mypy>=1.13.0,<2.0.0",
  "pyright>=1.1.350",
  "pre-commit>=4.0.0,<5.0.0",
  "bandit[toml]>=1.8.0,<2.0.0",
  "import-linter>=2.0",
]
```

### 3.5 Изменение Docker pod-image (большая выгода)

**До:** `pip install -e .` — тянет всё, включая paramiko (16 MB), runpod
(2 MB), prefect (50+ MB), dvc (40+ MB), typer, httpx, и весь pipeline-код.

**После:** `pip install ./packages/shared ./packages/community
./packages/pod` — тянет только их deps. Pod-image теряет: paramiko,
runpod-sdk, prefect, dvc, typer, fastapi-control-plane code, все
pipeline/, все providers/, всю api/, всю cli/.

**Эффект:** ~80% уменьшение image; тривиальный security audit (provider
секреты больше не в pod); нет accidental import "из соседней комнаты".

### 3.6 CodeSyncer изменения

`src/pipeline/stages/managers/deployment/code_syncer.py` сейчас rsync-ит
`src/` целиком. После split:

```python
REQUIRED_PACKAGES = ["packages/shared", "packages/community", "packages/pod"]
EXCLUDE_PATTERNS = ["__pycache__", "*.pyc", "tests", "*.md"]
```

Pod получит только то, что ему нужно для работы. Mac-only пакеты
(control, providers) физически не покинут Mac.

---

## 4. Phase A — Pre-cleanup (cycles + violations)

**Цель:** разрулить **все cross-package** циклы и нарушения границ ДО
переезда. Phase A — серия атомарных, независимых, ревьюаемых PR.

### A.1 Перенести `pipeline.cancellation` → `utils/cancellation`

**Что:** `PipelineCancelled` и `sleep_cancellable` — generic concurrency
primitives, не pipeline-specific.

**Файлы:**
- Move: `src/pipeline/cancellation.py` → `src/utils/cancellation.py`.
- Re-export shim в `src/pipeline/cancellation.py` (для backward compat в
  ветке): `from src.utils.cancellation import *`.
- Rewrite импортов в `src/providers/{runpod/lifecycle/pod_ssh_waiter,
  runpod/training/provider}` на `from src.utils.cancellation import …`.

**Тесты:** существующие unit-tests на cancellation — переехать в
`src/tests/unit/utils/test_cancellation.py`.

### A.2 Извлечь HTTP-clients из api в правильные места

**Что:** `JobClient`, `JobClientError`, `ControlPlaneHeartbeat`,
`SSHTunnelManager` лежат в `src/api/clients` / `src/api/services`,
но являются HTTP/SSH клиентами к runner / api / pod. Pipeline их
импортирует — отсюда цикл `pipeline → api`.

**Файлы:**
- `src/api/clients/job_client.py` → `src/utils/clients/job_client.py`
  (HTTP client to runner — generic).
- `src/api/services/control_plane_heartbeat.py` →
  `src/pipeline/heartbeat/heartbeat.py` (пайплайн пингует api — это
  pipeline-side client).
- `src/api/services/tunnel_service.py` →
  `src/utils/clients/ssh_tunnel.py` (SSH tunnel — generic infra).
- Rewrite импортов в pipeline (training_monitor, training_launcher).
- Backward compat shims в старых местах (на длительность ветки).

**Тесты:** перенести test_job_client, test_tunnel_service,
test_control_plane_heartbeat в новые home directories.

### A.3 Перенести `pipeline.inference.vllm` → `providers.inference.vllm`

**Что:** `VLLMEngine` — это provider-side inference engine.

**Файлы:**
- Move: `src/pipeline/inference/vllm.py` →
  `src/providers/inference/vllm/engine.py`.
- Rewrite импорта в `src/providers/single_node/inference/provider.py`.
- Если pipeline сам где-то импортирует VLLMEngine — rewrite через
  providers (или, если идиоматичнее, через protocol в shared).

**Тесты:** переезд test_vllm.

### A.4 Перенести `RUNTIME_IMAGE` → shared

**Что:** константа из `src/runner/__about__.py`, импортируемая pipeline.

**Файлы:**
- Add: `src/constants/runtime.py` с `RUNTIME_IMAGE = "..."`.
- Rewrite в `src/pipeline/stages/managers/deployment/dependency_installer.py`.
- Re-export из `src/runner/__about__.py` для backward compat (runner им
  тоже пользуется).

**Тесты:** smoke (один import-test).

### A.5 Извлечь `IMLflowManager` Protocol в shared

**Что:** pipeline импортирует concrete `MLflowManager` из training. Это
самое неприятное cross-side coupling.

**Файлы:**
- Add: `src/infrastructure/mlflow/protocol.py` (или `src/utils/mlflow/`):
  ```python
  class IMLflowManager(Protocol):
      def is_active(self) -> bool: ...
      def end_run(self) -> None: ...
      # … минимальный набор, нужный pipeline.mlflow_attempt
  ```
- Rewrite `src/pipeline/mlflow_attempt/manager.py:29`: тип `MLflowManager`
  заменить на `IMLflowManager`.
- В `src/training/managers/mlflow_manager` ничего не менять (concrete
  имплементация).

**Тесты:** unit-test на pipeline.mlflow_attempt с mock-MLflowManager,
который имплементирует Protocol.

### A.6 Удалить `src/utils/config.py` facade

**Что:** 86-строчный backward-compat re-export из `src/config`.

**Файлы:**
- Delete: `src/utils/config.py`.
- Codemod (`grep -l "from src.utils.config import" src/`):
  - rewrite ВСЕ `from src.utils.config import X` → `from src.config import X`.
  - Affected: 10+ файлов из `src/pipeline/`.

**Тесты:** ничего нового — существующие тесты вылавливают регрессии.

### A.7 Перенести pod-only utilities из `utils/` → `training/`

**Что:** ДВА файла, оба используются только trainer'ом и не нужны Mac:

1. **`utils/container.py` (718 LOC)** — god-object DI-контейнер.
   Источник scc-160/208/219.
2. **`utils/memory_manager.py`** (выявлено в §19 Q1.1) —
   GPU memory introspection (`GPUInfo`, `GPUPreset`, `MemoryStats`,
   `MemoryManager`). Mac не имеет GPU. Используется только
   `training/orchestrator/{strategy_orchestrator,phase_executor/training_runner}.py`
   и тестами.

**Файлы:**
- Move: `src/utils/container.py` → `src/training/container.py`.
- Move: `src/utils/memory_manager.py` → `src/training/memory_manager.py`.
- Rewrite импортов в:
  - `src/training/run_training.py`
  - `src/training/callbacks/training_events_callback.py`
  - `src/training/orchestrator/strategy_orchestrator.py`
  - `src/training/orchestrator/chain_runner.py`
  - `src/training/orchestrator/phase_executor/{mlflow_logger,executor,training_runner}.py`
  - `src/tests/test_dataset_loaders.py`
  - `src/tests/test_memory_manager.py` → переехать в `src/tests/unit/training/`
  - `src/tests/unit/test_phase_executor.py` (если импортит memory_manager)
  - `src/tests/unit/utils/test_container_*.py` → переехать в `src/tests/unit/training/`
  - `src/tests/unit/utils/test_memory_manager_*.py` (3 файла) →
    переехать в `src/tests/unit/training/`
  - `src/tests/integration/test_memory_*.py` (2 файла) →
    переехать в `src/tests/integration/training/`
  - `src/tests/e2e/test_strategy_orchestrator_e2e.py`
- Удаляет 3 SCC из 16 (scc-160, scc-208, scc-219 все вращаются вокруг
  container.py). Также убирает GPU-specific код из shared.

**Тесты:** существующие test_container_*.py + test_memory_manager_*.py —
переезд + проверка зелёного.

### A.8 (необязательная, опционально) Локальные циклы

`scc-273` (dataset_validator), `scc-267` (runpod/inference/__init__.py) —
fix внутренних циклов через reorder импортов или extract-interface. Они
не пересекают будущие границы пакетов, так что можно отложить на Phase C.

### A.* — выходной критерий Phase A

**Definition of Done:**
- ✅ `grep -rn "^from src\.pipeline" src/{providers,runner,training,api}` — пусто.
- ✅ `grep -rn "^from src\.api" src/{pipeline,providers,runner,training}` — пусто.
- ✅ `grep -rn "^from src\.training" src/{pipeline,providers,runner,api}` — пусто.
- ✅ `grep -rn "^from src\.runner" src/{pipeline,providers,training,api}` — пусто.
- ✅ Все 164+ unit-теста ветки зелёные.
- ✅ `make test-unit` green.

---

## 5. Phase B — Big-bang Move (single PR, all-or-nothing)

После Phase A граф чист. Теперь физически переносим файлы.

### B.1 Skeleton

```bash
mkdir -p packages/{shared,community,pod,providers,control}/src
```

Создать `packages/<pkg>/pyproject.toml` для каждого по шаблону §3.4.

### B.2 git mv (preserve history)

Для каждого пакета:

```bash
# shared
git mv src/utils                   packages/shared/src/ryotenkai_shared/utils
git mv src/config                  packages/shared/src/ryotenkai_shared/config
git mv src/constants               packages/shared/src/ryotenkai_shared/constants
git mv src/inference               packages/shared/src/ryotenkai_shared/inference
git mv src/infrastructure          packages/shared/src/ryotenkai_shared/infrastructure

# community (plugin framework)
git mv src/community               packages/community/src/ryotenkai_community

# pod (runner + trainer)
git mv src/runner                  packages/pod/src/ryotenkai_pod/runner
git mv src/training                packages/pod/src/ryotenkai_pod/trainer

# providers
git mv src/providers               packages/providers/src/ryotenkai_providers
git mv src/pipeline/inference/vllm packages/providers/src/ryotenkai_providers/inference/vllm  # уже перенесён в Phase A.3

# control
git mv src/pipeline                packages/control/src/ryotenkai_control/pipeline
git mv src/api                     packages/control/src/ryotenkai_control/api
git mv src/cli                     packages/control/src/ryotenkai_control/cli
git mv src/data                    packages/control/src/ryotenkai_control/data
git mv src/evaluation              packages/control/src/ryotenkai_control/evaluation
git mv src/reports                 packages/control/src/ryotenkai_control/reports
git mv src/workspace               packages/control/src/ryotenkai_control/workspace
git mv src/cli_state               packages/control/src/ryotenkai_control/cli_state
```

### B.3 Перенос тестов

Каждый тест едет за своим production-кодом:

```bash
# Например:
git mv src/tests/unit/utils         packages/shared/tests/unit/utils
git mv src/tests/unit/community     packages/community/tests/unit
git mv src/tests/unit/runner        packages/pod/tests/unit/runner
git mv src/tests/unit/training      packages/pod/tests/unit/trainer
git mv src/tests/unit/providers     packages/providers/tests/unit
git mv src/tests/unit/pipeline      packages/control/tests/unit/pipeline
git mv src/tests/unit/api           packages/control/tests/unit/api
git mv src/tests/integration        packages/control/tests/integration  # default
git mv src/tests/contract           packages/control/tests/contract
git mv src/tests/conftest.py        packages/control/tests/conftest.py  # split позже
```

Общий conftest.py — лежит на верхнем уровне `packages/_test_support/`
(внутренний пакет, не workspace member; либо просто скопировать в каждый
`tests/conftest.py`).

### B.4 Codemod импортов

**Codemod выполняется в ДВА прохода — не один!** AST-уровневый pass
ловит только `from … import …` / `import …`. Строковые литералы (mock.patch,
`importlib.import_module`, subprocess argv) требуют отдельного прохода.

#### B.4.1 Pass 1 — AST rewrite (libcst)

```bash
# Скрипт: scripts/codemod/rewrite_imports.py
# Использует libcst для AST-уровневого rewrite импортов.
# Mapping:
#   from src.utils.X        → from ryotenkai_shared.utils.X
#   from src.config         → from ryotenkai_shared.config
#   from src.constants      → from ryotenkai_shared.constants
#   from src.inference      → from ryotenkai_shared.inference
#   from src.infrastructure → from ryotenkai_shared.infrastructure
#   from src.community.X    → from ryotenkai_community.X
#   from src.runner.X       → from ryotenkai_pod.runner.X
#   from src.training.X     → from ryotenkai_pod.trainer.X
#   from src.providers.X    → from ryotenkai_providers.X
#   from src.pipeline.X     → from ryotenkai_control.pipeline.X
#   from src.api.X          → from ryotenkai_control.api.X
#   from src.cli.X          → from ryotenkai_control.cli.X
#   from src.data.X         → from ryotenkai_control.data.X
#   from src.evaluation.X   → from ryotenkai_control.evaluation.X
#   from src.reports.X      → from ryotenkai_control.reports.X
#   from src.workspace.X    → from ryotenkai_control.workspace.X
#   from src.cli_state.X    → from ryotenkai_control.cli_state.X
#
# Также: import src.X → import ryotenkai_<pkg>.X
# Применяется ко всем .py файлам в репо (не только packages/), включая:
# packages/, community/ (плагины), examples/, docker/, scripts/.
```

#### B.4.2 Pass 2 — String-literal rewrite (КРИТИЧНО)

**Найдено grep-ом 2026-05-03:** 518 строк типа `patch("src.X")` +
~10 dynamic imports + 2 subprocess argv. Codemod на импортах их НЕ ловит.

Целевые паттерны (regex-уровень, осторожно с false positives):

```python
# Скрипт: scripts/codemod/rewrite_string_refs.py
# Регулярки на double-quoted и single-quoted "src.X..." строки.

PATTERNS = [
    # 1. mock.patch / monkeypatch — основная масса
    (r'(["\'])src\.utils\.([\w.]+)\1',         r'\1ryotenkai_shared.utils.\2\1'),
    (r'(["\'])src\.config(\.[\w.]*|\1)',       r'\1ryotenkai_shared.config\2'),
    (r'(["\'])src\.constants(\.[\w.]*|\1)',    r'\1ryotenkai_shared.constants\2'),
    (r'(["\'])src\.inference(\.[\w.]*|\1)',    r'\1ryotenkai_shared.inference\2'),
    (r'(["\'])src\.infrastructure\.([\w.]+)\1', r'\1ryotenkai_shared.infrastructure.\2\1'),
    (r'(["\'])src\.community\.([\w.]+)\1',     r'\1ryotenkai_community.\2\1'),
    (r'(["\'])src\.runner\.([\w.]+)\1',        r'\1ryotenkai_pod.runner.\2\1'),
    (r'(["\'])src\.training\.([\w.]+)\1',      r'\1ryotenkai_pod.trainer.\2\1'),
    (r'(["\'])src\.providers\.([\w.]+)\1',     r'\1ryotenkai_providers.\2\1'),
    (r'(["\'])src\.pipeline\.([\w.]+)\1',      r'\1ryotenkai_control.pipeline.\2\1'),
    (r'(["\'])src\.api\.([\w.]+)\1',           r'\1ryotenkai_control.api.\2\1'),
    (r'(["\'])src\.cli\.([\w.]+)\1',           r'\1ryotenkai_control.cli.\2\1'),
    (r'(["\'])src\.data\.([\w.]+)\1',          r'\1ryotenkai_control.data.\2\1'),
    (r'(["\'])src\.evaluation\.([\w.]+)\1',    r'\1ryotenkai_control.evaluation.\2\1'),
    (r'(["\'])src\.reports\.([\w.]+)\1',       r'\1ryotenkai_control.reports.\2\1'),
    (r'(["\'])src\.workspace\.([\w.]+)\1',     r'\1ryotenkai_control.workspace.\2\1'),
    (r'(["\'])src\.cli_state\.([\w.]+)\1',     r'\1ryotenkai_control.cli_state.\2\1'),
    # 2. src.main (entry point string)
    (r'(["\'])src\.main(:[\w]+)?\1',           r'\1ryotenkai_control.main\2\1'),
]
```

**Места которые точно затронет (все найдены grep-ом 2026-05-03):**

| # файлов | Pattern | Примеры |
|---:|---|---|
| 518 | `patch("src.X.Y")` в тестах | [src/tests/test_dataset_loaders.py:296](src/tests/test_dataset_loaders.py:296) и др. |
| 7 | `importlib.import_module("src.X")` | [src/config/validators/cross.py:148](src/config/validators/cross.py:148), [src/providers/training/factory.py:203](src/providers/training/factory.py:203) |
| 1 | `sys.modules.get("src.X")` | [src/pipeline/stages/model_retriever/retriever.py:179](src/pipeline/stages/model_retriever/retriever.py:179) |
| 2 | subprocess argv `"src.X.Y"` | [training_launcher.py:281](src/pipeline/stages/managers/deployment/training_launcher.py:281) (`"src.training.run_training"`), [runtime.py:179](src/pipeline/launch/runtime.py:179) (`"src.pipeline.worker"`) |
| ~10 | `patch("src.config.validators.cross.importlib.import_module")` | [test_cross_validators.py](src/tests/unit/config/validators/test_cross_validators.py) — двойная косвенность |
| 1 | re-export comment | [model_retriever/__init__.py:16](src/pipeline/stages/model_retriever/__init__.py:16) |

**Verification после Pass 2:**
```bash
# После B.4.2 этот grep должен вернуть пусто:
grep -rn '"src\.\|'\''src\.' --include="*.py" \
    --exclude-dir=__pycache__ --exclude-dir=.venv \
    --exclude-dir=.git . | grep -v '\.md:'
```

#### B.4.3 Pass 3 — non-Python configs

| Файл | Что искать | На что менять |
|---|---|---|
| [Makefile:203](Makefile:203) | `src.api.openapi_dump` | `ryotenkai_control.api.openapi_dump` |
| [Makefile:97-101,108-115,119-121](Makefile) | `src/`, `src.tests` paths | `packages/*/src/`, `packages/*/tests/` |
| [.pre-commit-config.yaml:67,85,99,114,125](.pre-commit-config.yaml) | `^src/`, `^src/tests/` | `^packages/.*/src/`, `^packages/.*/tests/` |
| [pyrightconfig.json](pyrightconfig.json) | include patterns | per-package include + extraPaths |
| [pytest.ini](pytest.ini) | testpaths | `packages/*/tests` |
| [.dockerignore:71](.dockerignore) | `src/tests` | `packages/*/tests` |
| [.claude/launch.json:18](.claude/launch.json:18) | `"-m", "src.main"` | `"-m", "ryotenkai_control.main"` |
| [examples/quickstart-qlora-sft/pipeline_config.yaml:14](examples/quickstart-qlora-sft/pipeline_config.yaml:14) | `python -m src.pipeline.orchestrator …` (комментарий) | `python -m ryotenkai_control.pipeline.orchestrator …` |
| [examples/quickstart-qlora-sft/README.md:60](examples/quickstart-qlora-sft/README.md:60) | то же | то же |
| [docker/inference/build_and_push.sh:184](docker/inference/build_and_push.sh:184) | `src/inference/__about__.py` | `packages/shared/src/ryotenkai_shared/inference/__about__.py` |
| [docker/training/Dockerfile.runtime](docker/training/Dockerfile.runtime) | install из src/ | install из packages/{shared,community,pod} |
| [docker/training/entrypoint.sh](docker/training/entrypoint.sh) | uvicorn launch line | `ryotenkai_pod.runner.main:app` |
| [setup.sh](setup.sh), [run.sh](run.sh) | `src.X` paths | `ryotenkai_*` |
| `docs/` | best-effort, не блокирует | — |

**Affected scope:** ~1500 .py файлов в pass 1, ~70-80 файлов в pass 2,
~12 не-Python файлов в pass 3. Каждый pass — отдельный коммит внутри
коммита-5 PR Phase B.

### B.5 Внешние ссылки на `src.X`

См. полную таблицу в §B.4.3. Кратко: Makefile, .pre-commit-config.yaml,
pyrightconfig.json, pytest.ini, .dockerignore, .claude/launch.json,
examples/quickstart-qlora-sft/{pipeline_config.yaml,README.md},
docker/{training,inference}/{Dockerfile,entrypoint.sh,build_and_push.sh},
setup.sh, run.sh, root pyproject.toml.

### B.6 Console scripts

```toml
# packages/control/pyproject.toml
[project.scripts]
ryotenkai = "ryotenkai_control.cli.app:app"

# packages/pod/pyproject.toml
[project.scripts]
ryotenkai-trainer-run = "ryotenkai_pod.trainer.run_training:main"
# Runner запускается через uvicorn (см. ниже), отдельный console_script
# не нужен.
```

`runner_launcher.py` теперь делает:
```python
"/usr/local/bin/python3 -m uvicorn ryotenkai_pod.runner.main:app ..."
```

`python -m uvicorn` — каноничный способ запуска FastAPI; uvicorn сам
управляет lifecycle. Сохраним.

### B.7 Удалить старый `src/`

В конце Phase B `src/` пуст (только `__init__.py`, `__pycache__/`).
Удалить полностью:

```bash
git rm -r src
```

### B.8 Обновить `web/src/api/openapi.json`

```bash
make openapi  # = uv run -m ryotenkai_control.api.openapi_dump
```

Проверить, что сгенерированный `openapi.json` идентичен старому
(operationId, schema names, route paths не должны измениться).

### B.* выходной критерий Phase B

**Definition of Done:**
- ✅ `find . -path ./packages -prune -o -name '*.py' -print | xargs grep -l '^from src\.\|^import src\.'` — пусто.
- ✅ `uv sync` (с workspace-config) проходит.
- ✅ `uv run --package ryotenkai-control pytest packages/control/tests/unit` зелёный.
- ✅ `uv run --package ryotenkai-pod pytest packages/pod/tests/unit` зелёный.
- ✅ `uv run --package ryotenkai-providers pytest packages/providers/tests/unit` зелёный.
- ✅ `uv run --package ryotenkai-community pytest packages/community/tests/unit` зелёный.
- ✅ `uv run --package ryotenkai-shared pytest packages/shared/tests/unit` зелёный.
- ✅ `make openapi` генерирует identical schema.
- ✅ Console scripts работают: `ryotenkai --help`, `ryotenkai-trainer-run --help`.
- ✅ `python -m uvicorn ryotenkai_pod.runner.main:app` — startup smoke OK.
- ✅ Docker image build (training runtime) — успешный, размер измерен.

---

## 6. Phase C — Polish & Tooling

### 6.1 importlinter contracts

```toml
# Root pyproject.toml
[tool.importlinter]
root_packages = ["ryotenkai_shared", "ryotenkai_community", "ryotenkai_pod",
                 "ryotenkai_providers", "ryotenkai_control"]

[[tool.importlinter.contracts]]
name = "shared has no internal deps"
type = "forbidden"
source_modules = ["ryotenkai_shared"]
forbidden_modules = ["ryotenkai_community", "ryotenkai_pod",
                     "ryotenkai_providers", "ryotenkai_control"]

[[tool.importlinter.contracts]]
name = "community depends only on shared"
type = "forbidden"
source_modules = ["ryotenkai_community"]
forbidden_modules = ["ryotenkai_pod", "ryotenkai_providers", "ryotenkai_control"]

[[tool.importlinter.contracts]]
name = "pod depends only on shared+community"
type = "forbidden"
source_modules = ["ryotenkai_pod"]
forbidden_modules = ["ryotenkai_providers", "ryotenkai_control"]

[[tool.importlinter.contracts]]
name = "providers depend only on shared (no community)"
type = "forbidden"
source_modules = ["ryotenkai_providers"]
forbidden_modules = ["ryotenkai_community", "ryotenkai_pod", "ryotenkai_control"]

[[tool.importlinter.contracts]]
name = "control may depend on shared+community+providers, not on pod"
type = "forbidden"
source_modules = ["ryotenkai_control"]
forbidden_modules = ["ryotenkai_pod"]

# Внутри ryotenkai_pod допускаем cross-references runner ↔ trainer
# (на самом деле runner spawn-ит trainer как subprocess — без import).
# Но защитимся: trainer не должен импортить runner (и наоборот).
[[tool.importlinter.contracts]]
name = "trainer subpackage does not import runner subpackage"
type = "forbidden"
source_modules = ["ryotenkai_pod.trainer"]
forbidden_modules = ["ryotenkai_pod.runner"]

[[tool.importlinter.contracts]]
name = "runner subpackage does not import trainer subpackage"
type = "forbidden"
source_modules = ["ryotenkai_pod.runner"]
forbidden_modules = ["ryotenkai_pod.trainer"]
```

CI gate: `uv run lint-imports`. Зелёный — обязателен.

### 6.2 CI updates

- Per-package matrix: тесты гоняются параллельно для каждого пакета.
- Coverage: combined report (`pytest --cov=packages/*/src` + merge).
- Lint: ruff/black/mypy остаются глобальными (workspace-level), но запускаются
  с per-package output.

### 6.3 Pre-commit

`.pre-commit-config.yaml` — обновить пути с `src/` на `packages/*/src/`.
Hooks ничего другого не меняют.

### 6.4 Test fixtures: `_test_support/` (опциональный)

Если общие conftest.py-fixtures начнут дрейфовать между пакетами — создать
**internal-only** workspace member `packages/_test_support/`, NOT
installable, только для dev-deps. Хранит общие фикстуры (`tmp_workspace`,
`mock_secrets`, `dummy_pipeline_config`).

Альтернатива: каждый пакет держит свои фикстуры в `tests/conftest.py`,
без cross-package совместного использования. Проще, до тех пор пока не
будет дублирования.

### 6.5 Documentation

- Update `README.md`: новая структура.
- Update `CLAUDE.md` (project instructions): новые import paths.
- Update `CONTRIBUTING.md`: how-to add a new package.
- ADR (`docs/adrs/`): "Decision: split src/ into 5 uv-workspace packages"
  с rationale.
- НЕ обновлять старые `docs/plans/` — они исторический record.

### 6.6 Docker

Pod-image `docker/training/Dockerfile.runtime` использует **editable
installs** (§19 Q3.1) — это позволяет CodeSyncer продолжать rsync
обновления без `pip install --force-reinstall` каждый раз.

```dockerfile
# Multi-stage build с editable install
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY packages/shared    ./packages/shared
COPY packages/community ./packages/community
COPY packages/pod       ./packages/pod
RUN pip install uv && \
    # Resolve и install только runtime deps (без workspace members)
    uv export --package ryotenkai-pod \
              --frozen --no-dev \
              --no-emit-package ryotenkai-pod \
              --no-emit-package ryotenkai-community \
              --no-emit-package ryotenkai-shared \
              > requirements.txt && \
    pip install -r requirements.txt

FROM python:3.12-slim
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Editable install workspace members в фиксированный path /workspace.
# CodeSyncer rsync-ит сюда новый код — Python подхватывает без reinstall.
WORKDIR /workspace
COPY packages/shared    ./packages/shared
COPY packages/community ./packages/community
COPY packages/pod       ./packages/pod
RUN pip install --no-deps -e ./packages/shared \
                          -e ./packages/community \
                          -e ./packages/pod

# entrypoint.sh запускает uvicorn ryotenkai_pod.runner.main:app
COPY docker/training/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

**Важно:** CodeSyncer обновляет `/workspace/packages/{shared,community,pod}/`
через rsync. Editable install (`pip install -e`) создаёт `.pth` файл,
указывающий на эти пути — Python видит новый код мгновенно, без
reinstall.

**Drift guard:** `pip show ryotenkai-pod` после rsync должен показать
`Location: /workspace/packages/pod/src` — т.е. editable путь, не site-packages.

Reference: [Jimmy Yeung's uv-workspace migration](https://dev.to/jimmyyeung/journey-migrating-to-uv-workspace-34a7).

### 6.7 Coverage gate

Текущий gate: `fail_under = 83`. После split:
- **Combined gate:** 83 (как сейчас) — для root coverage report.
- **Per-package gate:** ПОСЛЕ Phase B обязательно сделать **calibration
  step** (§19 Q4.2):
  ```bash
  for pkg in shared community pod providers control; do
    uv run --package ryotenkai-$pkg pytest packages/$pkg/tests \
      --cov=packages/$pkg/src --cov-report=term | tail -3
  done
  ```
  Записать измеренные coverage % per-package, затем set floor =
  `measured - 2` в каждом `packages/<X>/pyproject.toml` под
  `[tool.coverage.report].fail_under`.
- **НЕ назначать floors заранее** (числа shared 90 / community 85 /
  pod 78 / providers 70 / control 75 — это иллюстрация, не команда).

---

## 7. Tests Strategy (7 категорий, обязательно)

Согласно policy пользователя, для каждого нового/измененного юнита кода
обязательны 7 типов тестов:

1. **Positive paths** — happy path, основной use case.
2. **Negative paths** — ошибочный input, отказы зависимостей.
3. **Boundary conditions** — пустой dataset, single row, max-length config.
4. **Invariants** — DI-контейнер всегда возвращает singleton; Result-types
   `Ok` ⊕ `Failure`.
5. **Dependency-error simulation** — сетевые таймауты, ssh-disconnect,
   subprocess crash; assert корректный AppError-code.
6. **Regression scenarios** — фикс конкретных багов из git log
   (например: 16-crash chain — sentinel test, который импортит трансзависимости
   и assert-ит, что `ryotenkai_providers` НЕ может затащить `ryotenkai_control`).
7. **Combinatorial edge** — pairwise combinations для config schemas
   (per-strategy, per-provider).

### 7.1 Phase-specific:

- **Phase A:** для каждого PR обязательны категории 1, 2, 5, 6.
- **Phase B:** sentinel-тесты на boundaries (importlinter-style, in-test):
  ```python
  def test_no_pod_imports_in_control():
      """Regression guard for the architectural border."""
      import ast
      for path in (Path("packages/control/src")).rglob("*.py"):
          tree = ast.parse(path.read_text())
          for node in ast.walk(tree):
              if isinstance(node, ast.ImportFrom) and node.module:
                  assert not node.module.startswith("ryotenkai_pod"), path
  ```
- **Phase C:** все 7 категорий + e2e dry-run (mock pod).

### 7.2 Drift guards (как в `test_required_modules_ships_full_src_tree`)

- `test_pod_depends_only_on_shared` — runtime check.
- `test_trainer_subprocess_uses_console_script_only` — runner вызывает
  trainer ТОЛЬКО через `subprocess.Popen(["ryotenkai-trainer-run", ...])`,
  а не через `import ryotenkai_pod.trainer`.
- `test_codesyncer_ships_only_pod_packages` — REQUIRED_PACKAGES = ["packages/shared", "packages/community", "packages/pod"].

### 7.3 Симметричные boundary sentinels (из §19 Q4.1)

Пара sentinel-тестов на КАЖДУЮ запрещённую границу — defense in depth
поверх importlinter (static check). importlinter может быть отключён
случайно (typo в config), sentinel test упадёт всегда.

```python
# packages/control/tests/sentinel/test_no_pod_imports.py
def test_control_does_not_import_pod():
    """Симметрия с test_pod_does_not_import_control."""
    for path in Path("packages/control/src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("ryotenkai_pod"), \
                    f"{path}: imports ryotenkai_pod (pod-side code in control)"

# packages/pod/tests/sentinel/test_no_control_imports.py
def test_pod_does_not_import_control():
    for path in Path("packages/pod/src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("ryotenkai_control"), \
                    f"{path}: imports ryotenkai_control (control-side code in pod)"
                assert not node.module.startswith("ryotenkai_providers"), \
                    f"{path}: imports ryotenkai_providers (Mac-only deploy code in pod)"

# packages/providers/tests/sentinel/test_no_pod_imports.py
def test_providers_does_not_import_pod():
    for path in Path("packages/providers/src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("ryotenkai_pod"), path
                assert not node.module.startswith("ryotenkai_control"), path
                assert not node.module.startswith("ryotenkai_community"), path

# packages/shared/tests/sentinel/test_shared_is_leaf.py
def test_shared_does_not_import_anything_internal():
    for path in Path("packages/shared/src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in ["ryotenkai_community", "ryotenkai_pod",
                                  "ryotenkai_providers", "ryotenkai_control"]:
                    assert not node.module.startswith(forbidden), \
                        f"{path}: shared imports {forbidden} (must be leaf)"
```

Итого: **4 пары sentinels** (8 тестов), покрывающие ВСЕ запрещённые
направления графа §2.2.

---

## 8. 3-iteration Risk Audit

> Каждая итерация — взгляд под другим углом. Все риски с **mitigation**
> (план действий) или **decision** (решение принято в плане).

### 8.1 Iteration 1 — Архитектурные риски

| # | Риск | Severity | Mitigation / Decision |
|---|---|---|---|
| R1 | Тестовые фикстуры (`src/tests/conftest.py`) cross-cut. После split может оказаться, что фикстура нужна в 2-3 пакетах. | High | **Decision:** dedicated `packages/_test_support/` (NOT a workspace member, dev-deps only). Если не нужно — сделать на Phase C, не блокирует Phase B. |
| R2 | Pickling/dill — сериализация полных имён классов между Mac и pod. | Medium | **Mitigation:** проверено grep'ом — `pickle.dump`/`dill.dump` в production-коде НЕТ. Безопасно. |
| R3 | MLflow run inheritance — runs создаются в trainer (`ryotenkai_pod.trainer`) и потом читаются в control. Class names в metadata? | Medium | **Mitigation:** MLflow хранит params/metrics/tags как plain key/value, не FQN классов. Безопасно. Подтвердить в Phase A.5. |
| R4 | Pydantic class identity — `PipelineConfig` в shared. Если control и trainer на разных версиях shared — validation drift. | Low | **Mitigation:** workspace enforce-ит single version. До разделения версий невозможно. |
| R5 | Root `community/` плагины импортят `src.X` (см. grep §2.5: src.community, src.training, src.evaluation, src.data, src.reports, src.utils). | High | **Mitigation:** rewrite импортов в Phase B (это ~25 файлов в `community/`). Plugin manifest schema может потребовать version-pin. |
| R6 | Console script entry: `ryotenkai = "src.main:app"`. | Low | **Decision:** в Phase B меняем на `"ryotenkai_control.cli.app:app"`. Тривиально. |
| R7 | `setuptools.packages.find { include = ["src*"] }` ломает workspace. | Low | **Decision:** Phase B убирает root setuptools-config. Каждый member использует свой `[build-system]`. |
| R8 | Re-exports в `__init__.py`. ~50 файлов с `from .X import Y`. | Medium | **Mitigation:** codemod (libcst) обрабатывает; ручной аудит каждого `__init__.py` в Phase B. |
| R9 | API/control единственный пакет, импортирующий fastapi. Pipeline — на других машинах вообще НЕ нужен fastapi (он же не сервер). | Low | **Decision:** fastapi в `ryotenkai-control` deps — норма. Pipeline-only тесты не подтянут fastapi (если организовано чисто). |
| R10 | Workspace package uses `src/` layout — IDE indexing (pyright/pylance) должен корректно обрабатывать. | Low | **Mitigation:** Astral docs подтверждают src-layout совместим. Проверим в Phase B. |

### 8.2 Iteration 2 — Операционные риски

| # | Риск | Severity | Mitigation / Decision |
|---|---|---|---|
| R11 | CodeSyncer rsync logic меняется радикально (REQUIRED_MODULES → REQUIRED_PACKAGES). Регрессия в pod-deploy. | High | **Mitigation:** Phase B включает обновление CodeSyncer + drift-guard test (test_codesyncer_ships_only_pod_packages). Phase 0 уже ввёл "ship full src/" guard — заменим на "ship full packages/{shared,community,pod}/". |
| R12 | Docker build: multi-stage с `uv export --package`. Незнакомый паттерн. | Medium | **Mitigation:** reference Jimmy Yeung article + Apache Airflow Dockerfile. Тестируется локально перед merge. |
| R13 | Pre-commit hooks: пути `src/` → `packages/*/src/`. Может сломать формат-проверки. | Low | **Mitigation:** обновить `.pre-commit-config.yaml` в Phase B; smoke-test `pre-commit run --all-files`. |
| R14 | mypy strict mode пакетный — текущий relaxed mode мог скрывать type errors через циклы. После разделения могут всплыть. | Medium | **Mitigation:** mypy остаётся relaxed на Phase B; per-package strict mode — отдельная задача в Phase C. |
| R15 | Pytest testpaths: текущий `["src"]`. Изменение влияет на test discovery. | Low | **Mitigation:** root `pyproject.toml` остаётся (workspace root); testpaths становится `["packages/*/tests"]`. |
| R16 | Pre-existing 9 failing tests (per current coverage report) — могут стать ещё более flaky после переезда. | Medium | **Decision:** до Phase B — `git log --grep "@pytest.mark.skip"` и каталогизировать. Не блокирует Phase B (они уже сломанные). |
| R17 | Worktree: PR с переездом ~1500 файлов — невозможно ревьюить на github. | High | **Mitigation:** Phase B разделить на 5 коммитов в одной PR: (1) skeleton + pyproject.toml-ы (5 пакетов), (2) git mv shared+community, (3) git mv pod (runner+trainer), (4) git mv providers+control, (5) codemod imports + delete src/. Каждый коммит ревьюаемый. PR один — atomic merge. **Решение пользователя — закреплено.** |
| R18 | IDE workspace indexing — VSCode + pyright. Может потребоваться `python.analysis.extraPaths`. | Low | **Mitigation:** workspace-aware из коробки в современных pyright/pylance. Проверим, добавим конфиг при необходимости. |
| R19 | `make openapi` (Makefile:203) — `python -m src.api.openapi_dump` ломается. | Low | **Decision:** Phase B обновляет Makefile-target на `python -m ryotenkai_control.api.openapi_dump`. |
| R20 | Coverage merge между пакетами — pytest-cov по дефолту делает один data file. | Low | **Mitigation:** uv `pytest --cov=packages/X/src` для каждого + `coverage combine` в CI. |

### 8.3 Iteration 3 — Стратегические / Tail-риски

| # | Риск | Severity | Mitigation / Decision |
|---|---|---|---|
| R21 | Связь с transport unification plan (`2026-05-03-transport-unification-http-only-runtime.md`). Тот план touches те же файлы. | Medium | **Decision:** этот план идёт ПЕРВЫМ. Transport plan rebases поверх и продолжает (его import paths становятся `ryotenkai_*`). Phase 3.2 transport plan (importlinter) — надёжно работает поверх packages border. |
| R22 | Frontend (`web/`): `web/src/api/openapi.json` зависит от api routes. После rename может появиться diff. | Low | **Mitigation:** route paths не меняются, только Python paths внутри. openapi.json должен быть identical. Проверка в Phase B-DoD. |
| R23 | Pipeline state JSON (`runs/<run-id>/pipeline_state.json`) — сериализация stage-state. Если schema хранит FQN — ломается. | Medium | **Mitigation:** проверить `src/pipeline/state/store.py` на dill/pickle. Если только pydantic-JSON — безопасно. Если есть FQN — добавить migration helper в Phase B. |
| R24 | Versioning policy: все 5 пакетов держат `1.0.0`. Если вдруг кому-то понадобится bump одного без других — bump-tool? | Low | **Decision:** initial — все `1.0.0`, синхронный bump. Independent versioning — future work. |
| R25 | PyPI publishing: не сейчас. Но префикс `ryotenkai-*` — registered? | Low | **Decision:** out of scope. Если в будущем понадобится — зарегистрировать заранее. |
| R26 | Transitive imports в скриптах вне `src/` (root `community/`, `examples/`, `scripts/` — последнего нет). Потребуют rewrite. | Medium | **Mitigation:** codemod охватывает любые `.py` файлы в репозитории, не только `src/`. Грeп после Phase B: пусто. |
| R27 | Performance: `uv sync` в монорепе медленнее, чем single-package. | Low | **Decision:** acceptable. uv быстрее всего альтернативного. Cache shared между members. |
| R28 | `setup.sh` / `Makefile` / `run.sh` могут содержать `src.X` ссылки. | Low | **Mitigation:** `grep -rn 'src\.' Makefile setup.sh run.sh` в Phase B; rewrite найденные. |
| R29 | Documentation drift: `docs/plans/*.md` ссылаются на `src/X` paths. | Negligible | **Decision:** не обновлять (исторические записи). Новые планы пишут в новой нотации. |
| R30 | CONTRIBUTING.md рассказывает про single src/-структуру. | Low | **Mitigation:** Phase C обновляет. |
| R31 | **518 строк `mock.patch("src.X")` в тестах.** Codemod на импортах их НЕ ловит. Если пропустить — все тесты с моками сразу красные. | **CRITICAL** | **Mitigation:** §B.4.2 обязательный second pass codemod на string literals (regex). `grep -rn '"src\.'` после Pass 2 должен быть пустым. |
| R32 | **Dynamic imports** через `importlib.import_module("src.X")` — 7 мест в src/, ~10 в тестах. | High | **Mitigation:** §B.4.2 Pass 2 ловит. Sentinel: `grep -rn 'import_module("src\.\|sys\.modules\.get("src\.'` пуст после Phase B. |
| R33 | **Subprocess argv с строкой src.X**: training_launcher.py:281 (`"src.training.run_training"`), runtime.py:179 (`"src.pipeline.worker"`). Ломает рантайм если не обновить. | High | **Mitigation:** §B.4.2 Pass 2 + ручная верификация. Subprocess вызов trainer становится либо `ryotenkai-trainer-run` (console_script — preferred), либо `python -m ryotenkai_pod.trainer.run_training`. |
| R34 | `.claude/launch.json:18` — `python -m src.main serve`. | Low | **Decision:** §B.4.3 включает обновление. После Phase B — `ryotenkai_control.main`. |
| R35 | `examples/quickstart-qlora-sft/{pipeline_config.yaml,README.md}` — туториал ссылается на `python -m src.pipeline.orchestrator`. | Medium | **Mitigation:** §B.4.3 обновляет. Юзеры столкнутся с ImportError если пропустим. |
| R36 | `docker/inference/build_and_push.sh:184` — ссылка на `src/inference/__about__.py`. | Low | **Decision:** §B.4.3 обновляет на `packages/shared/src/ryotenkai_shared/inference/__about__.py`. |
| R37 | `.dockerignore:71` excludes `src/tests`. После Phase B путь — `packages/*/tests`. | Low | **Decision:** §B.4.3 обновляет. |
| R38 | `.pre-commit-config.yaml` имеет hard-coded `^src/(config|data|pipeline|providers|reports|training|utils)/` (line 67) и `^src/` (lines 85, 99, 114) и exclude `^(generated_reports/|src/tests/|...)` (line 125). Регрессия → pre-commit пропускает или ломает не те файлы. | Medium | **Mitigation:** Phase B полностью переписывает hooks-секцию под `^packages/.*/src/` и per-package excludes. Smoke: `pre-commit run --all-files`. |
| R39 | **CI отсутствует** (нет `.github/workflows/`). Не нужно ничего обновлять, но `make`-цели становятся единственным CI gate. | Low | **Decision:** out-of-scope (нет CI). Если в будущем добавим — workflow будет под packages-структуру. |
| R40 | `dill` в deps но НЕ используется в src/ (transitive через peft/trl). | Negligible | **Decision:** оставляем; `pip install --no-deps` точно даст deps. |
| R41 | Phase A.5 IMLflowManager Protocol — нужен **точный список методов**, которые pipeline вызывает на trainer-MLflowManager. Если Protocol неполный — runtime AttributeError. | Medium | **Mitigation:** перед написанием Protocol — `grep -rn "mlflow_manager\.\w" src/pipeline/` чтобы каталогизировать ВСЕ вызовы. Тест c mock-Protocol ловит регрессии. |
| R42 | Phase A.7 container.py (718 LOC) — может содержать ссылки на pipeline (cross-side import). Если есть — Phase A.7 не закрывает sccp-160 полностью. | Medium | **Mitigation:** перед переездом — `grep -n "from src.pipeline\|from src.api" src/utils/container.py`. Если есть — расширить scope A.7. |
| R43 | `src/pipeline/stages/model_retriever/__init__.py:16` — re-export `SSHClient` для теста-патча `patch("src.pipeline.stages.model_retriever.SSHClient")`. После B.4.2 patch-строка обновится — но re-export должен остаться, иначе patch упадёт. | Low | **Mitigation:** §B.4.2 Pass 2 не трогает code, только строки. Re-export сохраняется. |
| R44 | `src/config/validators/cross.py:148-156` — runtime `importlib.import_module("src.providers.training")` + проверка ошибки `if missing.startswith("src.providers")`. Это runtime detection если providers не установлен. | Medium | **Mitigation:** §B.4.2 Pass 2 обновит обе строки на `ryotenkai_providers.training` / `ryotenkai_providers`. Тест на runtime детект сохранится. |
| R45 | 11 уже-skipped тестов (`@pytest.mark.skipif`). Каталог — см. §18.4. Не блокируют, но нужно знать состояние baseline. | Negligible | **Decision:** документировано в §18.4. После Phase B те же тесты остаются skipped — не регрессия. |
| R46 | **Mock path semantics после codemod B.4.2** (см. §19 Q4.5). `mock.patch("X.Y.Z")` имеет специальную семантику: путь должен указывать на module, где Z **импортирован**, а не где определён. Codemod literal-rewrite сохраняет это для `from src.X import Y; patch("src.X.Y")` → `patch("ryotenkai_X.Y")` ✓. Но если `from src.utils.helper import do_thing` затем `patch("src.foo.bar.do_thing")` — local binding в `bar` ≠ `helper.do_thing`. Codemod может ошибочно унифицировать. | Medium | **Mitigation:** после Phase B обязательный шаг verification: `pytest --collect-only -v` ловит import errors, `pytest -x` ловит broken mocks (false-positive pass / false-negative fail). Если test поведение изменилось — ручной разбор каждого ambiguous patch. |

### 8.4 Открытые вопросы — статус (по итогам диалога с пользователем)

- ✅ **OQ-1 Naming:** **flat** (`ryotenkai_pod`, `ryotenkai_shared`,
  `ryotenkai_community`, …).
- ✅ **OQ-2 providers separate:** отдельный пакет (default из плана,
  пользователь не возражал).
- ✅ **OQ-3 pod packages:** **один** `ryotenkai-pod` (runner + trainer
  вместе).
- ✅ **OQ-4 Phase A blocking:** да, Phase A полностью завершается до
  Phase B.
- ✅ **OQ-5 Phase B PR:** 1 PR из 5 коммитов.
- ✅ **OQ-6 BC-shims:** удаляются в начале Phase B как часть codemod.
- ✅ **OQ-7 src/community framework:** **отдельный workspace member**
  `packages/community/` (видимый в корне `packages/`), а не растворяется
  в shared. Корневой `community/` — по-прежнему данные, не member.

---

## 9. Best Practices Alignment

### 9.1 PEP / стандарты

- ✅ **PEP 517/518** — `[build-system]` для каждого пакета.
- ✅ **PEP 621** — `[project]` metadata (name, version, deps).
- ✅ **PEP 660** — editable installs (uv делает их по умолчанию для workspace).
- ✅ **PEP 735** — dependency groups (используем optional-dependencies, не group syntax — для совместимости).
- ✅ **src-layout** — рекомендован Python Packaging User Guide для prevent accidental in-place imports.

### 9.2 Astral / uv

- ✅ Workspace per [Astral docs](https://docs.astral.sh/uv/concepts/projects/workspaces/).
- ✅ Single `uv.lock` для всех members.
- ✅ `[tool.uv.sources]` с `workspace = true` для cross-package.

### 9.3 Apache Airflow

- ✅ Namespace prefix `airflow.providers.amazon.X` → у нас `ryotenkai_<pkg>.X` (flat).
- ✅ Single workspace, multiple distributions.
- ✅ Pre-commit на root, не per-package.
- ⚠️ Airflow использует `prek` (faster pre-commit). Out of scope сейчас.

### 9.4 SOLID / KISS / DRY / YAGNI

- **SRP:** каждый пакет — одна причина для изменения (control plane vs pod runtime vs shared utils).
- **OCP:** providers — extension point (новый provider = новый module внутри `ryotenkai-providers`, без изменения остальных).
- **LSP:** `IMLflowManager` Protocol в shared — обеспечивает substitutability (mock/concrete).
- **ISP:** маленькие Protocols в shared, не fat interfaces.
- **DIP:** pipeline зависит от Protocol, не от concrete trainer-MLflowManager.
- **KISS:** flat package names (не namespace), один lock, один venv.
- **DRY:** общие deps только в shared.
- **YAGNI:** НЕ создаём `ryotenkai-cli` отдельным пакетом, НЕ выделяем `ryotenkai-data`, НЕ заводим `_test_support` если не нужен.

### 9.5 12-Factor

- **III. Config** — `ryotenkai-shared.config` — чистая Pydantic schema, ENV-driven.
- **VI. Processes** — runner/trainer — stateless processes, kicked off через subprocess.
- **VIII. Concurrency** — runner и trainer — separate processes (тут уже было).

---

## 10. Testing & Verification Strategy (per phase)

### 10.1 Phase A Verification

После каждого PR:
- `make test-unit` зелёный.
- `make lint` зелёный.
- `grep -rn "from src.<deprecated_path>" src/` — empty (для конкретного PR scope).
- Manual: запустить один realtime run (см. §10.4 acceptance smoke).

### 10.2 Phase B Verification

После переезда:
1. `uv sync` проходит чисто, single `.venv` создаётся.
2. `uv run --package <pkg> pytest packages/<pkg>/tests/unit -x` для всех 5 пакетов.
3. `uv run pytest` (вся монорепа) — все integration + contract тесты.
4. Sentinel-тесты §7.1 (no pod-imports in control, etc.).
5. `make openapi` — diff-test с baseline (golden file).
6. Console scripts: `ryotenkai --help`, `ryotenkai-trainer-run --help`, `python -m uvicorn ryotenkai_pod.runner.main:app` startup smoke.
7. Docker build training-image: `docker build -f docker/training/Dockerfile.runtime .` — успех + измеренный размер.

### 10.3 Phase C Verification

- importlinter contracts pass.
- Per-package coverage gates pass.
- Pre-commit `--all-files` clean.
- mypy clean.

### 10.4 Acceptance: end-to-end smoke

После Phase B (перед merge):
1. Запустить полный pipeline run в single_node provider (mock GPU).
2. Запустить run в RunPod provider (с настоящим pod).
3. Проверить что:
   - state files читаются корректно после restart.
   - mlflow runs создаются с правильными tags.
   - logs архивируются и достаются на Mac.
   - openapi.json identical с baseline.
4. Зафиксировать size training-image до/после.

### 10.5 3-attempts policy

Если тест падает после изменения — попыток фикс не более 3:
- Attempt 1: minimal fix (изменения < 10 строк).
- Attempt 2: deeper analysis (читаем источник теста, проверяем что именно сломалось).
- Attempt 3: STOP, делаем deep-think анализ. Если не находим решения — флагуем
  тест skip с TODO + создаём issue, и продолжаем. Запрещено игнорировать
  падающий тест без флага.

---

## 11. Связь с другими планами

### 11.1 Phase 0 (`docs/plans/2026-05-02-monitor-cleanup-and-control-plane-redesign.md`)

✅ **Завершено** — 3 коммита в feature branch:
- `e6e6f0f` monitor cleanup
- `5293bbf` ship full `src/`
- `93a7185` plan doc

Этот packagization план — естественное продолжение: ship full src/ →
теперь ship only `packages/{shared,runner,trainer}/`.

### 11.2 Transport Unification (`docs/plans/2026-05-03-transport-unification-http-only-runtime.md`)

📝 **APPROVED, not started.** План на Phase 1+2+3 с importlinter
(Phase 3.2). Этот packagization план **МЕНЯЕТ контекст**:

**Решение:** packagization идёт ПЕРВЫМ. После завершения:
- Transport plan rebases поверх packages.
- Phase 3.2 importlinter становится **дополнительной** проверкой поверх
  физических границ (defense-in-depth).
- Phase 1 (problem+json) и Phase 2 (HTTP-only runtime) — никак не
  пересекаются с границами; легко переносятся.

### 11.3 Будущие планы

Создавать всё **после** packagization. Любой новый код должен с самого
начала жить в правильном пакете. `src/` больше не существует.

---

## 12. Workflow & Process Compliance

Согласно правилам пользователя:

### 12.1 Принципы

- ✅ Senior Python / DevOps / MLOps роль — все решения обоснованы (uv vs
  poetry, src-layout vs flat, package count) с industry-grade фактурой.
- ✅ SOLID / KISS / DRY / YAGNI — см. §9.4.
- ✅ Best practices — см. §9 (Astral, PEP 621, Apache Airflow).
- ✅ Архитектурно-корректно "на перспективу" — physical границы вместо
  conventions; namespace-friendly даже если пока flat.
- ✅ NO backward compat — старый `src/` удалён в конце Phase B; никаких
  permanent shim-имиторов (только короткоживущие в Phase A между PRs).

### 12.2 Workflow

- ✅ **(1) Дипсинк-анализ перед каждой задачей** — выполнен (§§1-3).
- ✅ **(2) Helixir research в N итераций** — `search_memory` × 1
  (preliminary). До Phase A повторим перед каждым PR.
- ✅ **(3) План работ** — этот документ.
- ✅ **(4) 3-iteration risk audit** — §§8.1-8.3 (30 рисков).
- ✅ **(5) Deep-think для решения рисков** — каждый риск имеет
  Mitigation/Decision.
- ✅ **(6) Все риски + ответы зафиксированы** — §§8.1-8.4.
- ⏳ **(7) Ждать одобрения пользователя** — это сейчас.
- ✅ **(8) TODO листы** — активно используются (TodoWrite).
- ✅ **MCP активно** — repowise (overview, risk, search), helixir
  (search_memory), tavily (research, search, extract).

### 12.3 Tests strategy

7 категорий — см. §7. Применяется к КАЖДОМУ изменению начиная с Phase A.

### 12.4 3-attempts policy

См. §10.5.

---

## 13. Open Questions — RESOLVED

Все вопросы закрыты в диалоге с пользователем 2026-05-03:

| ID | Вопрос | Решение |
|---|---|---|
| OQ-1 | Naming: flat vs namespace? | ✅ **Flat** (`ryotenkai_pod`, `ryotenkai_shared`, `ryotenkai_community`, `ryotenkai_providers`, `ryotenkai_control`). |
| OQ-2 | `providers` separate? | ✅ **Separate** (default, не оспорено). |
| OQ-3 | runner + trainer — один пакет или два? | ✅ **Один** (`ryotenkai-pod`). |
| OQ-4 | Phase A blocking? | ✅ **Blocking** перед Phase B. PRы Phase A независимы между собой. |
| OQ-5 | Phase B — 1 PR / 5 коммитов? | ✅ **1 PR / 5 коммитов**. |
| OQ-6 | BC-shims после Phase A? | ✅ Удалить в начале Phase B как часть codemod. |
| OQ-7 | `src/community` framework — где живёт? | ✅ **Отдельный workspace member** `packages/community/` (видим в корне `packages/`). Корневой `community/` остаётся как данные плагинов. |
| OQ-8 | Layout пакетов: src-layout (`packages/foo/src/foo/`) vs flat (`packages/foo/foo/`)? | ✅ **src-layout** (`packages/<pkg>/src/ryotenkai_<pkg>/`). Стандарт PyPA, защищает от случайных cwd-импортов, pyproject.toml не попадает в import path. Дополнительная вложенность необходима — без неё все 5 пакетов получают одинаковое имя `src` на sys.path и коллизируют. |

---

## 14. Estimate

| Phase | PRs | LOC touched | ETA (single-developer) |
|---|---:|---:|---|
| **A. Pre-cleanup** | 6-7 | ~500 (move + rewrite imports) | 2-3 дня |
| **B. Big move** | 1 (5 commits) | ~1500 файлов, ~30k LOC import statements | 2-3 дня (5 пакетов: shared, community, pod, providers, control) |
| **C. Polish** | 3-4 | ~200 (Docker, CI, importlinter, docs) | 1-2 дня |
| **Total** | 10-12 | ~32k LOC touched | **5-8 дней** |

Это блокирует другие работы только в момент Phase B (atomic merge).

---

## 15. Definition of "done" для всего плана

- ✅ Phase A merged: 0 cross-package import violations в `src/`.
- ✅ Phase B merged: `src/` удалён, 5 пакетов работают, все тесты зелёные.
- ✅ Phase C merged: importlinter contracts pass, Docker image сжат на ≥50%.
- ✅ End-to-end smoke run на single_node + RunPod успешен.
- ✅ ADR записан, README/CLAUDE.md обновлены.
- ✅ `make openapi` без diff с baseline.

---

## 16. Что НЕ делаем (явно)

- ❌ Не трогаем `web/`, кроме Makefile target и openapi regeneration.
- ❌ Не разделяем версии пакетов (все `1.0.0`).
- ❌ Не публикуем на public PyPI.
- ❌ Не вводим bzlmod / Bazel / Pants.
- ❌ Не меняем wire-protocol (HTTP/SSH endpoints).
- ❌ Не переписываем существующий бизнес-логи код — только перемещаем.
- ❌ Не меняем CLI UX, Python API сигнатуры функций (за исключением
  заменённых facade re-exports — пользователи импортов получат
  ImportError, codemod это обработает).
- ❌ Не апгрейдим dependencies (torch/transformers/peft etc.) — отдельная
  задача.

---

## 17. Закрытие плана

Все open questions §13 решены (8/8). Все известные дыры закрыты §18 (15/15).
Полный risk-pool §8: **45 рисков**, ни одного «open».

После явного «GO» от пользователя:
1. Создать TODO list для Phase A (6-7 PRs).
2. Начать Phase A.1 — перенос `pipeline.cancellation` → `utils/cancellation`.
3. Каждый Phase A PR — отдельный commit + green tests + ревью.

🛑 **STOP** — никаких изменений до явного «GO».

---

## 18. Plan Audit — найденные дыры и закрытия

> Финальный аудит плана 2026-05-03 (по запросу пользователя «пробегись
> ещё раз, нет ли дыр»). Из 15 проверенных вещей **15 закрыты** —
> либо явным разделом плана, либо новым риском R31-R45.

### 19.1 Codemod-blind spots (CRITICAL)

| # | Что упускалось | Закрыто в |
|---|---|---|
| 1 | 518 строк `mock.patch("src.X")` в тестах — codemod на импортах их НЕ ловит | §B.4.2 Pass 2 + R31 |
| 2 | 7 dynamic imports `importlib.import_module("src.X")` в src/ + 10 в тестах | §B.4.2 Pass 2 + R32 |
| 3 | 2 subprocess argv strings (`"src.training.run_training"`, `"src.pipeline.worker"`) | §B.4.2 Pass 2 + R33 |
| 4 | `sys.modules.get("src.X")` в model_retriever | §B.4.2 Pass 2 + R32 |
| 5 | `src/main.py` entry point string `"src.main:app"` | §B.4.2 Pass 2 (паттерн `src\.main`) + R6 |

**Почему критично:** если бы агент-исполнитель сделал только обычный
libcst-codemod на импортах, то после Phase B:
- Все 518 mock-тестов начали бы падать с `ModuleNotFoundError: src.X`
- Trainer subprocess не запустился бы (argv ссылается на несуществующий модуль)
- Worker не загрузился бы
- Provider factory сломалась бы (importlib.import_module)

Теперь явно прописан Pass 2.

### 19.2 External configs not yet covered

| # | Файл | Что в нём | Закрыто в |
|---|---|---|---|
| 6 | [.claude/launch.json:18](.claude/launch.json:18) | `python -m src.main serve` для web-backend dev launcher | §B.4.3 + R34 |
| 7 | [examples/quickstart-qlora-sft/pipeline_config.yaml:14](examples/quickstart-qlora-sft/pipeline_config.yaml:14) | туториал с `python -m src.pipeline.orchestrator` | §B.4.3 + R35 |
| 8 | [examples/quickstart-qlora-sft/README.md:60](examples/quickstart-qlora-sft/README.md:60) | то же | §B.4.3 + R35 |
| 9 | [docker/inference/build_and_push.sh:184](docker/inference/build_and_push.sh:184) | путь к `src/inference/__about__.py` для VERSION | §B.4.3 + R36 |
| 10 | [.dockerignore:71](.dockerignore:71) | `src/tests` exclude | §B.4.3 + R37 |
| 11 | [.pre-commit-config.yaml](.pre-commit-config.yaml) | hard-coded `^src/(config|data|pipeline|providers|reports|training|utils)/` (line 67) и три `^src/` (85, 99, 114) и exclude `src/tests/` (line 125) | §B.4.3 + R38 |
| 12 | [pytest.ini](pytest.ini) (отдельный файл!) | testpaths | §B.4.3 |
| 13 | [pyrightconfig.json](pyrightconfig.json) | include patterns | §B.4.3 |

**Не нашлось** (проверено grep-ом 2026-05-03):
- `.github/workflows/` — CI отсутствует, обновлять нечего → R39 (low).
- `.dvc/`, `dvc.yaml` — нет, хотя `dvc` в deps → R40 неактуален.
- pickle/dill в production code — нет (R2 confirmed).

### 19.3 Phase A неточности

| # | Что было нечётко | Закрыто в |
|---|---|---|
| 14 | A.5 (IMLflowManager Protocol) — не указано как составить **точный список методов**, которые pipeline вызывает на MLflowManager (если пропустить какой-то — runtime AttributeError) | R41 |
| 15 | A.7 (container.py move) — может содержать импорты pipeline/api (тогда A.7 не закроет scc-160 полностью) | R42 |

**Pre-commit обязательная проверка перед началом каждого Phase A PR:**
```bash
# Для A.5 — каталогизировать вызовы pipeline → trainer.MLflowManager:
grep -rn "mlflow_manager\.\|mlflow_attempt\.manager\.\w*\.\w" src/pipeline/

# Для A.7 — проверить что container.py не тащит pipeline:
grep -n "from src.pipeline\|from src.api" src/utils/container.py
```

### 19.4 Pre-existing skipped tests (baseline)

11 тестов имеют `@pytest.mark.skipif(...)`. Они уже skipped и до Phase B,
останутся skipped после. **Не регрессия.** Каталог:

| Файл | Линия | Условие skip |
|---|---:|---|
| [src/tests/unit/training/test_runner_event_callback_wiring.py](src/tests/unit/training/test_runner_event_callback_wiring.py) | 240 | env-зависимый |
| [src/tests/unit/infrastructure/mlflow/test_entrypoint_allowed_hosts.py](src/tests/unit/infrastructure/mlflow/test_entrypoint_allowed_hosts.py) | 66 | `entrypoint.mlflow.sh not found` |
| [src/tests/integration/test_eval_runner_offline.py](src/tests/integration/test_eval_runner_offline.py) | 270 | env-зависимый |
| [src/tests/integration/test_cerebras_api_live.py](src/tests/integration/test_cerebras_api_live.py) | 82, 83 | API key не задан |
| (+ ~6 других) | | env-зависимые |

**Drift guard:** в Phase B.* DoD добавить — `count(@pytest.mark.skipif)`
после Phase B == count до Phase B (или меньше).

### 19.5 Что осталось «open» (но не блокирует GO)

- **CI (`.github/workflows/`)** — не существует. Если когда-то появится,
  workflow будет под packages-структуру.
- **pyright IDE настройка** — extraPaths за каждый workspace member.
  Скорее всего pyright workspace-aware из коробки, но если не — добавим
  conf в Phase C.
- **uv export `--no-emit-package`** — синтаксис может варьироваться
  между uv-версиями. Закрепить версию uv в Phase B.
- **prek (faster pre-commit)** — Apache Airflow использует. Out of scope,
  возможно в будущем.
- **`packages/_test_support/` workspace member** — создавать только если
  conftest.py фикстуры реально дублируются между пакетами. Если нет —
  не создавать (YAGNI).

### 19.6 Вердикт аудита

**План закрывает все известные дыры.** Все 15 находок интегрированы:
- 5 codemod blind spots → §B.4.2 + R31-R33
- 8 config-файлов → §B.4.3 + R34-R38
- 2 Phase A нечёткости → R41-R42

Риск-pool: 30 → **45 рисков** (15 новых R31-R45).
Все с Mitigation/Decision, ни одного «open».

**Готов к GO.**

---

## 19. Architecture Self-Review (5 итераций, 25 вопросов)

> Структурный аудит **архитектуры** (не плана миграции — это §18).
> 5 итераций под разными углами: SRP/coupling/lifecycle/testing/future.
> На каждый вопрос — deep-think ответ + verdict + action item.
>
> **Цель:** убедиться, что новая архитектура не имеет архитектурных
> дыр, не просто закрытие миграционных рисков.

### 19.1 Iteration 1 — Boundaries & SRP

#### Q1.1 — `MemoryManager` в `shared/utils` — GPU-specific код в shared?

**Контекст:** [src/utils/memory_manager.py](src/utils/memory_manager.py)
содержит `GPUInfo`, `GPUPreset`, `MemoryStats`, `MemoryManager` —
**GPU memory introspection**. Mac не имеет GPU.

**Кто реально импортит** (grep):
- `src/training/orchestrator/strategy_orchestrator.py` ✓
- `src/training/orchestrator/phase_executor/training_runner.py` ✓
- ВСЕ tests memory_manager.py
- **Никто из api/cli/pipeline/providers/data/evaluation/reports** не использует.

**Deep-think:** shared = «общее, нужное обеим сторонам». MemoryManager —
specifically pod-side. Если его положить в `ryotenkai-shared`, то Mac
тащит GPU-specific код вообще без причины — нарушение SRP shared'а.

**Verdict:** 🚨 **РЕАЛЬНАЯ ДЫРА.** MemoryManager должен быть в
`ryotenkai-pod.trainer` рядом с `container.py`, не в shared.

**Action item:** **расширить Phase A.7** — переезжает не только
`utils/container.py` → `training/container.py`, но и
`utils/memory_manager.py` → `training/memory_manager.py`. Тесты
переехать туда же.

#### Q1.2 — `infrastructure` (MLflow gateway) внутри shared vs отдельный пакет?

**Контекст:** `src/infrastructure/` всего 462 LOC (mlflow gateway,
uri_resolver, environment). MLflow используется и pod (trainer пишет
metrics) и control (api читает runs).

**Deep-think:** выделять в отдельный 6-й пакет ради 462 LOC —
overkill (YAGNI). shared/infrastructure/mlflow/ — естественное место
(абстракция инфраструктурных сервисов). Если завтра добавим
Wandb/Neptune — расширим shared/infrastructure/{wandb,neptune}/.
Когда вырастет до 5k+ LOC — выделим в `ryotenkai-observability`.

**Verdict:** ✅ OK. **Решение:** оставить в shared.

#### Q1.3 — Config schemas (training-aware) в shared — правильно?

**Контекст:** `src/config/` содержит `PipelineConfig`, `ModelConfig`,
`TrainingOnlyConfig`, `QLoRAConfig` — schemas сильно training-aware.

**Deep-think:** config — pure pydantic schema, без runtime импликаций
(не импортит torch). Mac валидирует юзерский ввод (CLI/API парсит и
валидирует config files), Pod парсит тот же config (для запуска
тренировки). **Both need same schema** — иначе drift. Невозможно
схему положить в pod-only (Mac не сможет валидировать).

**Verdict:** ✅ OK. **Решение:** оставить в shared.

#### Q1.4 — `reward_plugins` API в trainer vs community?

**Контекст:** [src/training/reward_plugins/base.py](src/training/reward_plugins/base.py)
определяет `RewardPlugin` ABC. Импортит `from datasets import Dataset`
(только TYPE_CHECKING) и `from src.utils.plugin_base import BasePlugin`.
Используется trainer (loader) + третьесторонними `community/reward/*`
плагинами.

**Deep-think:** если переместить в community — community будет знать
про training-domain (`Dataset`, реward signature). Это нарушение SRP:
community = generic plugin loader (catalog/manifest/sync), не
training-domain. Также community может стать light-weight (без torch
deps в runtime). Reward же — это специализация trainer'а.

**Verdict:** ✅ OK. **Решение:** reward_plugins API остаётся в
trainer. Переезжает в `ryotenkai-pod.trainer.reward_plugins`.
Третьесторонние плагины импортят оттуда — это часть trainer's public
extension API.

#### Q1.5 — `control` раздут (~52k LOC) — нарушение SRP?

**Контекст:** control = pipeline (19k) + api (8k) + cli (5k) +
reports (5k) + data (2k) + workspace (2k) + evaluation (1.4k) +
cli_state (200). Все Mac-side.

**Deep-think:** все эти подсистемы тесно связаны:
- pipeline = orchestrator (run lifecycle)
- api = HTTP интерфейс к pipeline
- cli = CLI интерфейс к pipeline
- reports/evaluation = post-run обработка результатов pipeline
- data/workspace/cli_state = supporting state

Все они share `PipelineConfig`, `RunContext`, `state.PipelineStateStore`.
Разделение преждевременно (YAGNI). Если когда-то api станет отдельным
сервисом или мы захотим headless mode (без CLI) — выделим. Сейчас все
живут одной кодовой базой control plane'а.

**Альтернатива:** разделить на ryotenkai-pipeline + ryotenkai-api +
ryotenkai-cli. **Минусы:** 3 pyproject.toml, transitive deps между
ними, dependency hell при breaking change в pipeline.

**Verdict:** ✅ OK. **Решение:** оставить control как single
Mac-side package. **Re-evaluate** через 6 мес. если control вырастет
до >70k LOC.

### 19.2 Iteration 2 — Dependencies & Coupling

#### Q2.1 — shared тащит httpx через JobClient — OK?

**Контекст:** Phase A.2 переносит `JobClient` в `shared/utils/clients/`.
shared deps += httpx. Это значит httpx становится transitive dep
ВСЕХ workspace members.

**Реальные потребители httpx (grep сегодня):**
- `src/pipeline/stages/model_evaluator.py` ✓ (control)
- `src/training/callbacks/runner_event_callback.py` ✓ (pod)
- `src/providers/runpod/runtime/lifecycle_client.py` ✓ (providers)

httpx уже всюду используется. Добавление в shared — лишь формализация.

**Verdict:** ✅ OK. **Решение:** httpx в shared deps. Все пакеты
получают консистентную версию. Размер ~1.5 MB (приемлемо).

#### Q2.2 — control plane → MLflow read path: gateway или прямой client?

**Контекст:** `src/api/routers/integrations.py` (control) читает
MLflow runs через `mlflow.tracking.MlflowClient` напрямую. shared
имеет gateway (для URI resolution и env), но не reader.

**Deep-think:** control — read-only consumer MLflow. Использовать
mlflow library напрямую — proper. Дублировать в shared/gateway —
нет cohesion (gateway — про environment, не про reads).

**Verdict:** ✅ OK. **Решение:** не блокирует. Документ как future
improvement если будет много места MLflow-reads → shared/infrastructure/mlflow/reader.py.

#### Q2.3 — providers совмещает «deploy» + «runtime client» (vLLM) — cohesion violation?

**Контекст:** providers содержит:
- `runpod/lifecycle/`, `runpod/training/` — deploy logic к RunPod
- `single_node/training/`, `single_node/runtime/` — local docker deploy
- `inference/vllm/engine.py` (после Phase A.3) — runtime client к vLLM

Это два разных уровня: deploy (контролирует жизненный цикл pod) и
runtime client (отправляет inference requests).

**Deep-think:** оба — части одной интеграции "RuntimeBackend". Когда
provider deploys pod, тот же provider потом разговаривает с pod.
Cohesion: high-level — «интеграция с runtime». Альтернатива (выделить
inference clients в отдельный `ryotenkai-inference-clients`) —
overkill для 200 LOC vLLM. YAGNI.

**Verdict:** ✅ OK. **Решение:** оставить как single providers pkg.

#### Q2.4 — trainer пишет в runner через HTTP. Кто owns JobClient в trainer?

**Контекст:** trainer (на pod) пишет события в локальный runner (тоже
на pod) через `httpx.post("http://localhost:8080/api/v1/internal/events")`.
Loopback HTTP внутри pod. Сейчас trainer использует httpx напрямую,
не JobClient.

**Deep-think:** loopback всегда отличается от внешнего HTTP (нет
auth, нет retry на сетевые ошибки, всегда localhost). JobClient в
shared — для control → runner (внешний HTTP). Trainer → runner
loopback — другая семантика.

**Verdict:** ✅ OK. **Решение:** trainer продолжает использовать
httpx напрямую для loopback. JobClient — для внешнего HTTP. Можно
extract `LoopbackEventClient` в shared/utils/clients позже — не
блокирует.

#### Q2.5 — evaluation orchestration → vLLM через providers. Граф разрешает?

**Контекст:** evaluation (control) запускает model inference через
vLLM (генерация ответов на eval prompts). После Phase A.3 vLLM в
providers. control → providers разрешено importlinter контрактом.

**Deep-think:** граф зависимостей: control → providers → shared.
evaluation внутри control импортит `ryotenkai_providers.inference.vllm`.
Чисто.

**Verdict:** ✅ OK. **Решение:** подтверждено.

### 19.3 Iteration 3 — Lifecycle & Deployment & Versioning

#### Q3.1 — CodeSyncer rsync vs pip install после Phase B?

**Контекст:** CodeSyncer сейчас rsync-ит src/ на pod. Pod-image
содержит Python + deps. После Phase B src/ → packages/. Что
устанавливается в pod, как обновляется?

**Deep-think:** есть два варианта:
1. **Rsync + editable install** (`pip install -e ./packages/X`) —
   pod-image при build делает editable install из placeholder paths.
   CodeSyncer rsync-ит `packages/{shared,community,pod}/` в pod, в те
   же paths. Editable install подхватывает новый код без reinstall.
   **Плюсы:** быстро (как сейчас), no install overhead.
2. **Wheel build + force-reinstall** — control plane строит wheels
   локально, scp-ит, pod делает `pip install --force-reinstall *.whl`.
   **Плюсы:** воспроизводимо. **Минусы:** медленно, build overhead.

**Verdict:** 🚨 **ТРЕБУЕТ УТОЧНЕНИЯ В ПЛАНЕ.** План §3.5
говорит «pip install ./packages/...» в Docker build, но не уточняет
runtime update flow (rsync vs wheel).

**Action item:** **дополнить §6.6 Docker** — pod-image использует
editable install, CodeSyncer продолжает rsync (только paths меняются).
Drift guard: `pip show ryotenkai-pod` после rsync должен показать
актуальный path.

#### Q3.2 — Версионирование контракта control ↔ pod (HTTP wire)?

**Контекст:** runner exposes `/api/v1/...`. Если control bumps до
`/v2/` — pod на старом image не понимает.

**Deep-think:** это адресовано **transport unification plan** (Phase 1
problem+json + version negotiation). Не наш scope.

**Verdict:** ✅ Out of scope (handled elsewhere). **Решение:**
ссылка на transport plan в §11.2.

#### Q3.3 — Pod-image baked Docker. Drift detection?

**Контекст:** Pod скачивает Docker image при запуске (RUNTIME_IMAGE
из shared/constants). Stale image → stale code.

**Deep-think:** runner exposes `/version` endpoint. Control plane
после deploy сравнивает с expected. Это уже работает (видел в
`src/runner/main.py`). Packagization не меняет.

**Verdict:** ✅ Already handled.

#### Q3.4 — Несинхронный bump одного пакета?

**Контекст:** план говорит «синхронный bump всех». Но если bug fix
только в pod, зачем bump shared/community/control?

**Deep-think:** uv workspace **не enforce-ит** synchronous version.
Пакеты могут расходиться. НО: если pod-1.0.1 зависит от shared-1.0.0
и control-1.0.0 зависит от shared-1.0.1 — конфликт shared single
version в venv. Workspace-deps указывают `{ workspace = true }` — это
эквивалент `==X.Y.Z` где X.Y.Z = current workspace member version.

Поэтому: bump одного пакета = bump его patch, остальные могут
оставаться. shared bump → все workspace members на одну shared
version.

**Verdict:** ✅ OK. **Решение:** документировать в §3.4 — bump
policy: leaf-package bump (pod/providers/control) не требует bump
других. shared/community bump — может требовать bump зависящих (если
breaking change).

#### Q3.5 — pip install vs rsync — производительность?

**Контекст:** Rsync incremental, ~ms. Pip install — секунды.

**Deep-think:** связано с Q3.1. С editable install rsync continues
to be fast. Pip install только в build time pod-image.

**Verdict:** ✅ OK с editable install. См. Q3.1 action item.

### 19.4 Iteration 4 — Testing & Observability

#### Q4.1 — Reverse sentinel: `test_no_control_imports_in_pod`?

**Контекст:** План имеет sentinel `test_no_pod_imports_in_control`,
но не reverse.

**Deep-think:** importlinter contract «pod depends only on
shared+community» — defends static analysis. Sentinel test —
runtime/AST defence in depth. Симметричный sentinel желателен.

**Verdict:** 🚨 **MISSING.** **Action item:** добавить в §7.1 /
§7.2 второй sentinel `test_no_control_imports_in_pod` + аналогичные
для providers (`test_no_providers_imports_in_pod`,
`test_no_pod_imports_in_providers`).

#### Q4.2 — Coverage baseline после Phase B — кто измерит?

**Контекст:** План назначает per-package floors (shared 90, community
85, pod 78, providers 70, control 75) — **гипотетически**. Реальный
baseline нужно измерить.

**Verdict:** 🚨 **ТРЕБУЕТ УТОЧНЕНИЯ.** **Action item:** добавить
в §6.7 — Phase C explicit step «Measure baseline coverage per-package,
set floor = baseline − 2». Не назначать floors заранее.

#### Q4.3 — Integration тесты с pod-mocks — где?

**Контекст:** integration тесты тестируют end-to-end (Mac+pod).
Используют mock pod (httpx mocks). Сейчас живут в `src/tests/integration/`.
После Phase B в `packages/control/tests/integration/`.

**Deep-think:** mocks pod-side endpoints — ответственность control
(он тестирует свой path). Pod-side integration (как pod обрабатывает
real запросы) — ответственность pod's tests/integration. Cross-pkg
integration — own by control.

**Verdict:** ✅ OK. **Решение:** документировано.

#### Q4.4 — GPU-тесты на Mac dev venv — skipif?

**Контекст:** Mac не имеет GPU. Trainer тесты с GPU — skipif?

**Deep-think:** уже есть marker `requires_gpu` в pyproject.toml.
`@pytest.mark.requires_gpu` + `pytest -m "not requires_gpu"` на Mac.
Не меняется после Phase B.

**Verdict:** ✅ OK.

#### Q4.5 — `patch("ryotenkai_X.Y")` после codemod — может неправильно патчить?

**Контекст:** Семантика mock.patch: `patch("module.path.where.imported")`
≠ `patch("module.path.original")`. Codemod B.4.2 переписывает строки
literally — может не сохранить правильный patch path.

**Пример:**
```python
# src/foo/bar.py
from src.utils.helper import do_thing
...

# test
patch("src.foo.bar.do_thing", ...)  # patches bar's local binding
patch("src.utils.helper.do_thing", ...)  # patches global helper
```

После B.4.2:
```python
patch("ryotenkai_control.foo.bar.do_thing", ...)  # OK
patch("ryotenkai_shared.utils.helper.do_thing", ...)  # OK
```

Оба правильны (literal rewrite сохраняет семантику). НО если codemod
ошибочно превратит первое в `ryotenkai_shared.utils.helper.do_thing` —
mock patches global вместо local → test pass false positively.

**Verdict:** 🚨 **ВОЗМОЖНАЯ ДЫРА.** **Action item:** добавить риск
**R46** + Phase B verification step:
```bash
# После B.4.2 — каждый @patch должен быть проверен:
pytest --collect-only -v 2>&1 | grep -E "PASSED|FAILED|ERROR"
# Затем: pytest -x — все mocks должны работать как раньше.
```
Если падает — ручной разбор тех мест где patch path неоднозначен.

### 19.5 Iteration 5 — Future-proofing & Anti-patterns

#### Q5.1 — Если в будущем PyPI publishing — flat → namespace breaking?

**Контекст:** flat naming (`ryotenkai_pod`) → если перейдём на
namespace (`ryotenkai.pod`) — все импорты ломаются.

**Deep-think:** мы НЕ публикуем (план §1.3). Если когда-то решим:
- Через aliasing (`from ryotenkai_pod import *` в `ryotenkai/pod/__init__.py`)
- Или deprecation period (warn → break)

**Verdict:** ✅ Acceptable trade-off.

#### Q5.2 — API versioning (`/v1/` → `/v2/`) — структура папок?

**Контекст:** routes сейчас flat (`ryotenkai_control.api.routers/`)
с префиксом `/api/v1/` в FastAPI app. v2 — нужны `routers_v1/`,
`routers_v2/`?

**Deep-think:** не блокирует packagization. Когда будет v2 —
рефакторинг внутри control. Lift из routers/X.py → routers/v2/X.py.

**Verdict:** ✅ Out of scope.

#### Q5.3 — pyproject.toml duplication — uv inherit?

**Контекст:** каждый member имеет [project], [build-system]. Tools
([tool.ruff], [tool.mypy], [tool.black]) могут жить только в root.

**Deep-think:** ruff/mypy/black ищут конфиг по файловой иерархии вверх
от анализируемого файла. Root pyproject.toml даёт настройки всем
members автоматически. Per-package overrides только если нужны.

**Verdict:** ✅ OK. **Решение:** root pyproject.toml содержит
[tool.ruff/black/mypy] глобально, members — только [project] +
[build-system].

#### Q5.4 — Structured logging (JSON) для production?

**Контекст:** loguru pretty terminal output. Production observability
требует JSON.

**Deep-think:** loguru поддерживает sinks (файл, JSON, stderr).
Production sink настраивается через ENV. Не блокирует
packagization — отдельная задача.

**Verdict:** ✅ Out of scope.

#### Q5.5 — Secrets handling — dedicated package?

**Контекст:** Secrets через config.Secrets (Pydantic) + load_secrets
из env/file. Не Vault/AWS Secrets.

**Deep-think:** для текущего scope (single Mac dev + private RunPod)
достаточно env vars. Vault/AWS Secrets — когда будет multi-tenant.
Тогда — выделим `ryotenkai-secrets` package.

**Verdict:** ✅ OK. Future expansion path open.

### 19.6 Verdict архитектурного аудита

**Из 25 вопросов:**

| Категория | Count | Что |
|---|---:|---|
| ✅ Архитектура подтверждена | **20** | Q1.2-Q1.5, Q2.1-Q2.5, Q3.2-Q3.5, Q4.1, Q4.3-Q4.4, Q5.1-Q5.5 |
| 🚨 Найдены реальные дыры | **2** | **Q1.1** (MemoryManager → trainer), **Q4.1** (reverse sentinel) |
| 🚨 Требуют уточнения плана | **3** | **Q3.1** (editable install), **Q4.2** (coverage calibration), **Q4.5** (mock path verification) |

**Action items (5 шт):**

1. **Q1.1 → Phase A.7 расширить:** `utils/memory_manager.py` →
   `training/memory_manager.py` (вместе с container.py).
2. **Q4.1 → §7.1/§7.2 добавить:** sentinel `test_no_control_imports_in_pod`
   + аналогичные для providers границ.
3. **Q3.1 → §6.6 уточнить:** pod-image использует **editable install**
   (`pip install -e ./packages/{shared,community,pod}`); CodeSyncer
   продолжает rsync — editable install подхватывает обновления.
4. **Q4.2 → §6.7 уточнить:** в Phase C explicit «измерить baseline,
   set floor = baseline − 2», не назначать заранее.
5. **Q4.5 → новый риск R46:** mock.patch path verification после
   B.4.2 — `pytest --collect-only` ловит broken mocks.

**Архитектура — корректна.** 5 точечных уточнений в плане + 1 риск
добавлены ниже.

---

## 20. Handover для следующего агента

> Этот раздел — для агента, который будет ИСПОЛНЯТЬ план после моего
> ухода из контекста. Прочитай **сначала** §0 TL;DR + §3.1 layout +
> §13 open questions (всё RESOLVED) + §18 audit (15 закрытых дыр) +
> §19 architecture self-review (5 action items), потом этот §20 —
> здесь практические команды и gotchas.

### 18.1 Что уже сделано до тебя

**Контекст ветки `claude/dazzling-rosalind-b482fa`:**
- `e6e6f0f` — refactor monitor cleanup (Phase 0)
- `5293bbf` — ship full src/ (CodeSyncer гарантия)
- `93a7185` — план doc для transport unification (NB: тот план
  СУПЕРСИДЕДЕД этим — см. §11.2)
- `fc36a8e` — log filename literals consolidation
- `fced7f0` — pull trainer.stdio.log + runner.log via PodLayout
- `a5f3d03` — pull-only ground-truth + per-run PodLayout

**Pre-existing baseline (must preserve):**
- 164+ unit tests на этой ветке должны оставаться зелёными после каждого PR.
- Coverage gate `fail_under = 83`.
- 9 уже-падающих тестов (pre-existing, не твои) — можно не трогать,
  но не делать ХУЖЕ.

### 18.2 Где что лежит (key files reference)

**Граничные нарушения (Phase A targets):**
- [src/pipeline/cancellation.py](src/pipeline/cancellation.py) → переехать в shared (Phase A.1)
- [src/api/clients/job_client.py](src/api/clients/job_client.py) → shared/utils/clients (Phase A.2)
- [src/api/services/control_plane_heartbeat.py](src/api/services/control_plane_heartbeat.py) → pipeline/heartbeat (Phase A.2)
- [src/api/services/tunnel_service.py](src/api/services/tunnel_service.py) → shared/utils/clients (Phase A.2)
- [src/pipeline/inference/vllm.py](src/pipeline/inference/vllm.py) → providers/inference/vllm (Phase A.3)
- [src/runner/__about__.py](src/runner/__about__.py) RUNTIME_IMAGE → shared/constants/runtime.py (Phase A.4)
- [src/training/managers/mlflow_manager](src/training/managers/mlflow_manager) → extract Protocol (Phase A.5)
- [src/utils/config.py](src/utils/config.py) facade — DELETE + codemod (Phase A.6)
- [src/utils/container.py](src/utils/container.py) → training/container.py (Phase A.7, 718 LOC)

**Импортёры, требующие rewrite в каждом Phase A PR (вход):**
- pipeline.cancellation импортёры: `grep -rn "from src.pipeline.cancellation" src/`
- api.clients/services импортёры: `grep -rn "from src.api.clients\|from src.api.services" src/`
- pipeline.inference.vllm импортёр: `src/providers/single_node/inference/provider.py:23`
- runner.__about__ импортёр: `src/pipeline/stages/managers/deployment/dependency_installer.py:25`
- training.mlflow_manager импортёр: `src/pipeline/mlflow_attempt/manager.py:29`
- utils.config импортёры: `grep -rln "from src.utils.config" src/` (10+ файлов)
- utils.container импортёры: см. §A.7 — список фиксирован.

**Critical config files (правки в Phase B):**
- [pyproject.toml](pyproject.toml) — root, become workspace declaration
- [Makefile](Makefile) — `src/` → `packages/*/src/`, `src.api.openapi_dump` → `ryotenkai_control.api.openapi_dump`
- [pytest.ini](pytest.ini) и [pyrightconfig.json](pyrightconfig.json) — paths
- [.pre-commit-config.yaml](.pre-commit-config.yaml) — pre-commit paths
- [docker/training/Dockerfile.runtime](docker/training/Dockerfile.runtime) — multi-stage build
- [docker/training/entrypoint.sh](docker/training/entrypoint.sh) — uvicorn line (если ссылается; check first)
- [src/pipeline/stages/managers/deployment/runner_launcher.py:181](src/pipeline/stages/managers/deployment/runner_launcher.py:181) — uvicorn launch string
- [src/pipeline/stages/managers/deployment/code_syncer.py](src/pipeline/stages/managers/deployment/code_syncer.py) — REQUIRED_PACKAGES update
- [src/main.py](src/main.py) — entry point shim, удалить после Phase B (становится `ryotenkai_control/main.py`)

**Plugin files в корневом community/ (требуют codemod в Phase B):**
- ~25 .py файлов в [community/](community/) импортят `src.community`,
  `src.training`, `src.evaluation`, `src.data`, `src.reports`, `src.utils`
- Точный список: `grep -rln "^from src\.\|^import src\." community/`

### 18.3 Quick commands

```bash
# Worktree info
pwd  # должен быть /Users/daniil/MyProjects/RyotenkAI/.claude/worktrees/dazzling-rosalind-b482fa
git branch --show-current  # claude/dazzling-rosalind-b482fa

# Tests
pytest src/tests/unit -v           # быстрая проверка после каждого PR Phase A
make test                          # полный
make lint                          # ruff + mypy
make openapi                       # перегенерация openapi.json (gold check после Phase B)

# Cycle / violation проверки (drift guards для Phase A)
grep -rn "^from src\.pipeline" src/api src/providers src/runner src/training
grep -rn "^from src\.api"      src/pipeline src/providers src/runner src/training
grep -rn "^from src\.training" src/pipeline src/providers src/runner src/api
grep -rn "^from src\.runner"   src/pipeline src/providers src/training src/api

# uv workspace команды (после Phase B)
uv sync                            # установить ВСЕ packages в один .venv
uv sync --package ryotenkai-pod    # только pod
uv run --package ryotenkai-pod pytest packages/pod/tests/unit
uv run --package ryotenkai-control pytest packages/control/tests
uv export --package ryotenkai-pod --frozen --no-dev > /tmp/pod-reqs.txt  # для Docker

# Импорт-аудит после Phase B (должно вернуть пусто)
find . -path ./packages -prune -o -name '*.py' -print \
  | xargs grep -l '^from src\.\|^import src\.'

# Sentinel sanity (после Phase B)
python -c "import ryotenkai_shared, ryotenkai_community, ryotenkai_pod, ryotenkai_providers, ryotenkai_control; print('OK')"
ryotenkai --help
ryotenkai-trainer-run --help
python -m uvicorn ryotenkai_pod.runner.main:app --help

# Docker
docker build -f docker/training/Dockerfile.runtime -t ryotenkai-pod:test .
docker images | grep ryotenkai-pod  # сравнить размер с предыдущим baseline
```

### 18.4 Workflow rules (от пользователя — не переспрашивать)

1. **Перед каждой крупной задачей** — deep-think анализ связанного кода.
2. **Helixir memory** — `mcp__helixir__search_memory` несколько итераций перед началом.
3. **Plan first, action second** — для нетривиального шага составить
   мини-план в TODO.
4. **3-iteration risk audit** для любого плана, прежде чем выполнять.
5. **3-attempts policy** — если тест падает после фикса:
   - Attempt 1: minimal fix (<10 строк).
   - Attempt 2: глубже разобраться (читать источник, что именно сломалось).
   - Attempt 3: STOP, deep-think. Если не находишь — flag skip с TODO + issue,
     ПРОДОЛЖАЙ. Не игнорируй красный тест без флага.
6. **Активно использовать TodoWrite** — обновлять статус каждой подзадачи.
7. **Очень активно использовать MCP** — repowise, helixir, tavily, context7.
8. **7 типов тестов** для каждого нового/изменённого юнита: positive,
   negative, boundary, invariants, dep-error sim, regression, combinatorial
   edge. См. §7.
9. **No backward compatibility** — старый `src/` удалён в конце Phase B,
   никаких permanent shims.
10. **Frontend деприоритизирован** — только Makefile target `make openapi`,
    больше web/ не трогать.

### 18.5 Gotchas / "об что споткнётся"

1. **`git mv` обязателен** для всех переездов файлов (preserve history).
   `mv` + `git add -A` ломает git blame. Если случайно сделал `mv` —
   `git mv` сверху НЕ исправит, нужно reset и переделать.
2. **Тесты на текущей ветке упадут после Phase A.7 (container.py move)**
   если не обновить путь в `src/tests/unit/utils/test_container_*.py`.
   Перенести в `src/tests/unit/training/` в том же PR.
3. **`utils/config.py` facade удаление (Phase A.6)** ломает 10+ файлов
   pipeline. Codemod обязателен в одном PR.
4. **`mlflow_attempt` Protocol (Phase A.5)** — pipeline сейчас вызывает
   methods на concrete `MLflowManager`. После Protocol-extract проверить
   все вызовы `manager.X(...)` — может оказаться, что Protocol неполный.
5. **Phase B atomic merge** — если хоть один коммит из 5 не попал в
   merge, репо broken (циклы импортов или missing imports). PR должен
   landить ВСЕ 5 commits либо ни одного.
6. **uv workspace + `[build-system]`** — каждый member нуждается в своём
   `[build-system]` (`requires = ["setuptools>=68", "wheel"]`,
   `build-backend = "setuptools.build_meta"`) **И** в `[tool.setuptools.packages.find]`
   с `where = ["src"]` (раз мы используем src-layout per package).
7. **`from src.X` остаточные импорты** легко пропустить в:
   - docstrings (не критично, но смотрится плохо)
   - error messages (`raise ImportError("…src.X…")`)
   - test fixtures, конфигах logging (`"src.X.logger"`)
   - `__module__`-зависимый код (например, dynamic class registries)
8. **`/openapi.json` baseline** — сделать `make openapi` ДО Phase B,
   сохранить копию, после Phase B сравнить `diff` — должно быть identical.
9. **Pre-existing 9 failing tests** — проверь `git log` на предмет skip
   markers, не паникуй если они продолжают падать.
10. **Trainer запускается subprocess'ом из runner** — runner НЕ
    импортит trainer Python-код. Если случайно появится
    `from ryotenkai_pod.trainer import …` в `ryotenkai_pod.runner.*` —
    importlinter contract упадёт (см. §6.1).
11. **Coverage merge** — pytest-cov создаёт separate `.coverage.X` файлы
    при per-package runs. Используй `coverage combine` перед `coverage report`.
12. **Codemod через libcst** — установи `pip install libcst` сначала.
    Простой regex-codemod НЕ годится: ломает строки в docstrings,
    typing.cast Cast() аргументы, форматирование. Готовый pattern:
    ```python
    from libcst import codemod
    class RewriteImports(codemod.VisitorBasedCodemodCommand):
        def leave_ImportFrom(self, original, updated):
            module = updated.module
            if module and module.value == "src.pipeline.cancellation":
                return updated.with_changes(module=cst.Attribute(
                    value=cst.Name("ryotenkai_shared"),
                    attr=cst.Name("utils.cancellation")  # упрощённо
                ))
            return updated
    ```

### 18.6 Что НЕ делать (явно)

- ❌ НЕ запускать Phase B пока Phase A не на 100% завершена и зелёная.
- ❌ НЕ менять wire-protocol (HTTP/SSH endpoints) — это другой план
  (transport unification, см. §11.2).
- ❌ НЕ трогать `web/` кроме `make openapi`-цели.
- ❌ НЕ обновлять зависимости (torch, transformers, etc.) — отдельная задача.
- ❌ НЕ публиковать на PyPI.
- ❌ НЕ делать `git rebase --interactive` Phase B-коммитов после ревью —
  ломает ревью-контекст. Лучше доп. fixup-commit.
- ❌ НЕ удалять старый `docs/plans/*.md` — они исторический record.
- ❌ НЕ менять CLI UX (команды и флаги `ryotenkai *`).
- ❌ НЕ вводить namespace package позже — пользователь явно выбрал flat.
- ❌ НЕ объединять providers с control — пользователь явно выбрал separate.

### 18.7 Если что-то идёт сильно не так

**Сценарий A: Phase B ломает import resolution.**
- Откати к предыдущему commit ветки (`git reset --hard <hash>`).
- Запусти `pytest src/tests/unit -x` — должно быть зелёно.
- Спроси пользователя через AskUserQuestion прежде чем повторять.

**Сценарий B: importlinter contract падает после Phase A.**
- НЕ ослаблять контракт, чтобы пройти. Найти violating import и refactor.
- Если refactor требует архитектурного решения — STOP, спросить
  пользователя.

**Сценарий C: Docker build ломается после Phase B.**
- Проверь `uv export --package ryotenkai-pod --frozen --no-dev` локально.
- Если `uv` не находит пакет — root pyproject.toml `[tool.uv.workspace]
  members` неверный.
- Если зависимости не разрешаются — `[tool.uv.sources]` не содержит
  workspace-record для пакета.

**Сценарий D: пользователь отменил план/изменил решения.**
- Зафиксировать новое решение в §13 этого документа.
- Если break, переместить план в `docs/plans/abandoned/` с краткой
  причиной в YAML frontmatter.

### 18.8 Полезные ссылки

- [uv workspaces docs](https://docs.astral.sh/uv/concepts/projects/workspaces/) — каноничная документация.
- [Apache Airflow FOSDEM 2026 talk](https://fosdem.org/2026/schedule/event/WE7NHM-modern-python-monorepo-apache-airflow/) — крупнейший прецедент.
- [Apache Airflow Modern Monorepo Part 2](https://medium.com/apache-airflow/modern-python-monorepo-for-apache-airflow-part-2-9b53e21bcefc) — практические уроки.
- [Jimmy Yeung: Journey migrating to uv workspace](https://dev.to/jimmyyeung/journey-migrating-to-uv-workspace-34a7) — Docker multi-stage build pattern (важно для §6.6).
- [src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) — Python Packaging User Guide.
- [importlinter docs](https://import-linter.readthedocs.io/) — для §6.1.
- [PEP 517/518/621/735](https://peps.python.org/) — стандарты pyproject.toml.

### 18.9 MCP-запросы для restoration of context

Если контекст истёк, начни с этого набора:

```
1. mcp__helixir__search_memory(query="монорепа uv workspace packagization")
2. mcp__repowise__get_overview()
3. mcp__repowise__get_risk(targets=["src/pipeline/orchestrator.py",
                                     "src/pipeline/stages/training_monitor.py",
                                     "src/providers/runpod/training/provider.py",
                                     "src/runner/main.py"])
4. Bash: grep -rn "^from src.pipeline" src/api src/providers src/runner src/training
5. Bash: cat docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md  (этот файл)
6. Bash: git log --oneline -20  (recent commits)
```

После этого у тебя будет 90% контекста для продолжения.

---

🛑 **КОНЕЦ ПЛАНА.** Жди явного «GO» от пользователя.
