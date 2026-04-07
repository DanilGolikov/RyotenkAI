<p align="center">
  <img src="../logo_final.png" alt="RyotenkAI" width="400">
</p>
<h1 align="center">RyotenkAI</h1>

<p align="center">
  Декларативный fine-tuning для LLM.<br>
  Вы описываете пайплайн в YAML и задаете датасеты, а RyotenkAI берет на себя валидацию, подготовку GPU, обучение, деплой инференса, оценку качества и отчетность.
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> |
  🇷🇺 Русский |
  <a href="README.ja.md">🇯🇵 日本語</a> |
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> |
  <a href="README.ko.md">🇰🇷 한국어</a> |
  <a href="README.es.md">🇪🇸 Español</a> |
  <a href="README.he.md">🇮🇱 עברית</a>
</p>

<p align="center">
  <a href="#быстрый-старт">Быстрый старт</a> ·
  <a href="#как-это-работает">Как это работает</a> ·
  <a href="#стратегии-обучения">Стратегии</a> ·
  <a href="#gpu-провайдеры">Провайдеры</a> ·
  <a href="#системы-плагинов">Плагины</a> ·
  <a href="#конфигурация">Конфигурация</a>
</p>

<p align="center">
  <a href="https://discord.gg/QqDM2DbY">
    <img src="https://img.shields.io/badge/Discord-Присоединиться%20к%20сообществу-5865F2?logo=discord&logoColor=white" alt="Присоединиться к Discord">
  </a>
  <br>
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97_Transformers-4.x-FFD21E" alt="Transformers">
  <img src="https://img.shields.io/badge/TRL-multi--strategy-blueviolet" alt="TRL">
  <img src="https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-8A2BE2" alt="PEFT">
  <br>
  <img src="https://img.shields.io/badge/vLLM-inference-00ADD8?logo=v&logoColor=white" alt="vLLM">
  <img src="https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/RunPod-cloud_GPU-673AB7" alt="RunPod">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT">
</p>

---

## Что такое RyotenkAI?

RyotenkAI - это декларативная control plane-система для fine-tuning LLM. Вы описываете workflow в YAML, подключаете датасеты, а RyotenkAI исполняет весь жизненный цикл: валидацию датасета, подготовку GPU, многофазное обучение, выгрузку модели, деплой инференса, оценку качества и отчетность по эксперименту в MLflow.

| Ручной workflow fine-tuning | RyotenkAI |
|---|---|
| Вручную проверять качество датасета | Валидация через плагины: формат, дубликаты, длина, разнообразие |
| Подключаться к GPU по SSH и запускать скрипты | Одна команда: подготовка GPU, деплой обучения, мониторинг |
| Ждать и надеяться, что все отработает | Мониторинг в реальном времени: метрики GPU, loss curves, детект OOM |
| Вручную забирать веса | Автоматически забирает adapters, merge LoRA и публикация в HF Hub |
| Отдельно поднимать inference server | Деплоит vLLM endpoint с health checks |
| Проверять ответы руками | Оценка через плагины: синтаксис, semantic match, LLM-as-judge |
| Вести заметки в документе | Трекинг экспериментов в MLflow + генерация Markdown-отчета |

---

## Как это работает

### Поток пайплайна

```text
YAML Config
    │
    ▼
┌─────────────────┐
│ Dataset Validator│  Проверка качества данных до обучения
│ (plugin system)  │  min_samples, diversity, format, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU Deployer    │  Подготовка вычислительных ресурсов (SSH или RunPod API)
│                  │  Деплой training container + кода
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Training Monitor  │  Мониторинг процесса, парсинг логов, детект OOM
│                  │  Метрики GPU, loss curves, health checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Retriever  │  Скачивание adapters / merged weights
│                  │  Опциональная публикация в HuggingFace Hub
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Inference Deployer│  Запуск vLLM server в Docker
│                  │  Health checks, OpenAI-compatible API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Evaluator  │  Запуск evaluation plugins на live endpoint
│ (plugin system)  │  syntax, semantic match, LLM judge, custom
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Report Generator  │  Сбор всех данных из MLflow
│ (plugin system)  │  Рендер Markdown-отчета по эксперименту
└─────────────────┘
```

### Выполнение обучения

```text
Pipeline (control plane)              GPU Provider (single_node / RunPod)
         │                                         │
    SSH / API ──────────────────────────► Docker container
         │                                   │
    rsync code ─────────────────────────►  /workspace/
         │                                   │
    start training ─────────────────────►  accelerate launch train.py
         │                                   │
    monitor ◄───────────── logs, markers, GPU metrics
         │                                   │
    retrieve artifacts ◄────── adapters, checkpoints, merged weights
```

### Цепочка стратегий обучения (внутри GPU container)

Многофазное обучение с автоматическим управлением состоянием, восстановлением после OOM и системой checkpoint'ов:

```text
run_training(config.yaml)
  │
  ├── MemoryManager.auto_configure()     Определяет класс GPU и выставляет пороги VRAM
  │     └── GPUPreset: margin, critical%, max_retries
  │
  ├── load_model_and_tokenizer()         Загружает базовую модель (без PEFT)
  │     └── MemoryManager: snapshot до/после, очистка CUDA cache
  │
  ├── DataBuffer.init_pipeline()         Инициализирует tracking состояния
  │     └── pipeline_state.json          Статусы фаз, пути к checkpoint'ам
  │     └── phase_0_sft/                 Директории с артефактами по фазам
  │     └── phase_1_dpo/
  │
  └── ChainRunner.run(strategies)        Исполняет цепочку фаз
        │
        │   Для каждой фазы (например CPT → SFT → DPO):
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  PhaseExecutor.execute(phase_idx, phase, model, buffer) │
  │                                                         │
  │  1. buffer.mark_phase_started(idx)                      │
  │     └── Атомарное сохранение состояния в pipeline_state.json
  │                                                         │
  │  2. StrategyFactory.create(phase.strategy_type)         │
  │     ├── SFTStrategy     (messages → instruction tuning) │
  │     ├── DPOStrategy     (chosen/rejected → alignment)   │
  │     ├── ORPOStrategy    (preference → odds ratio)       │
  │     ├── GRPOStrategy    (reward-guided RL)              │
  │     ├── SAPOStrategy    (self-aligned preference)       │
  │     └── CPTStrategy     (raw text → domain adaptation)  │
  │                                                         │
  │  3. dataset_loader.load_for_phase(phase)                │
  │     └── strategy.validate_dataset + prepare_dataset     │
  │                                                         │
  │  4. TrainerFactory.create_from_phase(...)               │
  │     ├── strategy.get_trainer_class() → TRL Trainer      │
  │     ├── Объединение hyperparams: global ∪ phase overrides
  │     ├── Создание PEFT config (LoRA / QLoRA / AdaLoRA)   │
  │     ├── Подключение callbacks (MLflow, GPU metrics)     │
  │     └── Обертка через MemoryManager.with_memory_protection
  │                                                         │
  │  5. trainer.train()                                     │
  │     └── MemoryManager.with_memory_protection            │
  │           ├── Мониторинг использования VRAM            │
  │           ├── При OOM → aggressive_cleanup + retry      │
  │           └── max_retries берется из GPU preset         │
  │                                                         │
  │  6. Save checkpoint-final                               │
  │     ├── buffer.mark_phase_completed(metrics)            │
  │     └── buffer.cleanup_old_checkpoints(keep_last=2)     │
  │                                                         │
  └─────────────────────┬───────────────────────────────────┘
                        │
                        ▼  модель передается в следующую фазу (в памяти)
                        │
               ┌────────┴────────┐
               │  Следующая фаза? │
               │  idx < total     │──── Нет ──► вернуть обученную модель
               └────────┬────────┘
                        │ Да
                        ▼
                 (повтор PhaseExecutor)
```

### DataBuffer - управление состоянием между фазами

```text
DataBuffer
  │
  ├── Pipeline State (pipeline_state.json)
  │     {
  │       "status": "running",
  │       "phases": [
  │         { "strategy": "sft", "status": "completed", "checkpoint": "phase_0_sft/checkpoint-final" },
  │         { "strategy": "dpo", "status": "running",   "checkpoint": null }
  │       ]
  │     }
  │
  ├── Phase Directories
  │     output/
  │     ├── phase_0_sft/
  │     │   ├── checkpoint-500/     (промежуточный, автоочистка)
  │     │   ├── checkpoint-1000/    (промежуточный, автоочистка)
  │     │   └── checkpoint-final/   (сохраняется - вход для следующей фазы)
  │     └── phase_1_dpo/
  │         └── checkpoint-final/
  │
  ├── Resume Logic
  │     При сбое/перезапуске:
  │       1. load_state() → найти первую незавершенную фазу
  │       2. get_model_path_for_phase(idx) → предыдущий checkpoint-final
  │       3. Загрузить PEFT adapters в базовую модель
  │       4. get_resume_checkpoint(idx) → mid-phase checkpoint (если есть)
  │       5. Продолжить обучение с места остановки
  │
  └── Cleanup
         cleanup_old_checkpoints(keep_last=2)
         Удаляет промежуточные checkpoint-N/ директории, сохраняет checkpoint-final
```

### MemoryManager - защита от GPU OOM

```text
MemoryManager.auto_configure()
  │
  ├── Определяет GPU: название, VRAM, compute capability
  │     ├── RTX 4060  (8GB)  → consumer_low   tier
  │     ├── RTX 4090  (24GB) → consumer_high  tier
  │     ├── A100      (80GB) → datacenter     tier
  │     └── Unknown          → safe fallback
  │
  ├── GPUPreset для каждого tier:
  │     margin_mb:    резерв по VRAM (512-4096 MB)
  │     critical_pct: порог срабатывания OOM recovery (85-95%)
  │     warning_pct:  порог warning-логов (70-85%)
  │     max_retries:  количество авто-повторов (1-3)
  │
  └── with_memory_protection(operation):
         ┌─────────────────────────────┐
         │  Attempt 1                  │
         │  ├── Проверить запас VRAM   │
         │  ├── Выполнить операцию     │
         │  └── Success → return       │
         │                             │
         │  OOM detected?              │
         │  ├── aggressive_cleanup()   │
         │  │   ├── gc.collect()       │
         │  │   ├── torch.cuda.empty_cache()
         │  │   └── Очистить градиенты │
         │  ├── Логировать OOM event в MLflow
         │  └── Повторить (до max)     │
         │                             │
         │  Все повторы исчерпаны?     │
         │  └── OOMRecoverableError    │
         └─────────────────────────────┘
```

### Поток оценки качества

```text
EvaluationRunner
  1. Загрузить JSONL eval dataset → список (question, expected_answer, metadata)
  2. Получить ответы модели через vLLM endpoint → list[EvalSample]
  3. Для каждого включенного plugin (по priority):
       result = plugin.evaluate(samples)
  4. Агрегировать результаты → RunSummary (passed/failed, metrics, recommendations)
```

### Поток генерации отчета

```text
ryotenkai report <run_dir>
  │
  ▼
MLflow ──► выгрузить runs, metrics, artifacts, configs
  │
  ▼
Собрать модель отчета (phases, issues, timeline)
  │
  ▼
Запустить plugins (каждый рендерит свою секцию отчета)
  │
  ▼
Срендерить Markdown → experiment_report.md
  │
  └── затем снова сохранить в MLflow как artifact
```

---

## Быстрый старт

### Установка одной командой

```bash
git clone https://github.com/DanilGolikov/ryotenkai.git
cd ryotenkai
bash setup.sh
source .venv/bin/activate
```

### Настройка

1. Отредактируйте `secrets.env` и добавьте API-ключи (RunPod, HuggingFace)
2. Скопируйте пример конфигурации и адаптируйте его под себя:

```bash
cp src/config/pipeline_config.yaml my_config.yaml
# Настройте my_config.yaml: модель, датасет и параметры провайдера
```

### Запуск

```bash
# Проверить конфиг
ryotenkai config-validate --config my_config.yaml

# Запустить полный пайплайн
ryotenkai train --config my_config.yaml

# Или запустить обучение локально (для разработки)
ryotenkai train-local --config my_config.yaml
```

### Интерактивный TUI

```bash
ryotenkai tui
```

TUI дает удобную навигацию по запускам, позволяет смотреть статусы стадий и следить за live-пайплайнами прямо из терминала.

---

## Конфигурация

RyotenkAI использует один YAML-конфиг (schema v7). Ключевые секции:

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"

training:
  type: qlora                    # qlora | lora | adalora | full
  provider: single_node          # single_node | runpod
  strategies:
    - strategy_type: sft
      hyperparams: { epochs: 3 }
    - strategy_type: dpo
      hyperparams: { epochs: 1 }

datasets:
  default:
    source_hf:
      train_id: "your-org/dataset"

providers:
  single_node:
    connect:
      ssh: { alias: pc }
    training:
      workspace_path: /home/user/workspace
      docker_image: "ryotenkai/ryotenkai-training-runtime:latest"

mlflow:
  tracking_uri: "http://localhost:5002"
  experiment_name: ryotenkai
```

Полный справочник по конфигу: [`../src/config/CONFIG_REFERENCE.md`](../src/config/CONFIG_REFERENCE.md)

---

## Стратегии обучения

RyotenkAI поддерживает многофазное обучение с chaining стратегий. Стратегии определяют, **что** обучать; adapters (LoRA, QLoRA, AdaLoRA, Full FT) определяют, **как** именно это делать.

| Стратегия | Сигнал | Когда использовать |
|----------|--------|--------------------|
| **CPT** (Continued Pre-Training) | raw text | Добавить модели доменные знания |
| **SFT** (Supervised Fine-Tuning) | instruction → response pairs | Научить модель формату задачи |
| **CoT** (Chain-of-Thought) | reasoning traces | Улучшить пошаговое рассуждение |
| **DPO** (Direct Preference Optimization) | chosen vs rejected pairs | Выравнивание под человеческие предпочтения |
| **ORPO** (Odds Ratio Preference Optimization) | chosen vs rejected pairs | Alignment без отдельной reward model |
| **GRPO** (Group Relative Policy Optimization) | reward-guided RL | Reinforcement learning от reward-сигнала |
| **SAPO** (Self-Aligned Preference Optimization) | chosen vs rejected + self-alignment | Улучшенное preference learning |

Стратегии можно объединять в цепочку: `CPT → SFT → DPO` будет выполнено последовательно, где каждая следующая фаза использует checkpoint предыдущей. Вся цепочка полностью настраивается через YAML.

---

## GPU-провайдеры

Провайдеры отвечают за подготовку GPU для обучения и инференса. Для training и inference используются отдельные provider interfaces.

| Провайдер | Тип | Training | Inference | Способ подключения |
|----------|-----|----------|-----------|--------------------|
| **single_node** | Локальный | SSH к вашему GPU-серверу | vLLM через Docker по SSH | alias из `~/.ssh/config` или явные host/port/key |
| **RunPod** | Облако | Pod через GraphQL API | Provision volume + pod | API-ключ в `secrets.env` |

### single_node

Прямой SSH-доступ к машине с GPU. Пайплайн разворачивает Docker container с training runtime, синхронизирует код, запускает обучение и забирает артефакты - все по SSH. Инференс разворачивает vLLM container на той же машине.

Возможности: автоопределение GPU через `nvidia-smi`, health checks, очистка workspace.

### RunPod

Облачный GPU через RunPod API. Пайплайн создает pod с нужным типом GPU, ждет готовности SSH, запускает обучение и при необходимости удаляет pod после завершения. Для инференса поднимается persistent volume и отдельный pod.

Возможности: spot instances, несколько типов GPU, автоочистка через `cleanup.auto_delete_pod`.

---

## Системы плагинов

В RyotenkAI есть три plugin system, и все они устроены одинаково: декоратор `@register`, auto-discovery и изолированные namespace для секретов через `secrets.env`.

### Валидация датасетов

Проверяет датасеты до старта обучения. Плагины валидируют формат, качество, разнообразие и доменно-специфичные ограничения. Это первая стадия пайплайна - если валидация не проходит, обучение не начинается.

Пространство секретов: `DTST_*` - Документация: [`../src/data/validation/README.md`](../src/data/validation/README.md)

### Оценка качества

Оценивает качество модели после обучения по live vLLM endpoint. Плагины запускают детерминированные проверки (syntax, semantic match) и LLM-as-judge scoring. Результаты затем попадают в отчет по эксперименту.

Пространство секретов: `EVAL_*` - Документация: [`../src/evaluation/plugins/README.md`](../src/evaluation/plugins/README.md)

### Генерация отчетов

Генерирует отчеты по экспериментам на основе данных из MLflow. Каждый плагин отвечает за одну секцию Markdown-документа: header, summary, metrics, issues и так далее. Готовый отчет затем логируется обратно в MLflow как artifact.

Документация: [`../src/reports/plugins/README.md`](../src/reports/plugins/README.md)

Все plugin systems поддерживают и кастомные плагины: достаточно реализовать базовый класс, пометить его через `@register`, и пайплайн автоматически его подхватит.

---

## Интеграция с MLflow

Запустите стек MLflow:

```bash
make docker-mlflow-up
```

Интерфейс будет доступен по адресу `http://localhost:5002`. Все запуски пайплайна трекаются с метриками, артефактами и снапшотами конфигурации.

---

## Docker-образы

| Образ | Назначение |
|-------|------------|
| `ryotenkai/ryotenkai-training-runtime` | CUDA + PyTorch + зависимости для обучения |
| `ryotenkai/inference-vllm` | Runtime для vLLM-инференса (serve + merge deps + SSH) |

Образы можно собирать локально или публиковать в Docker Hub. См. [`../docker/training/README.md`](../docker/training/README.md) и [`../docker/inference/README.md`](../docker/inference/README.md).

---

## Справочник CLI

| Команда | Описание |
|---------|----------|
| `ryotenkai train --config <path>` | Запустить полный training pipeline |
| `ryotenkai train-local --config <path>` | Запустить обучение локально (без удаленного GPU) |
| `ryotenkai validate-dataset --config <path>` | Запустить только валидацию датасета |
| `ryotenkai config-validate --config <path>` | Статические pre-flight проверки конфига |
| `ryotenkai info --config <path>` | Показать конфигурацию пайплайна и модели |
| `ryotenkai tui [run_dir]` | Запустить интерактивный TUI |
| `ryotenkai inspect-run <run_dir>` | Проинспектировать директорию запуска |
| `ryotenkai runs-list [dir]` | Показать список всех запусков с краткой сводкой |
| `ryotenkai logs <run_dir>` | Показать pipeline log конкретного запуска |
| `ryotenkai run-status <run_dir>` | Live-мониторинг запущенного пайплайна |
| `ryotenkai run-diff <run_dir>` | Сравнить конфиг между попытками |
| `ryotenkai report <run_dir>` | Сгенерировать MLflow-отчет по эксперименту |
| `ryotenkai version` | Показать информацию о версии |

---

## Terminal UI (TUI)

В RyotenkAI есть встроенный терминальный интерфейс для мониторинга и разбора training runs:

```bash
ryotenkai tui             # посмотреть все запуски
ryotenkai tui <run_dir>   # открыть конкретный запуск
```

**Список запусков** - обзор всех pipeline runs со статусом, длительностью и именем конфига:

<p align="center">
  <img src="../docs/screenshots/tui_runs_list.png" alt="TUI Runs List" width="800">
</p>

**Детали запуска** - можно провалиться в любой run и посмотреть стадии, тайминги, outputs и результаты валидации:

<p align="center">
  <img src="../docs/screenshots/tui_run_detail.png" alt="TUI Run Detail" width="800">
</p>

**Ответы на evaluation** - просмотр ответов модели рядом с ожидаемыми ответами:

<p align="center">
  <img src="../docs/screenshots/tui_eval_answers.png" alt="TUI Evaluation Answers" width="800">
</p>

TUI включает вкладки **Details**, **Logs**, **Inference**, **Eval** и **Report** - все, что нужно для понимания training run, не выходя из терминала.

---

## Разработка

### Setup

```bash
bash setup.sh
source .venv/bin/activate
```

### Тесты

```bash
make test          # все тесты
make test-unit     # только unit-тесты
make test-fast     # пропустить slow-тесты
make test-cov      # с coverage
```

### Линтинг

```bash
make lint          # проверка
make format        # автоформатирование
make fix-all       # автоисправление
```

### Pre-commit

Pre-commit hooks запускаются автоматически. Для ручного запуска:

```bash
make pre-commit
```

---

## Структура проекта

```text
ryotenkai/
├── src/
│   ├── config/          # Схемы конфигурации (Pydantic v2)
│   ├── pipeline/        # Оркестрация и реализации стадий
│   ├── training/        # Стратегии обучения и orchestration
│   ├── providers/       # GPU-провайдеры (single_node, RunPod)
│   ├── evaluation/      # Плагины оценки качества модели
│   ├── data/            # Работа с датасетами и плагины валидации
│   ├── reports/         # Плагины генерации отчетов
│   ├── tui/             # Terminal UI (Textual)
│   ├── utils/           # Общие утилиты
│   └── tests/           # Набор тестов
├── docker/
│   ├── training/        # Docker-образ training runtime
│   ├── inference/       # Docker-образы для инференса
│   └── mlflow/          # Стек MLflow (docker-compose)
├── scripts/             # Вспомогательные скрипты
├── docs/                # Документация и диаграммы
├── setup.sh             # Установка одной командой
├── Makefile             # Команды для разработки
└── pyproject.toml       # Метаданные пакета и конфигурация инструментов
```

## Сообщество

Присоединяйтесь к Discord-серверу для поддержки, обсуждения roadmap, конфигов и fine-tuning workflow:

[discord.gg/QqDM2DbY](https://discord.gg/QqDM2DbY)

## Участие в проекте

См. [../CONTRIBUTING.md](../CONTRIBUTING.md).

## Лицензия

[MIT](../LICENSE) © Golikov Daniil
