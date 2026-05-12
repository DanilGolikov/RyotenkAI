# L11 — Visual regression (Phase 6)

Уровень визуальных регрессионных тестов для фронтенда. **Низкий приоритет**
(зафиксировано пользователем) — реализация ожидается в Phase 6 из
[docs/plans/structured-hopping-starfish.md](../../docs/plans/structured-hopping-starfish.md).

## Что здесь будет

```
tests/visual/
├── README.md                  # этот файл
├── web/                       # Playwright traces + lost-pixel baselines
│   ├── baselines/             # PNG/JPEG snapshots (LFS)
│   ├── components/            # per-Storybook-story снимки
│   └── flows/                 # end-to-end user-flow снимки
├── conftest.py                # Playwright + lost-pixel runner
└── test_*_visual.py           # pytest-обёртки для запуска
```

## Инструментарий

* **Storybook** (`web/.storybook/`) — каждый UI-компонент описывается
  story-ями; визуальные baselines генерируются из stories.
* **lost-pixel** (self-hosted) — diff PNG-снимков. Хост — наш собственный,
  не Chromatic (см. ниже).
* **Playwright** + `trace.video()` — для flow-level снимков.
* **@axe-core/playwright** — accessibility checks вместе со снимками.

## Почему не Chromatic

* Бюджет: Chromatic у нас нет (open-source проект).
* Self-hosting предпочтительнее: lost-pixel катится локально и в CI без
  внешних зависимостей.
* Контроль над хранением baselines — мы выкладываем в Git LFS, а не во
  внешний сервис.

## Запуск (когда будет реализовано)

```bash
# Локально
make test-visual

# Полный регрессионный прогон с regen baselines
make test-visual-update
```

В CI этот lane триггерится:

* `presubmit-blocking` — если PR трогает `web/src/**` или `*.stories.tsx`.
* Бюджет: <5 минут на PR.

## Конвенции

* **Baselines в LFS**, не в обычном Git — иначе репозиторий вспухает.
* **Threshold = 0.1%** разницы пикселей — больше = блок merge.
* **Story = единица снимка**: одна story → одна baseline. Никаких
  динамически генерируемых snapshots.
* **Dark/Light темы** снимаются обе.
* **Mobile viewport** (375px) + **desktop** (1280px) — два размера.

## TODO

* [ ] Поднять `web/.storybook/`.
* [ ] Настроить lost-pixel + LFS под baselines.
* [ ] Добавить `make test-visual` цель в [Makefile](../../Makefile).
* [ ] Workflow `.github/workflows/visual.yml` с self-hosted runner.
* [ ] Stories для core-компонентов: `FieldRenderer`, `RunCard`,
  `ConfigBuilder`, `ReportViewer`.
