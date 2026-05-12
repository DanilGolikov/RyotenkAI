# L12 — Replay regression (Phase 6)

Уровень replay-регрессий: записанные user-flows воспроизводятся как тест
на каждый релиз. **Phase 6 territory** — реализация после визуальных
снимков, см. [docs/plans/structured-hopping-starfish.md](../../docs/plans/structured-hopping-starfish.md).

## Что здесь будет

```
tests/replay/
├── README.md                  # этот файл
├── corpus/                    # Playwright .zip traces (LFS)
│   ├── 2026-05/               # rotated monthly
│   │   ├── create-project.zip
│   │   ├── launch-grpo-run.zip
│   │   └── ...
│   └── rotation.json          # manifest: trace → first-seen, last-seen
├── conftest.py                # Playwright trace replay runner
└── test_replay_<flow>.py      # one pytest per archived flow
```

## Инструментарий

* **Playwright `trace.zip`** — формат записи. Полный DOM-snapshot,
  network log, screenshots на каждое действие.
* **Playwright trace viewer** (`npx playwright show-trace`) — для
  ручного дебага упавшего replay.
* **Hermetic stack** (`tests/_harness/stack/`) — backend поднимается из
  того же fake-стека, что и в L6 (один процесс на тест).

## Политика ротации corpus

* **90 дней** — стандартный TTL для traces.
* Trace, не вызвавший падение 90 дней подряд, перекладывается в архив
  (`corpus/_archive/`) и больше не запускается в CI, но остаётся доступным
  для ручного re-record.
* **Re-record** триггерится:
  * после major-релиза (новая mажорная версия UI);
  * после рефакторинга компонента, который трогает >5 traces подряд;
  * при ручном `make replay-rerecord FLOW=create-project`.

## Запуск (когда будет реализовано)

```bash
# Один flow
make replay FLOW=launch-grpo-run

# Полный corpus
make replay-all
```

В CI запускается в **release-gate** lane — блокирует merge tags. Бюджет:
<10 минут на полный corpus.

## Конвенции

* **Один .zip — один flow**: `create-project`, `launch-grpo-run`,
  `view-report`, `edit-config`. Длинные flows резать на стадии.
* **Traces в LFS** — обычный Git не масштабируется на 30+ МБ архивы.
* **Idempotency**: replay должен пройти 3 раза подряд без падений
  (если flake — quarantine немедленно).
* **Sensitive data**: traces sanitize'аются перед commit (никаких токенов
  RunPod/MLflow в network log).
* **Trace metadata** в `rotation.json`: первая дата записи, последний
  upgrade пути, ответственный owner.

## Связь с другими уровнями

* L6 (smoke) — sanity check на каждом push; L12 — глубокий regression
  на release-gate. Не дублируем покрытие.
* L11 (visual) — снимки UI; L12 — flow behaviour. Оба нужны.

## TODO

* [ ] Playwright trace recorder + конфиг.
* [ ] LFS-конфигурация для `corpus/*.zip`.
* [ ] `make replay` + `make replay-all` цели.
* [ ] Стартовый corpus: top-5 user flows.
* [ ] `release-gate` workflow триггер.
* [ ] Rotation cron (`scripts/replay/rotate.py`) с 90-day TTL.
