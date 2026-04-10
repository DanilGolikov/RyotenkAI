# Resilient Pipeline — Risks & Questions

## Контекст
Три фичи для поддержки long-running training с laptop в спящем режиме:
- **Feature A**: Resilient Resume — корректное возобновление мониторинга после сна
- **Feature B**: Pod Auto-Stop — автоматическая остановка пода после завершения тренировки
- **Feature C**: Metrics Buffer — локальный буфер метрик на поде со smart decimation

---

## Риски

### R1: API-ключ RunPod на поде (Feature B)
- **Описание**: Для вызова `stop_pod()` с пода нужен RunPod API key
- **Уровень**: LOW
- **Ответ**: Под эфемерный, single-user, изолированный. API key уже используется control plane. Добавляем через `.env` аналогично `HF_TOKEN`. Не хуже текущей модели безопасности.

### R2: Размер файла буфера метрик (Feature C)
- **Описание**: Буфер может расти при долгих тренировках
- **Уровень**: VERY LOW  
- **Ответ**: 432 шага × 10 метрик × 8 байт ≈ 35KB. С decimation ещё меньше. Даже 10K шагов — менее 1MB.

### R3: SSH порт может измениться при рестарте пода (Feature A)
- **Описание**: При `start_pod()` после `stop_pod()` SSH endpoint может быть другим
- **Уровень**: MEDIUM
- **Решение**: При resume query RunPod API для получения актуального SSH endpoint. Не полагаться на сохранённый в lineage.

### R4: Race condition — pipeline resume пока под останавливается (Feature A+B)
- **Описание**: Pipeline пробуждается и пытается подключиться к поду, который ещё в процессе остановки
- **Уровень**: LOW
- **Решение**: Проверить статус пода через API перед SSH. Если `STOPPING` — подождать. Если `STOPPED` — `start_pod()`.

### R5: Порядок метрик при replay (Feature C)
- **Описание**: Буферизованные метрики могут прибыть не по порядку
- **Уровень**: LOW
- **Решение**: Буфер хранит `(metric_name, value, step, timestamp)`. Replay по step order.

### R6: Stale SSH ControlMaster socket (Feature A)
- **Описание**: После сна SSH мультиплексированный сокет будет невалидным
- **Уровень**: LOW
- **Решение**: `SSHClient` уже пересоздаёт соединение при reconnect. Добавить explicit `close_master()` перед reconnect.

### R7: stop_pod() vs terminate_pod() — что с данными? (Feature B)
- **Описание**: Сохраняется ли `/workspace` при `stop_pod()`?
- **Уровень**: LOW (уточнено)
- **Ответ**: RunPod docs: `stop_pod()` — GPU освобождается, billing прекращается, container disk + volume **СОХРАНЯЮТСЯ**. `start_pod()` возобновляет с теми же данными. Подтверждено: это именно то, что нужно.

### R8: Decimation теряет важные метрики (Feature C)
- **Описание**: Smart decimation может потерять точку перелома в обучении
- **Уровень**: LOW
- **Ответ**: Первые 10 мин (≈60 шагов) — ВСЕ метрики. Следующие 20 мин (≈120 шагов) — каждая вторая. Ранняя динамика полностью сохранена, позже кривая сглаживается. 170/432 ≈ 40% метрик сохранено — достаточно для trend analysis.

### R9: MLflow отвергает старые step numbers при flush (Feature C)
- **Описание**: Если MLflow получил step=100, а потом мы шлём step=50 из буфера
- **Уровень**: MEDIUM
- **Решение**: Буфер трекает `last_synced_step`. Flush только метрик с `step > last_synced_step`. MLflow принимает metrics с произвольным step (не строго монотонный), но для clean UI — шлём по порядку.

### R10: Что если stop_pod() call падает с пода? (Feature B)
- **Описание**: Сетевая ошибка при вызове API с пода
- **Уровень**: LOW
- **Решение**: Retry 3 раза с exponential backoff в bash-обёртке. Если всё равно не удалось — под остаётся running (graceful degradation к текущему поведению). Pipeline при resume найдёт running pod и остановит.

### R11: Порядок имплементации
- **Описание**: Зависимости между фичами
- **Уровень**: LOW
- **Ответ**: Все три фичи НЕЗАВИСИМЫ. Рекомендуемый порядок: B → A → C (по ROI: B сразу экономит деньги).

### R12: Крэш тренировки до записи маркера (Feature B)
- **Описание**: OOM/SIGKILL убивает процесс до записи TRAINING_COMPLETE/FAILED
- **Уровень**: LOW
- **Решение**: Bash wrapper использует `trap` на EXIT. Проверяет exit code Python процесса. Если не 0 и нет маркера → пишет TRAINING_FAILED → вызывает stop_pod().

### R13: Совместимость с existing resume flow (Feature A)
- **Описание**: Новая логика может сломать текущий resume
- **Уровень**: LOW
- **Решение**: Все изменения аддитивные. Если под running — текущий flow работает. Новый flow (stopped pod → start → reconnect) — дополнительная ветка. Fallback к текущему поведению при любой ошибке.

### R14: Что если пользователь хочет inspect checkpoint на поде? (Feature B)
- **Описание**: Auto-stop может помешать debug
- **Уровень**: LOW
- **Решение**: Конфигурируемый `auto_stop_after_training: bool = true` в cleanup config. При `keep_pod_on_error: true` — не останавливать при ошибке.

---

## Открытые вопросы (все resolved)

Все вопросы разрешены в ходе анализа. Готов к имплементации.
