# Удалить мёртвый реестр активных подов RunPod

## Context

Файл `runs/ryotenkai_active_pods.json` и методы `RunPodCleanupManager.register_pod` /
`unregister_pod` / `list_registered_pods` были задуманы как on-disk реестр для
recovery-свипа сиротских GPU-подов после краша процесса. Но свип так и не был
написан: grep по всему репо показывает, что реестр **никто не читает**, кроме
юнит-тестов самого класса. Фактически это write-only артефакт, который:

- создаёт лишний I/O на hot path (read+write всего JSON на каждый
  `register_pod` / `unregister_pod`);
- race-prone при нескольких параллельных тренингах — без file-lock и без
  atomic-replace;
- замусоривает `runs/` и путает читателей кода ложным обещанием recovery.

Выкидываем половинчатую фичу. `RunPodCleanupManager.cleanup_pod` остаётся —
через него провайдер терминейтит под, это реально рабочий путь.

## Files to modify

### 1. [src/providers/runpod/training/cleanup_manager.py](src/providers/runpod/training/cleanup_manager.py)
- Удалить в `__init__` установку `self.registry_file` и `mkdir` для `runs/`
  (строки 42–46).
- Удалить методы `register_pod` (48–73), `unregister_pod` (75–87),
  `list_registered_pods` (100–109).
- Упростить `cleanup_pod` — убрать вызов `self.unregister_pod(pod_id)`, оставить
  только `self.api_client.terminate_pod(pod_id)` с логом.
- Вычистить ставшие ненужными импорты: `json`, `time`, `Path`.
- Обновить docstring класса — убрать упоминания «Register active pods in JSON
  file» и «Unregister pods after successful cleanup».
- `create_cleanup_manager` и `__all__` оставить как есть.

### 2. [src/providers/runpod/training/provider.py](src/providers/runpod/training/provider.py)
- Удалить строку 279:
  `self._cleanup_manager.register_pod(pod_id=pod_id, api_base=RUNPOD_API_BASE_URL)`.
- Больше точек вызова `register_pod` в провайдере нет — `cleanup_pod` (строки
  152, 164, 178, 187, 198, 206, 232, 242, 310, 362) остаются без изменений.

### 3. [src/providers/runpod/training/__init__.py](src/providers/runpod/training/__init__.py)
- Изменений нет. `RunPodCleanupManager` и `create_cleanup_manager` по-прежнему
  экспортируются — публичный API сохраняется.

### 4. [src/tests/unit/pipeline/providers/runpod/test_cleanup_manager.py](src/tests/unit/pipeline/providers/runpod/test_cleanup_manager.py)
- Удалить тесты: `test_register_and_unregister_pod` (23–37),
  `test_register_pod_tolerates_corrupted_registry` (40–48),
  `test_list_registered_pods_handles_missing_file` (51–56).
- Переписать `test_cleanup_pod_unregisters_on_success` (59–68) в
  `test_cleanup_pod_terminates_pod`: проверять только `res.is_success()` и
  `api.terminated == ["pod-1"]`.
- Переписать `test_cleanup_pod_keeps_registry_on_failure` (71–79) в
  `test_cleanup_pod_propagates_failure`: проверять `res.is_failure()` и
  `api.terminated == ["pod-1"]`.
- Убрать `import json` и `from pathlib import Path`, если больше не нужны.
- `test_create_cleanup_manager_uses_api_client` (82–96) оставить без изменений.

### 5. [src/tests/unit/pipeline/providers/runpod/test_provider.py](src/tests/unit/pipeline/providers/runpod/test_provider.py)
- В `StubCleanup` (34–44) удалить поле `registered` и метод `register_pod`.
- В `test_connect_success` удалить утверждения на строках 114–117
  (`cleanup.registered…`).

### 6. Файл [runs/ryotenkai_active_pods.json](runs/ryotenkai_active_pods.json)
- Удалить файл с диска. Сейчас он содержит только `{}`, ничего ценного.

## Verification

1. **Grep-аудит** — не должно остаться совпадений в production- и тест-коде:
   ```
   rg "ryotenkai_active_pods|register_pod|unregister_pod|list_registered_pods" src/
   ```
2. **Тесты**:
   ```
   pytest src/tests/unit/pipeline/providers/runpod/ -v
   ```
   Все оставшиеся тесты `test_cleanup_manager.py` и `test_provider.py` должны
   пройти.
3. **Линт и типы**:
   ```
   ruff check src/providers/runpod/training/ src/tests/unit/pipeline/providers/runpod/
   mypy src/providers/runpod/training/
   ```
4. **Sanity-импорт** — убедиться, что переэкспорт не сломан:
   ```
   python -c "from src.providers.runpod.training import RunPodCleanupManager, create_cleanup_manager; print('ok')"
   ```
5. **Repowise decision record** — после применения изменений вызвать
   `update_decision_records(action="create", ...)` с title вроде «Remove
   write-only RunPod active-pods registry» и affected_files = список выше.
