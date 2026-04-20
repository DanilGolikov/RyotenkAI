# ConfigBuilder v3 — polish + array editor + delete provider + discriminated training

## Context

Пользователь прислал скриншот с явными UX-проблемами текущего Config tab и
список запросов:

1. Поля мелкие и плотные — **нужно больше воздуха**.
2. **Описание поля** (Pydantic `description`) сейчас — мелкий серый текст под
   инпутом. Нужен **«?» над/рядом с полем** с tooltip'ом.
3. В местах где есть дискриминатор (`training.type = qlora | lora | adalora`)
   **делать явный выбор** — dropdown + только выбранная ветка.
4. На странице провайдера **нет кнопки удаления** (`useDeleteProvider`
   существует, UI нет).
5. **Кнопка добавления провайдера «в настройках»** — в контексте
   ProviderPicker проекта: когда зарегистрированных нет / нужен новый,
   быстрый путь «Create in Settings →» (user choice).
6. Блоки arrays (`plugins[]`, `strategies[]`, и т.д.) падают в
   "advanced — edit via YAML" — **нужен реальный array editor**.
7. На активной подвкладке (e.g. `TrainingOnlyConfig`) сейчас ещё раз висит
   chevron «collapse», хотя вкладка и так открыта — **убрать дропдаун**,
   но сохранить «?» help.
8. «Блок конфига внутри другого блока» — двойная рамка (Card → FormGroup):
   **убрать внешний бордер активной группы**, чтобы было одно общее
   пространство.
9. **Опциональные поля**: остаются скрытыми за "Show N optional fields"
   toggle, но когда показаны — визуально неотличимы от required (сейчас
   есть `border-t` сверху и визуальное зонирование).

Audit (conversation history) показал конкретные точки:
[`FieldRenderer.tsx:22-46`](../../web/src/components/ConfigBuilder/FieldRenderer.tsx) — LabelledRow с description СНИЗУ;
[`FieldRenderer.tsx:303-311`](../../web/src/components/ConfigBuilder/FieldRenderer.tsx) — AdvancedJsonPreview fallback для array/unknown;
[`FieldRenderer.tsx:231-298`](../../web/src/components/ConfigBuilder/FieldRenderer.tsx) — ObjectFields с `border-t` вокруг optional;
[`FormGroup.tsx:18-36`](../../web/src/components/ConfigBuilder/FormGroup.tsx) — всегда collapsible с chevron;
[`ConfigBuilder.tsx:130-142`](../../web/src/components/ConfigBuilder/ConfigBuilder.tsx) — activeNode всегда идёт через FormGroup (двойной border);
[`ProviderDetail.tsx:32-80`](../../web/src/pages/ProviderDetail.tsx) — нет delete-кнопки;
[`useProviders.ts:120-126`](../../web/src/api/hooks/useProviders.ts) — `useDeleteProvider` уже есть, не используется.

`UnionField.tsx` уже работает для `anyOf` с discriminator'ом, но
`training.type` — это **enum-поле `type`** + 3 optional object-поля в siblings
(не `anyOf`). Нужна отдельная эвристика.

## Decisions

- Задача UX-only, бэкенд не трогаем. Все правки в
  `web/src/components/ConfigBuilder/` + `ProviderDetail.tsx` + `Providers.tsx` +
  `ProviderPickerField.tsx`.
- Массивы и dict-поля — реальный inline editor. Fallback JSON preview
  остаётся только для `kind === 'unknown'` (edge-case).
- Discriminator-heuristic детектит pattern `object { type: enum, ...N
  siblings с именами из enum values }` и прячет не-matching siblings
  autoматически — generic, не hardcoded под `training`.
- «Кнопка добавления провайдера» — в ProviderPicker проекта (user choice).
  `/settings/providers#new` автоматически открывает NewProviderModal.
- tooltip'ы — pure CSS (group-hover) без новых зависимостей.

## Out of scope

- Drag-to-reorder массивов (add/remove только).
- Полноценный map-editor для `additionalProperties=Any` (редкий кейс —
  остаётся JSON preview).
- Undo/redo, per-field validation feedback (у нас debounced full-config
  validate уже рисует banner).
- Bulk edit.

---

## Task A — Field spacing + "?" help tooltip (header-level description)

Файлы:
- `web/src/components/ConfigBuilder/HelpTooltip.tsx` **(новый)** — кнопка «?»
  с CSS-only hover-tooltip (`group-hover:opacity-100`, absolute positioned
  box). Проп: `text: string`. Рендерится только если `text` не пуст.
- [`web/src/components/ConfigBuilder/FieldRenderer.tsx`](../../web/src/components/ConfigBuilder/FieldRenderer.tsx):
  - `LabelledRow`: label строка становится `flex items-center gap-2` —
    `{title} {required ? '*' : ''} <HelpTooltip text={description} />`.
    **Убрать description из-под инпута.**
  - Классы инпутов `px-2 py-1.5 text-xs` → `px-3 py-2 text-sm`.
    Consistency c `NewProviderModal.tsx`.
  - `ObjectFields`: `space-y-3` → `space-y-4`.

## Task B — Make FormGroup optionally non-collapsible

Файл: [`web/src/components/ConfigBuilder/FormGroup.tsx`](../../web/src/components/ConfigBuilder/FormGroup.tsx).

- Добавить проп `collapsible?: boolean` (default `true` — не ломает
  существующие вызовы).
- При `collapsible={false}`: не рендерить chevron, убрать `<button>`
  wrapper, `useState(defaultOpen)` → всегда `true`.
- Добавить место для `<HelpTooltip>` в header — проп
  `helpText?: string` (опциональный, используется при `collapsible=false`
  для top-level групп где нет отдельного FieldRenderer).

## Task C — ConfigBuilder flat render на depth=0 (одна рамка, без chevron)

Файл: [`web/src/components/ConfigBuilder/ConfigBuilder.tsx`](../../web/src/components/ConfigBuilder/ConfigBuilder.tsx:130-142).

Сейчас `FieldRenderer` при `depth === 0 && kind === 'object'` оборачивает в
`FormGroup` с `border + bg-surface-1 + chevron`. Это даёт «карточка внутри
карточки» и ненужный chevron.

- В `FieldRenderer.tsx` object-branch: при `depth === 0` рендерить **не
  FormGroup**, а плоско:
  ```
  <header>  {title}  {required marker}  {HelpTooltip}  </header>
  <ObjectFields ... depth={1} />
  ```
  Без bg, без border, без padding — используется окружающий Card (ProjectDetail).
- Вложенные объекты (`depth >= 1`) остаются как есть: `space-y-2 pl-3
  border-l border-line-1` — это визуальный indent, не рамка.

## Task D — Optional fields стилем = required

Файл: [`web/src/components/ConfigBuilder/FieldRenderer.tsx:285-298`](../../web/src/components/ConfigBuilder/FieldRenderer.tsx).

- Убрать `pt-1 border-t border-line-1/60` вокруг optional-секции.
- Toggle-кнопка «Show N optional fields» — сохранить, но сделать
  subtle (меньший stroke, без uppercase tracking).
- Когда expanded — rows в том же `space-y-4`, идентично required.

## Task E — Array editor

Файлы:
- `web/src/components/ConfigBuilder/ArrayField.tsx` **(новый)**. Пропы:
  `{ root, node, value, onChange, labelKey, required, path, hashPrefix }`.
  Схема берётся из `node.items` (resolveRef). Рендер:
  - Header: title + HelpTooltip + «+ Add» кнопка справа (ink-3).
  - Каждый item: row с «drag-handle placeholder» (не активный в MVP) +
    `FieldRenderer` на `schema.items` + «✕ Remove» кнопка.
  - При пустом массиве — outline-блок `+ Add` и подсказка "No items".
  - Value всегда trusted array (если value undefined → `[]`).
- [`FieldRenderer.tsx`](../../web/src/components/ConfigBuilder/FieldRenderer.tsx):
  ветка `kind === 'array'` → `<ArrayField>` вместо AdvancedJsonPreview.
  Остаётся `kind === 'unknown'` → JSON preview (edge-case).

## Task F — Delete provider affordance

Файл: [`web/src/pages/ProviderDetail.tsx`](../../web/src/pages/ProviderDetail.tsx).

- Добавить `useDeleteProvider` hook + `useNavigate`.
- В header напротив title (справа) — `Delete` кнопка (red outline, text-err).
- `onClick`: `window.confirm('Unregister provider "<name>"? Files on disk stay.')`.
  API endpoint сейчас **unregister-only** (файлы сохраняются) — донести в
  тексте.
- После успешного delete → `navigate('/settings/providers')`.
- Consistent тест — /settings/providers список обновится (hook уже
  инвалидирует query).

## Task G — Discriminated object rendering (`training.type` + siblings)

Файл: [`web/src/components/ConfigBuilder/FieldRenderer.tsx`](../../web/src/components/ConfigBuilder/FieldRenderer.tsx).

Эвристика **в `ObjectFields`**: перед обычным required-first разделением,
проверить:
1. Среди properties есть поле `type` (или любое одно-из enum-полей
   с `enum`/`anyOf-of-const`).
2. Значения его enum'а ⊆ именам других properties (siblings).

Если да — это discriminator pattern:
- Показываем `type` как dropdown.
- Из `N` siblings-блоков (qlora/lora/adalora) рендерим **только один** —
  соответствующий выбранному `type`.
- Остальные siblings остаются в значении (не стираем при переключении —
  user может хотеть вернуться), но в UI скрыты.
- Остальные non-sibling-non-type поля (hyperparams, strategies, provider,
  ...) рендерятся как обычно.

Реализация — маленькая pre-pass функция `detectDiscriminator(node) → {
enumKey, valueToSiblingMap } | null`, чистая функция, легко тестируется.

## Task H — ProviderPicker «Create in Settings →» + deep-link

Файлы:
- [`web/src/components/ConfigBuilder/ProviderPickerField.tsx`](../../web/src/components/ConfigBuilder/ProviderPickerField.tsx):
  в dropdown «Add from Settings» добавить постоянный пункт
  **«+ Create new provider in Settings →»** в конце списка (всегда viable,
  не только когда пусто). Navigate to `/settings/providers#new`.
- [`web/src/pages/Providers.tsx`](../../web/src/pages/Providers.tsx): на
  mount проверить `window.location.hash === '#new'`, тогда авто-открыть
  NewProviderModal и очистить hash (`history.replaceState`).

---

## Files — summary

**Create:**
- `web/src/components/ConfigBuilder/HelpTooltip.tsx`
- `web/src/components/ConfigBuilder/ArrayField.tsx`

**Modify:**
- `web/src/components/ConfigBuilder/FieldRenderer.tsx` — A, C (flat depth=0), D, E wiring, G (discriminator)
- `web/src/components/ConfigBuilder/FormGroup.tsx` — B (collapsible prop)
- `web/src/components/ConfigBuilder/ConfigBuilder.tsx` — проверить что C не требует изменений здесь (работа в FieldRenderer)
- `web/src/components/ConfigBuilder/ProviderPickerField.tsx` — H
- `web/src/pages/ProviderDetail.tsx` — F (delete кнопка)
- `web/src/pages/Providers.tsx` — H (hash #new → auto-open)

## Verification

1. `npx tsc --noEmit` clean.
2. `npm run build` green.
3. `pytest src/tests/integration/api/` green (API без изменений — 146/146).
4. `make verify-api-sync` green.
5. Manual preview на smoke-project и helixql-nl2hql-v7-mini:
   a. Config tab → активная вкладка **без chevron, без внешнего бордера**,
      header с «?» tooltip при hover.
   b. Поля — крупнее (`text-sm`, `py-2`), межстрочное пространство
      `space-y-4`.
   c. `training` группа → dropdown `type: qlora`, видны только
      qlora-поля + общие (hyperparams, strategies). Переключение на `lora`
      → скрывает qlora, показывает lora.
   d. `datasets.default.validations.plugins` → **array editor** с Add/Remove,
      каждый элемент рендерится через FieldRenderer. Сохраняется в YAML
      корректно (Save → verify через `/api/v1/projects/*/config`).
   e. Optional fields под toggle — визуально идентичны required (нет
      `border-t`).
   f. `/settings/providers/runpod-prod` → появилась **Delete** кнопка
      (красная). Clicking → confirm → redirect на /settings/providers,
      список обновлён.
   g. `/projects/*/config` → Providers tab → «+ Add from Settings» →
      **«+ Create new provider in Settings →»** внизу списка → navigate
      to `/settings/providers`, модалка открывается автоматически.

## Implementation order

1. **A + B + C** — базовый layout polish (spacing, «?», flat depth=0).
   Один коммит.
2. **D** — optional fields restyle. Коммит.
3. **E** — ArrayField. Коммит.
4. **G** — discriminator rendering. Коммит.
5. **F** — Delete provider. Коммит.
6. **H** — Create-in-Settings link + auto-open hash. Коммит.
7. Final verify (pytest + tsc + build + verify-api-sync + preview).
