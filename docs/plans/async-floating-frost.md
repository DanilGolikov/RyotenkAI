# Plan: Config page UX refactor + frontend guidelines

## Context

Страница `/projects/:slug/config` — центральный экран фронта RyotenkAI: здесь пользователь собирает pipeline для обучения модели. Я провёл аудит её с трёх точек зрения (обычный пользователь, ML‑инженер, фронт‑разработчик) и нашёл три класса проблем:

1. **Блокеры UX** — `html { font-size: 113% }` ломает rem‑based Tailwind breakpoints; `Save` disabled без объяснения; Dropdown пресетов не закрывается при переключении Form/YAML; tooltip'ы `?` перекрывают инпуты; «2 issues to fix» не ведёт настойчиво к Settings, когда не хватает секретов.
2. **ML‑поток ограничен** — нет индикатора статуса провайдера (есть ли ключ), нет client‑side валидации диапазонов, Strategies рендерятся как безликий массив (а это цепочка фаз SFT→DPO→…), preset применяется слепо без превью изменений, нельзя сравнить конфиг с last run, нет подсказок по типовым LoRA/QLoRA defaults.
3. **Качество UI** — YAML редактор — самопальный `<textarea>+overlay`, 7 разных токенов размера текста, `<label>` не связан с инпутами через `for/id`, эмодзи `🔗` вместо иконок, нет breadcrumbs и dirty‑guard.

Решение — трёхэтапный рефакторинг: P0 блокеры → P1 ML‑поток → P2 полировка + `FRONTEND_GUIDELINES.md`. **Скоуп: desktop only (`min-width: 1024px`)**; мобилка игнорируется.

**Решения принятые заранее (зафиксированы в плане):**
- font‑size 113% **оставляем**, добавляем custom breakpoints в tailwind config (ink-density специально подобрана под 18.08px).
- YAML editor — **CodeMirror 6** (≈50 KB gzip, YAML‑only нам достаточно; Monaco в 10× тяжелее).
- «Dataset coming soon» — поле остаётся, но скрываем красную `*` required и показываем muted‑баннер «configure via YAML».
- Provider status: если endpoint `/providers/:id/health` нет — **добавляем в Stage 2** минимальный ручка на бэке, а не скрейпим Settings UI.
- **Цветокор**: палитра «Grafana‑minimal» в `tailwind.config.ts` сознательно держит карточки серыми и резервирует `brand` только под CTAs/active/selection. На Config это воспринимается как монотонность, потому что весь экран = одна плоская карточка. Решение: **не расширять токены и не заливать формы брендом**, а вводить бренд **точечно через alpha 5–15%** (accent‑полоски, tinted gradients на header'ах, focus‑glow). Отдельная стадия `Stage 0.5` ниже.

---

## Reuse — не изобретать заново

- `useClickOutside` — паттерн уже inline в `PresetDropdown.tsx:17-32` и `HelpTooltip.tsx:16-31`. **Выносим в `/web/src/hooks/useClickOutside.ts`** и переиспользуем для новых панелей/модалок.
- `deepDiff` в `/web/src/lib/jsonDiff.ts` уже есть (питает `DiffBadge`). Переиспользовать для preset preview modal и «Compare with last run».
- `ValidationProvider` + `useFieldStatus` (`/web/src/components/ConfigBuilder/ValidationContext.tsx:85-110`) уже мёржит server errors + required+empty. Client‑side range валидацию вливать в тот же контекст.
- `deriveGroupValidity` + `SETTINGS_JUMP_TARGET` — уже в `ConfigTab.tsx` и работают; расширяем, не переписываем.
- `safeYamlParse`, `dumpYaml` — `/web/src/lib/yaml.ts`.
- `.btn-primary`, `.btn-ghost`, `.card`, `.pill-*`, `.nav-item` — в `/web/src/styles/globals.css:66-149`. Использовать эти классы, не плодить длинные Tailwind‑цепочки.
- `Spinner`, `SectionHeader`, `EmptyState` — `/web/src/components/ui.tsx`.
- **Иконок нет** — создать `/web/src/components/icons/` с inline SVG (`LinkIcon`, `CopyIcon`, `CheckIcon`, `ChevronIcon`, `InfoIcon`, `AlertIcon`, `DiffIcon`). `currentColor` + `aria-hidden`.

---

## Stage 0.5 — Visual hierarchy & brand accents (быстрый PR, отдельный)

Философия: бордо → фиолет проявляется **как дыхание, не как заливка**. Alpha 5–15%, акценты на краях, не на плоскостях. Политика «brand не течёт внутрь форм» сохраняется — но заголовки/активные состояния/состояние диалога получают brand‑tint.

### 0.5.1 Config card → `.card-hero`
- **Файлы**: `ConfigTab.tsx:125-126` (или слой выше в `ProjectDetail.tsx`, смотря где сейчас `.card`).
- **Что**: заменить `.card` на уже существующий `.card-hero` из `globals.css:84-89` (burgundy 22% → violet 16% wash сверху). Ровно один hero на экран — правило не нарушается.
- **Сложность**: S.

### 0.5.2 Section header «Model / Training …» — accent слева + мягкий tint
- **Файлы**: `/web/src/components/ConfigBuilder/FieldRenderer.tsx` (место рендера `<header>` у группы, искать `pb-3 border-b border-line-1`).
- **Что**: добавить `border-l-[3px] border-brand pl-3` к h3 **и** под header'ом секции `bg-gradient-to-r from-brand/8 via-transparent to-transparent` (edge‑glow слева). Чередование brand/brand‑alt по индексу группы — чтобы первая секция burgundy, следующая violet, и т.д. (visual rhythm).
- **Сложность**: S.

### 0.5.3 Aside «Sections» rail — tinted gradient + яркий active
- **Файлы**: `/web/src/components/ConfigBuilder/TocRail.tsx` (или где рендерится `<aside class="bg-surface-1 border-r">`).
- **Что**:
  - Aside bg: `bg-gradient-to-b from-brand/4 via-surface-1 to-brand-alt/4` — вертикальное «дыхание»
  - Активный пункт: сейчас `bg-surface-3` → заменить на `bg-brand-weak/50 text-brand-strong border-l-2 border-brand`. Активная секция светится брендом, не становится светло‑серой.
  - Dot‑маркеры слева от пунктов: серый = не посещено, `bg-brand-alt/60` = посещено, `bg-ok` = валидно, `bg-err` = ошибка. Значения уже приходят из `groupValidity`.
  - Добавить внизу aside небольшую легенду (tooltip или подпись: `ok · warn · err`) — чтобы цвет точек читался.
- **Сложность**: M.

### 0.5.4 Nested collapsible (Qlora, Global Hyperparameters, etc.) — слоистая иерархия
- **Файлы**: `/web/src/components/ConfigBuilder/FieldRenderer.tsx` (рендер вложенного group‑блока, где сейчас `rounded border border-line-1 bg-surface-2`).
- **Что**:
  - Корневой блок группы: `border-l-2 border-brand-alt/40` (вертикальная фиолетовая полоска)
  - Header раскрывалки **при раскрытии**: `bg-gradient-to-r from-brand-alt/10 to-transparent`
  - Тело раскрытой подсекции: `bg-surface-1` (темнее родителя) — создаёт ощущение «колодца», а не плоского наклеенного прямоугольника
- **Сложность**: S.

### 0.5.5 Input row — focus brand glow
- **Файлы**: `FieldRenderer.tsx` (рендер `LabelledRow`).
- **Что**: на focus любого input'а внутри ряда → у label‑колонки (левой) `bg-brand-weak/10 border-brand/30`. Реализация: через `:has(:focus-within)` в CSS, если поддерживается (Chrome 105+, OK для dev‑tool). Fallback — state через `useRef + onFocus/onBlur`.
- **Сложность**: S.

### 0.5.6 Dirty / Saving / Saved status — брендовый
- **Файлы**: `ConfigTab.tsx:86-93` + `:170`.
- **Что**:
  - `Unsaved changes`: `text-brand-strong` + маленькая пульсирующая точка `bg-brand animate-pulse-ring`
  - `Saving…`: `text-info` + `Spinner`
  - `Saved`: `text-ok` + `CheckIcon`, держать 2 сек затем затухание
  - `Validating…`: `text-ink-3` (остаётся нейтральным, не бренд)
- **Сложность**: S.

### 0.5.7 Preset applied / Validate success — flash
- **Файлы**: `DiffBadge.tsx`, `ValidationBanner.tsx`.
- **Что**:
  - После Apply preset: `DiffBadge` 3 секунды носит `shadow-[0_0_20px_rgba(237,72,127,0.35)]` → fade out. Анимация через Tailwind `transition-shadow duration-[1500ms]` + `useEffect` таймер.
  - Успешная валидация (`result.ok && !hasWarnings`): banner 800ms с `shadow-[0_0_40px_rgba(74,222,128,0.25)]` → затухание.
- **Сложность**: S.

### 0.5.8 Error fields — усиленный ring
- **Файлы**: `FieldRenderer.tsx` (рендер input'а с `fieldErrors[path]`), `ValidationContext.tsx` (`useFieldStatus`).
- **Что**: текущий `border-err` → заменить на `border-err ring-2 ring-err/25 bg-err/[0.03]`. Ошибки должны «прыгать».
- **Сложность**: S.

**Verify Stage 0.5** (преview_*):
- `preview_screenshot` до/после на разделе Model — видна hero‑карточка и accent‑слой слева у h3.
- `preview_click` в aside по другим секциям → активный пункт светится бордо, а не серый.
- `preview_eval` → поменять любое поле → статус в правом верху `text-brand-strong` с пульсацией.
- `preview_eval` — спровоцировать ошибку (bad type/range) → поле с `ring-2 ring-err/25`.
- Проверить: политика «brand не течёт в inputs» — сами `<input>` остаются `surface-1 border-line-1`, brand только на ряду/фокусе.

---

## Stage 1 — P0 Блокеры (отдельный PR)

### 1.1 Custom Tailwind breakpoints под масштаб 113%
- **Файлы**: `/web/tailwind.config.ts` (добавить `theme.extend.screens`).
- **Что**: прописать `{ sm: '566px', md: '680px', lg: '907px', xl: '1133px', '2xl': '1359px' }` (= default × 1.13, округлено). В `FRONTEND_GUIDELINES.md` пояснить: `lg:` ≈ 1024 CSS px.
- **Риск**: существующие `sm:`/`md:` классы сдвигаются на один шаг. После правки прогнать grep по `web/src` и визуально проверить основные экраны.
- **Сложность**: S.

### 1.2 Desktop‑only gate (`<1024px`)
- **Файлы**: новый `/web/src/components/DesktopOnlyGate.tsx`; обернуть `ProjectDetailPage` в `/web/src/pages/ProjectDetail.tsx`.
- **Что**: хук `useViewportWidth()` на `matchMedia('(min-width:1024px)')`; при false — plain‑текст «Use a wider screen (≥1024px) — this is a desktop dev tool». Без layout shift.
- **Сложность**: S.

### 1.3 Вынести `useClickOutside` + починить PresetDropdown при переключении Form/YAML
- **Файлы**: новый `/web/src/hooks/useClickOutside.ts`; рефакторинг `PresetDropdown.tsx:17-32`, `HelpTooltip.tsx:16-31`.
- **Что**: единая сигнатура `useClickOutside(ref, enabled, onOutside)`. В `PresetDropdown` добавить `useEffect` с зависимостью на `view` (из пропса/контекста) — при смене режима принудительно закрывать.
- **Сложность**: S.

### 1.4 `Save` — объяснимое disabled
- **Файлы**: `ConfigTab.tsx:222-232`.
- **Что**: мemo `disabledReason`: `!dirty → 'No changes to save'`; `validationResult && !ok → 'Fix validation errors first'`; pending → `'Saving…'`. Оборачиваем кнопку в `<span title={reason}>` **и** рендерим inline‑caption под рядом кнопок (`text-2xs text-ink-3`) — не только hover.
- **Сложность**: S.

### 1.5 «Dataset — coming soon» — убрать как required
- **Файлы**: `/web/src/components/ConfigBuilder/schemaUtils.ts` (там сейчас `comingSoon` override), `FieldRenderer.tsx` (рендер этого состояния).
- **Что**: оставить disabled инпут‑placeholder, **убрать синтетическую `required`‑метку** (чтоб валидация не орала), показать muted info‑баннер «Dataset picker coming soon — configure via YAML for now».
- **Сложность**: S.

### 1.6 Help tooltip — измерение и флип при оверлапе
- **Файлы**: `/web/src/components/ConfigBuilder/HelpTooltip.tsx:55-62`.
- **Что**: после открытия прочесть `getBoundingClientRect()` → если обрезается справа/снизу, переключить на `right-0` / `bottom-full`. ~20 строк, без Floating UI.
- **Сложность**: S.

### 1.7 ValidationBanner — крупная кнопка «Open Settings» для секретов
- **Файлы**: `/web/src/components/ConfigBuilder/ValidationBanner.tsx:84-92`.
- **Что**: когда среди failures есть `group === SETTINGS_JUMP_TARGET`, рендерить в раскрытой области **отдельным первым рядом** `btn-ghost` «Open project Settings → (N secrets missing)». Мелкая стрелка per‑row остаётся.
- **Сложность**: S.

**Verify Stage 1** (через preview_*):
- `preview_resize` 800×600 → скрин, видим gate; 1440 → Config загружается.
- `preview_click` на preset dropdown trigger, потом click вне → dropdown закрыт.
- `preview_eval` assert `Save` имеет `title="No changes to save"` при `!dirty`.
- `preview_click` на «2 issues to fix», assert в DOM есть `btn-ghost` с текстом «Open project Settings».

---

## Stage 2 — P1 ML‑поток (отдельный PR)

### 2.1 Provider status chip (+ при необходимости backend endpoint)
- **Файлы**: новый `/web/src/components/ConfigBuilder/ProviderStatusChip.tsx`; врезать в `TrainingProviderField.tsx:55`, `InferenceProviderField.tsx`. Если endpoint отсутствует — новый `GET /api/v1/providers/{id}/health` в `/src/api/routers/*`.
- **Что**: chip `pill-ok "✓ key set"` или `pill-err "✗ HF_TOKEN missing"` рядом с селектором провайдера. Клик → Settings. Повторно использует `deriveGroupValidity` + `SETTINGS_JUMP_TARGET`.
- **Сложность**: M.

### 2.2 Client‑side валидация диапазонов/enum
- **Файлы**: `/web/src/components/ConfigBuilder/ValidationContext.tsx` (добавить поле `clientErrors`), `FieldRenderer.tsx` (сайты scalar‑рендера).
- **Что**: хук `useClientFieldValidation(node, value)` возвращает `string | null` на основе `node.minimum/maximum/enum/pattern` из JSON schema. В `useFieldStatus`: `serverErr ?? clientErr`. Моментальный красный border без ожидания `/validate`.
- **Сложность**: M.

### 2.3 Strategies как визуальная цепочка фаз
- **Файлы**: новый `/web/src/components/ConfigBuilder/StrategiesChainField.tsx`; регистрация в `FieldRenderer.tsx` `CUSTOM_FIELD_RENDERERS['training.strategies']`.
- **Что**: горизонтальный flex‑контейнер из карточек фаз: `Phase N` бейдж + `strategy_type` селектор + `Dataset` + collapsible details. Между карточками SVG‑стрелка. Кнопки `+ Add phase`, `× remove`. Фоллбэк на существующий `ArrayField`, если schema сломана.
- **Сложность**: L.

### 2.4 Preset diff‑preview modal (перед применением)
- **Файлы**: `/web/src/components/ConfigBuilder/PresetDropdown.tsx:37-46`; новый `PresetPreviewModal.tsx`.
- **Что**: вместо мгновенного `onLoad` — открыть модалку с списком полей, которые изменятся (через `deepDiff(current, preset)`). Две кнопки: `Cancel` / `Apply preset`. Существующий dirty‑`window.confirm` переезжает внутрь модалки.
- **Сложность**: M.

### 2.5 «Compare with last run»
- **Файлы**: `ConfigTab.tsx:213-233` (кнопка рядом с Save); новый хук `useProjectLastRunConfig(projectId)` в `/web/src/api/hooks/useProjects.ts`.
- **Что**: `GET /api/v1/projects/{id}/config/versions` → берём самый свежий snapshot → открываем ту же модалку, что и 2.4, но baseline = last run, read‑only. `btn-ghost` стиль.
- **Сложность**: M.

### 2.6 «Use recommended» inline‑chips для LoRA/QLoRA
- **Файлы**: constants в `/web/src/lib/loraRecommendations.ts`; расширить `CUSTOM_FIELD_RENDERERS` для `training.strategies.*.qlora.r`, `.qlora.lora_alpha` (или инъекция на уровне `StrategiesChainField`).
- **Что**: chip рядом с инпутом `Recommend 7B: r=16, α=32`; клик записывает оба значения разом.
- **Сложность**: S.

**Verify Stage 2**:
- `preview_eval` — при `value=0, node.minimum=1` виден red border без вызова `/validate`.
- `preview_click` по preset → появляется modal (`preview_snapshot` видит role=dialog) → Apply → форма мутирует, модалка закрыта.
- `preview_network` — `GET /config/versions` возвращает 200 на открытии Compare.

---

## Stage 3 — P2 Полировка + стайлгайд (отдельный PR)

### 3.1 Заменить YAML‑editor на CodeMirror 6
- **Файлы**: новый `/web/src/components/YamlEditor/CodeMirrorYaml.tsx`, удалить `/web/src/components/YamlEditor.tsx` и `/web/src/lib/yamlTokens.ts`.
- **Что**: `@codemirror/lang-yaml` + dark theme. Получаем folding, line numbers, подсветку, линтер. Теряем precision‑выравнивание overlay'а (оно было проблемное).
- **Сложность**: L.

### 3.2 Section help panel вместо per‑field `?`
- **Файлы**: новый `/web/src/components/ConfigBuilder/SectionHelpPanel.tsx`; в `FieldRenderer.tsx` оставить inline‑`?` только для однострочных подсказок, а на уровне секции — новый `?` открывает panel.
- **Что**: правая плавающая панель (`createPortal`, sticky `right:0`, `w-[360px]`, z-40), собирает descriptions полей секции + narrative; close по Esc / outside.
- **Сложность**: L.

### 3.3 Типографическая шкала — 4 токена
- **Файлы**: `/web/tailwind.config.ts` (`theme.extend.fontSize`); codemod‑pass по `text-[13px]`, `text-[0.6rem]`, `text-[0.65rem]`, `text-2xs` во всём `/web/src`.
- **Что**: токены `caption` (11px), `body` (13px), `h3` (15px), `h2` (18px). Прописать в `FRONTEND_GUIDELINES.md`.
- **Сложность**: S.

### 3.4 a11y‑pass
- **Файлы**: `FieldRenderer.tsx` `LabelledRow` (около 115–120), `FieldAnchor.tsx:38`, `HelpTooltip.tsx`, все места с декоративными dots/emoji.
- **Что**: `id` у input'а (`cfg-${path}`) + `<label htmlFor>`; `aria-label` уникальный на каждом `?` (из названия поля); замена emoji `🔗` на `<LinkIcon>`; `aria-hidden` на декоре.
- **Сложность**: M.

### 3.5 Breadcrumbs + убрать дублирующий h1
- **Файлы**: новый `/web/src/components/Breadcrumbs.tsx`; `ProjectDetail.tsx:123-127`.
- **Что**: `Projects / <project.name> / Config` над card'ом. Заголовок `project.name` в card урезать до `text-sm font-medium`.
- **Сложность**: S.

### 3.6 Dirty‑state `beforeunload` + route guard
- **Файлы**: `ConfigTab.tsx` (effect).
- **Что**: `beforeunload` при `dirty`; React Router v6 — использовать `unstable_usePrompt` для внутренней навигации.
- **Сложность**: S.

### 3.7 `FRONTEND_GUIDELINES.md`
- **Файл**: новый `/web/FRONTEND_GUIDELINES.md`.
- **Разделы**:
  - Design tokens: `surface-0..4`, `ink-1..4`, `line-1..2`, `brand`/`brand-alt`, semantic `ok/warn/err/info` — значения и когда применять.
  - **Brand‑usage policy (явно, с примерами):**
    - `brand` / `brand-alt` **разрешено** на: CTAs (`.btn-primary`), активных состояниях навигации, focus‑ring, selection, required `*`, accent‑полосках на section header'ах, active section в aside (`bg-brand-weak/50`), dirty‑state статусе, tinted gradients alpha ≤15% на hero‑card и header‑областях, glow‑flash при success‑моментах (preset applied, validate ok), chain‑phase coloring для Strategies.
    - `brand` / `brand-alt` **запрещено** на: заливке `<input>`, заливке карточек (кроме одной `.card-hero` per screen), заливке таблиц, бордерах диалогов (там `line-2`), обычных hover‑состояниях (там `surface-3`).
    - Правило «один `.card-hero` на экран».
    - Alpha guidance: tints ≤10% для фоновых акцентов, 15–20% для hover‑состояний activewea, 35–50% для ring/focus, 100% только для текста/иконок/line‑accents.
  - Типография: 4 токена (caption/body/h3/h2), шрифты Inter + JetBrains Mono, правило `113% html scale`.
  - Breakpoints: custom screens (см. 1.1), desktop‑only policy.
  - Формы: `LabelledRow` пропорции 220px / fluid, `*` required + `(optional)` слово, порядок `label | ? | input | <LinkIcon>`.
  - Кнопки: `btn-primary` только для главного Save/Run CTA (один на экран), `btn-ghost` для вторичных, outline для нейтральных.
  - Иконография: **только SVG из `/web/src/components/icons/`**, никаких emoji в UI‑chrome.
  - a11y‑чеклист: label‑for‑id, уникальные aria‑label, aria‑hidden на декоре, focus‑visible, contrast ≥4.5:1.
  - State: TanStack Query для server state, `useState` для local; `useReducer` только для сложных flow; никакого redux/zustand.
  - Как добавить custom field renderer: регистрация в `CUSTOM_FIELD_RENDERERS` (FieldRenderer.tsx) + сигнатура + пример.
  - **Visual hierarchy recipes**: как использовать `border-l-2 border-brand-alt/40`, tinted gradients `from-brand/8 to-transparent`, nested levels через surface swap (родитель `surface-2` → ребёнок `surface-1`).
- **Сложность**: S.

**Verify Stage 3**:
- `preview_eval` `document.querySelectorAll('label:not([for])').length === 0`.
- `preview_inspect` section help panel → `role="complementary"`, `aria-label` установлен.
- После codemod — `rg "text-\[13px\]" web/src` возвращает пусто.

---

## Critical files (сводка путей)

- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ProjectTabs/ConfigTab.tsx`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ConfigBuilder/ConfigBuilder.tsx`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ConfigBuilder/FieldRenderer.tsx`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ConfigBuilder/TocRail.tsx` (Stage 0.5.3)
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ConfigBuilder/ValidationBanner.tsx`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ConfigBuilder/ValidationContext.tsx`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ConfigBuilder/PresetDropdown.tsx`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ConfigBuilder/DiffBadge.tsx`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ConfigBuilder/HelpTooltip.tsx`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/ConfigBuilder/ArrayField.tsx`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/components/YamlEditor.tsx` (→ CodeMirror)
- `/Users/daniil/MyProjects/RyotenkAI/web/src/styles/globals.css`
- `/Users/daniil/MyProjects/RyotenkAI/web/tailwind.config.ts`
- `/Users/daniil/MyProjects/RyotenkAI/web/src/api/hooks/useProjects.ts`
- `/Users/daniil/MyProjects/RyotenkAI/src/api/routers/projects.py` (при необходимости `/providers/:id/health`)
- `/Users/daniil/MyProjects/RyotenkAI/web/FRONTEND_GUIDELINES.md` (новый)

---

## End‑to‑end verification

После каждого Stage:
1. `mcp__Claude_Preview__preview_start name=web-frontend` → открыть `http://localhost:5173/projects/helixql-nl2hql-v7-mini/config`.
2. Скриншоты до/после ключевых состояний: default, Load preset open, Validate clicked, expand issues, switch to YAML, edit field → dirty, Save click.
3. `preview_resize` 1024, 1280, 1440, 1920 — layout держится; 900 — gate.
4. `preview_inspect` проверки из каждого Stage выше.
5. `preview_console_logs level=error` — пусто.
6. Ручная прогонка: `npm run build` + `npm run test` в `/web`.
7. `ruff check .` + `mypy .` если трогали бэкенд (Stage 2.1 `/providers/:id/health`).
