# Page Chrome Separation в ConfigTab

## Контекст

На странице проекта (`/projects/<id>/config`) в табе Config сейчас toolbar (переключатель Form/YAML + DiffBadge + PresetPickerModal + StatusPill), `ValidationBanner` и блок draft-restore визуально сливаются с формой ниже — всё живёт на одной поверхности `surface-2` без какой-либо разделительной линии. Пользователь жалуется: «нет границ, все в одном блоке — и поля, и вкладки».

Плановый документ [docs/plans/humble-frolicking-acorn.md §5](../../.claude/worktrees/jolly-lichterman-01f936/docs/plans/humble-frolicking-acorn.md) предлагает обернуть toolbar + `ValidationBanner` в явную «page chrome»-область, отделённую от формы одной тонкой hairline-линией (`border-b border-line-1`). Decision record «Page Chrome Separation» (proposed, confidence 60%) требует именно этого. Цель — чтобы пользователь читал toolbar + banner как «управление над страницей», а форму — как основное содержимое.

Изменение касается только одного файла и никак не трогает sibling-табы (ProviderConfigTab, IntegrationConfigTab), потому что там нет пары toolbar + ValidationBanner.

## Изменение

**Единственная правка:** `web/src/components/ProjectTabs/ConfigTab.tsx`, строка 244.

Сейчас:
```tsx
<div className="space-y-3">
```

Станет:
```tsx
<div className="pb-3 border-b border-line-1 space-y-3">
```

Всё остальное в файле остаётся как есть.

### Детали дизайна

1. **Без `mb-3`.** Плановый документ предлагает `pb-3 border-b border-line-1 mb-3 space-y-3`, но внешний враппер уже имеет `space-y-4` (строка 243), который даёт `margin-top: 16px` каждому следующему ребёнку. Добавление `mb-3` (12 px) сложится с этим в ~28 px воздуха между hairline и формой — это перебор. `pb-3 border-b border-line-1` достаточно: 12 px внутреннего отступа, тонкая линия, затем естественный 16-px промежуток от `space-y-4`.

2. **Draft-restore блок остаётся внутри chrome.** Блок «Local draft available» (строки 331-370) — это page-level state (восстановить локальный черновик), а не содержимое формы. Семантически он принадлежит chrome. К тому же он условный — в обычном случае его нет, и hairline сразу упирается в форму.

3. **Scope строго ConfigTab.tsx.** ProviderConfigTab и IntegrationConfigTab не трогаем — там нет переключателя Form/YAML + ValidationBanner, chrome-область нерелевантна.

## Критические файлы

- **Изменяется:** [`web/src/components/ProjectTabs/ConfigTab.tsx`](../../.claude/worktrees/jolly-lichterman-01f936/web/src/components/ProjectTabs/ConfigTab.tsx) — строка 244
- **Только для сверки (не менять):** [`web/src/components/ProjectTabs/ProviderConfigTab.tsx`](../../.claude/worktrees/jolly-lichterman-01f936/web/src/components/ProjectTabs/ProviderConfigTab.tsx), [`web/src/components/ProjectTabs/IntegrationConfigTab.tsx`](../../.claude/worktrees/jolly-lichterman-01f936/web/src/components/ProjectTabs/IntegrationConfigTab.tsx)
- **Evidence doc:** [`docs/plans/humble-frolicking-acorn.md`](../../.claude/worktrees/jolly-lichterman-01f936/docs/plans/humble-frolicking-acorn.md) §5 (строки 97-108)

Тестов, покрывающих layout ConfigTab, в коде нет — верификация ручная через preview.

## Verification

1. Запустить preview-сервер (`preview_start` с конфигурацией web-dev из `.claude/launch.json`), открыть `/projects/<any>/config`.

2. Через `preview_inspect` проверить на chrome-враппере (`.space-y-4 > .space-y-3`):
   - `border-bottom-width: 1px`
   - `border-bottom-style: solid`
   - `padding-bottom: 12px`
   - `margin-bottom: 0px` (подтверждает, что `mb-3` корректно опущен)

3. Через `preview_inspect` на форме (второй ребёнок `.space-y-4`): `margin-top: 16px` (штатный зазор `space-y-4`).

4. `preview_screenshot` во Form- и YAML-режимах: hairline занимает всю ширину content-pane, не смещается при переключении. Блок Save/Validate ниже визуально не изменился.

5. Если удастся вызвать draft-restore (сохранить локальный черновик и перезагрузить страницу): блок draft-restore помещается внутри chrome с 12-px зазором до hairline.

## Обязательный пост-шаг

После правки записать в Repowise новую ADR через `update_decision_records(action="create")`:

- `title`: "ConfigTab page-chrome separation"
- `status`: `active`
- `context`: краткое описание проблемы (toolbar + banner сливались с формой)
- `decision`: wrap в `pb-3 border-b border-line-1 space-y-3`, без `mb-3`
- `rationale`: single hairline = Notion/Shopify content-pane pattern; outer `space-y-4` уже держит вертикальный ритм
- `alternatives`: две линии (over-box), draft-restore снаружи (слишком большой зазор), `mb-3` буквально по плану (стекается до 28 px)
- `consequences`: 12 + 16 = 28 px зазор chrome→form; sibling-табы не затронуты
- `affected_files`: `["web/src/components/ProjectTabs/ConfigTab.tsx"]`
- `tags`: `["ui", "layout", "project-config"]`

Существующую proposed-ADR «Page Chrome Separation» (readme_mining, 60%) перевести в `superseded` со ссылкой на новую.
