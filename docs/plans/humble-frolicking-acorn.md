# Config page — видимые границы между навигацией, формой и полями

## Context

На странице проекта `Config` всё визуально сливается: left-rail с разделами, toolbar (Form/YAML, Load preset), Validation banner и сама форма живут на одной поверхности `surface-2`. Пользователь жалуется: "нету границ, все в 1м блоке, и поля и вкладки".

Код-аудит обнаружил три причины:
1. **TocRail и контент на одной поверхности** (`surface-2`) разделены только `gap-6` — нет цветового перепада, нет линии.
2. **Инверсия вложенности:** `CollapsibleCard` рисуется на `surface-1` внутри `surface-2`-родителя, т.е. вложенные группы становятся **темнее** вместо того чтобы подниматься. Material 3 прямо называет это антипаттерном.
3. **Toolbar без chrome-разделителя** — Form/YAML toggle и Preset dropdown сидят на той же поверхности что и поля, превращаются в часть формы.

Дизайн-ресерч (Polaris, Material 3, Carbon, Grafana, Notion) даёт ранжированный набор решений: **цветовая смена фона rail'а важнее тонкой линии-разделителя**, **вложенные карточки должны совпадать с родителем по фону и использовать border**, **page chrome отделяется одной hairline-линией**.

Цель: сдержанный набор поверхностных правок (не редизайн), который устраняет "один блок" и при этом не добавляет шума.

## Изменения

### 1. `web/src/components/ConfigBuilder/ConfigBuilder.tsx` — rail получает свою поверхность

Заменить grid (строка 96) на двухколоночный layout с полным bleed до краёв внешней Card. Rail на `surface-1`, контент на `surface-2` (унаследован).

```tsx
<div className="grid grid-cols-[208px_1fr] -mx-5 -my-5">
  <aside className="min-w-0 bg-surface-1 border-r border-line-1 px-3 py-5">
    <TocRail schema={schema} active={activeKey} onSelect={selectGroup} validity={groupValidity} />
  </aside>
  <div className="min-w-0 space-y-4 px-5 py-5">
    <section id={`cfg-${activeKey}`} className="min-w-0 scroll-mt-24">
      {renderActive()}
    </section>
    <FieldSearchOmniBox schema={schema} hashPrefix={hashPrefix} />
  </div>
</div>
```

*Почему:* `-mx-5 -my-5` сбрасывает `p-5` внешней Card ([ProjectDetail.tsx:88](web/src/pages/ProjectDetail.tsx:88)) чтобы rail и контент смыкались без плавающего gutter. `border-r border-line-1` — защитная hairline на случай, если глазу недостаточно только цвета. Ширина 208px (кратно 8).

### 2. `web/src/components/ConfigBuilder/TocRail.tsx` — active state на +1 ступень

Текущая активная кнопка: `bg-surface-2 border-l-2 border-brand` — после того как rail ушёл на `surface-1`, это нормально читается, но `hover:bg-surface-2` теперь совпадает с активной. Поэтому active поднимается ещё на шаг.

- Строка 44–45: active → `bg-surface-3 text-ink-1 border-l-2 border-brand -ml-0.5 pl-[0.625rem]`
- Строка 45: inactive hover → `hover:bg-surface-2` (было `hover:bg-surface-2`, теперь это меньше чем активное)
- Строка 29: убрать `pr-2` с `<nav>` — padding теперь на `aside`.

*Почему:* чтобы сохранить 3-ступенчатую лестницу elevation'а внутри rail'а (`surface-1` → hover `surface-2` → active `surface-3`).

### 3. `web/src/components/ConfigBuilder/FieldRenderer.tsx` — `CollapsibleCard` без инверсии

Строка 679:

```tsx
<div className="relative rounded border border-line-1 bg-surface-2">
```

Строка 693 (hover внутри header):

```tsx
className="flex items-center gap-2 px-4 py-2.5 cursor-pointer hover:bg-surface-3/40 transition-colors"
```

*Почему:* вложенная карточка теперь на той же поверхности что и родитель (`surface-2`), группировка передаётся border'ом + заголовком + accent-bar'ом — без цветового провала. Соответствует Material 3 "avoid nested fills".

### 4. `web/src/components/ConfigBuilder/FieldRenderer.tsx` — section heading (строки 363–382)

Depth-0 header получает размер, описание и hairline снизу:

```tsx
if (depth === 0) {
  return (
    <div className="space-y-5">
      <header className="pb-3 border-b border-line-1">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold text-ink-1">{label}</h3>
          <HelpTooltip text={description} />
        </div>
        {description && (
          <p className="text-xs text-ink-3 mt-1 leading-relaxed">{description}</p>
        )}
      </header>
      <ObjectFields
        root={root}
        node={node}
        value={fallback}
        onChange={onChange}
        depth={depth + 1}
        pathPrefix={path}
        hashPrefix={hashPrefix}
      />
    </div>
  )
}
```

*Почему:* даёт content-pane свой явный header'овский блок (паттерн Notion/Shopify). Одна hairline, не N per row — не мозолит глаза.

### 5. `web/src/components/ProjectTabs/ConfigTab.tsx` — chrome вокруг toolbar + validation banner

В строках 122–172 обернуть блок `[toolbar, ValidationBanner]` одной chrome-областью:

```tsx
<div className="pb-3 border-b border-line-1 mb-3 space-y-3">
  <div className="flex items-center gap-2">{/* Form/YAML, Preset, statusLine — без изменений */}</div>
  <ValidationBanner ... />
</div>
```

*Почему:* toolbar + banner читаются как "page chrome над формой", форма — как основное содержимое. Одна линия, не две. Validate/Save внизу оставить как есть — добавление там верхней линии over-box'ило бы.

## Критические файлы

- [web/src/components/ConfigBuilder/ConfigBuilder.tsx](web/src/components/ConfigBuilder/ConfigBuilder.tsx)
- [web/src/components/ConfigBuilder/TocRail.tsx](web/src/components/ConfigBuilder/TocRail.tsx)
- [web/src/components/ConfigBuilder/FieldRenderer.tsx](web/src/components/ConfigBuilder/FieldRenderer.tsx) (строки 363–382 и 679, 693)
- [web/src/components/ProjectTabs/ConfigTab.tsx](web/src/components/ProjectTabs/ConfigTab.tsx)
- [web/src/pages/ProjectDetail.tsx](web/src/pages/ProjectDetail.tsx) — НЕ редактируется, но `p-5` на строке 88 связан negative margin'ом из правки #1; оставить комментарий в ConfigBuilder.

## Verification (в preview)

URL: `http://localhost:5173/projects/<any>/config`

С помощью `preview_inspect`:
- `aside` → `background-color: rgb(31, 34, 38)` (surface-1), `border-right-width: 1px`
- `aside + div` (контент) → `background-color: rgb(38, 42, 47)` (surface-2, унаследован)
- `nav button[class*="border-brand"]` → `background-color: rgb(47, 51, 56)` (surface-3)
- `header h3.text-lg` → `font-size: 18px`, `border-bottom-width` на родителе `<header>` = 1px
- В ConfigTab первая `<div>` с toolbar → `border-bottom-width: 1px`, `padding-bottom: 12px`

Сценарии через `preview_click`:
1. Открыть `config` → кликнуть в TocRail по `Training` → нет сдвига layout, active переезжает в rail.
2. В Training раскрыть `qlora` (вложенный `CollapsibleCard`) → карточка **не темнее** родительского пана; в `preview_inspect` на ней `background-color: rgb(38, 42, 47)`.
3. Перейти `providers` (использует `ProviderPickerField`, не `FieldRenderer` depth-0): убедиться, что страница не сломалась, просто нет нового section-header'а — ожидаемо, это custom renderer.
4. Открыть `evaluation` → discriminated union с metrics → вложенные ветки читаются на ровной поверхности.
5. Toggle Form ↔ YAML → chrome-разделитель не прыгает, YAML editor стоит под ним.

Для визуальной проверки: `preview_screenshot` после каждого сценария.

## Риски

1. **Связь negative-margin и `ProjectDetail.p-5`.** Если когда-то Card padding изменится, правка #1 сломается. Митигация: inline-комментарий над грид-блоком с ссылкой на `ProjectDetail.tsx:88`.
2. **`CollapsibleCard` используется глубоко:** nested objects, discriminated unions ([FieldRenderer.tsx:385](web/src/components/ConfigBuilder/FieldRenderer.tsx:385)), array-of-objects в `ArrayField`. Проверить все через верификацию #2–#4.
3. **`ProviderPickerField` (custom renderer)** не получит новый section header — приемлемо, у него свой UI. Отметить в коммите.
4. **`FieldSearchOmniBox`** сидит в контент-пане; после bleed-to-edge его `bg-surface-1` input теперь совпадает с фоном rail'а — визуально не мешает, но если рядом со швом выглядит странно, поправим точечно.
5. **VersionsTab / InfoTab и т.д.** используют тот же outer Card с `p-5` — правки затрагивают только ConfigBuilder, остальные tab'ы не получают bleed. Ожидаемо.
