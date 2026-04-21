# Plan: Theme migration — burgundy/pink → cool violet (RunPod-style flat dark)

## Context

Текущая тема — тёмная, но «moody»: тёплый бордо-розовый бренд (`#e63570` / `#f25088`) + ambient радиалы бордо top-left / violet bottom-right на `#root::before` + burgundy hero band поверх карточки — в сумме даёт «кровь, туман, warning». Эталон, который нравится пользователю — **RunPod dashboard**: плоский чистый чёрный фон, один холодный пурпурный акцент (`#7c3aed`-ish), никаких градиентов в углах, один h1 обычным белым.

**Intent**: переехать на cool-violet бренд, удалить ambient-градиенты + hero band, сохранив identity (тот же акцент на active states, CTA, логотипе). Без кардинального переписывания компонентов.

Это **не** полный перекрас UI. Это **token swap + очистка пары CSS-утилит**. Благодаря тому, что 95% компонентов пользуются токенами (`bg-brand`, `text-brand-alt`, `shadow-glow-brand` и т.п.), смена значений в `tailwind.config.ts` + `globals.css` автоматически перекрашивает 130+ мест, не трогая ни одного `.tsx`.

---

## Reuse — existing tokens + structure

Менять значения, но не структуру:
- **Все цветовые токены** (`brand`, `brand-strong`, `brand-weak`, `brand-alt`) в `web/tailwind.config.ts:59-63` — останутся с теми же именами, поменяются hex-ы. Все `text-brand`, `bg-brand`, `border-brand`, `focus:border-brand` (28 мест в inputs), `text-brand-alt` (15+ мест) перекрасятся автоматически.
- **`gradient-brand` / `gradient-brand-soft`** (tailwind.config.ts:73-78) — остаются как токены для логотипа/accent-иконок (7 файлов: Sidebar, LaunchModal, ProjectCard, CommandPalette, SelectField, HFModelField, UnionField). Меняем только hex-ы gradient-stop'ов.
- **`shadow-glow-brand`, `shadow-inset-accent`** — то же.
- **`ringColor.DEFAULT`** — то же.
- **Классы `.gradient-text`, `.card-hero`, `.card-hero-ambient`, `.btn-primary`, `.nav-item-active`, `::selection`, `:focus-visible`** в `globals.css` — остаются как точки редактирования. Компоненты их имена не меняют.

Не трогаем (никак):
- AppShell, TopBar, Sidebar структуру
- Semantic colors (`ok`, `warn`, `err`, `info`, `idle`)
- Surface / ink / line tokens — уже нейтральные, менять не надо
- Custom breakpoints, fontFamily, fontSize, keyframes
- 30+ component files — все под токенами

---

## Новая палитра

| Token | Текущее | Новое | Roles |
|---|---|---|---|
| `brand` | `#ed487f` (hot pink) | **`#8b5cf6`** (violet-500) | primary CTA, active tab underline, required `*` |
| `brand-strong` | `#f76398` | **`#7c3aed`** (violet-600) | btn-primary hover |
| `brand-weak` | `#44182b` (dark burgundy) | **`#1d1341`** (dark violet) | `bg-brand-weak` translucent bg (TocRail, FieldRenderer) |
| `brand-alt` | `#b8a1fb` (violet-300) | `#b8a1fb` (оставить) | secondary accent, tag labels |
| `ringColor.DEFAULT` | `#f25088` | **`#a78bfa`** (violet-400) | focus ring everywhere |

Почему violet-500/600 как якорь — это holy Tailwind violet stack, выглядит профессионально «techno», не «emotional».

---

## Stage 1 — `web/tailwind.config.ts`

Одним куском правок:

```ts
colors: {
  // ...
  'brand':        '#8b5cf6',   // violet-500 (was #ed487f)
  'brand-strong': '#7c3aed',   // violet-600 (was #f76398)
  'brand-weak':   '#1d1341',   // dark violet (was #44182b)
  'brand-alt':    '#b8a1fb',   // unchanged
  // ...
},
backgroundImage: {
  'gradient-brand':
    'linear-gradient(135deg, #7c3aed 0%, #b8a1fb 100%)',      // was #e63570 → #a78bfa
  'gradient-brand-soft':
    'linear-gradient(135deg, rgba(124,58,237,0.22) 0%, rgba(184,161,251,0.18) 100%)',
},
boxShadow: {
  'glow-brand':    '0 0 28px rgba(139, 92, 246, 0.42)',       // was rgba(230,53,112,0.48)
  'card':          '0 1px 0 rgba(255,255,255,0.04) inset, 0 6px 18px rgba(0,0,0,0.35)',  // unchanged
  'inset-accent':  'inset 2px 0 0 #8b5cf6',                   // was #e63570
},
ringColor: {
  DEFAULT: '#a78bfa',                                         // was #f25088
},
```

Также обновить комментарий в шапке файла (строки 3-15) — «burgundy → violet brand» на «cool violet brand», убрать упоминания про «warm».

**Сложность**: S.

---

## Stage 2 — `web/src/styles/globals.css`

### 2.1 — `#root::before` ambient: УДАЛИТЬ

Строки 35-44. Целиком снести блок (оставить `#root { position: relative; isolation: isolate }`). RunPod — плоский фон. Если позже захотим cool-fog, вернём с `rgba(139,92,246,0.05)` и `rgba(167,139,250,0.04)` — но по умолчанию убираем совсем.

### 2.2 — `::selection` (строки 47-50)

```css
::selection {
  background: rgba(139, 92, 246, 0.38);   /* violet, was #e63570 */
  color: #ffffff;
}
```

### 2.3 — `:focus-visible` (строки 60-64)

```css
:focus-visible {
  outline: 2px solid #a78bfa;             /* was #f25088 */
  outline-offset: 2px;
  border-radius: 4px;
}
```

### 2.4 — `.gradient-text` (строки 68-73)

По запросу пользователя — **удаляем класс целиком**. Все usage-сайты (`Overview.tsx:14`, `LaunchPage.tsx`, `Sidebar.tsx`) заменяем на плоский `text-ink-1` (как RunPod «Aloha, Даниил!»). Это часть Stage 3.

### 2.5 — `.gradient-text-burgundy` (строки 75-83)

Удаляем. Единственный usage — `ProjectDetail.tsx` h1, заменяем на `text-ink-1`.

### 2.6 — `.card-hero` / `.card-hero-ambient` (строки 100-125)

Bunny hero band полностью удаляем. Оставляем просто:
```css
.card-hero {
  @apply bg-surface-2 border border-line-2 rounded-lg shadow-card relative overflow-hidden;
  /* background-image: none — flat surface */
}
.card-hero-ambient {
  @apply border border-line-2 rounded-lg shadow-card relative overflow-hidden;
  background-color: transparent;
}
```

Либо — поскольку теперь оба класса делают по сути одно и то же минус бг-цвет — оставить один `.card-hero` и в `ProjectDetail.tsx` использовать обычный `<div>` с прозрачным бг (как сейчас и есть с `!rounded-none !border-0`).

### 2.7 — `.btn-primary` (строки 128-143)

**Механику переливания ОСТАВЛЯЕМ** (user её любит). Меняем только hex-ы цветовых stop'ов — тёплый бордо-виолет на холодный виолет-моно:

```css
.btn-primary {
  @apply inline-flex items-center justify-center gap-2 px-3.5 py-2 rounded-md text-sm font-medium text-white relative overflow-hidden;
  /* 3-stop violet gradient: deep → light → deep. 220% bg-size + hover
     bg-position shift дают тот же shimmer, но в cool-violet палитре. */
  background-image: linear-gradient(120deg, #7c3aed 0%, #b8a1fb 50%, #7c3aed 100%);
  background-size: 220% 100%;
  background-position: 0% 50%;
  box-shadow: 0 1px 0 rgba(255,255,255,0.18) inset, 0 4px 18px rgba(139, 92, 246, 0.42);
  transition: transform .08s ease, filter .15s ease, background-position .9s ease, box-shadow .3s ease;
}
.btn-primary:hover:not(:disabled) {
  background-position: 100% 50%;
  box-shadow: 0 1px 0 rgba(255,255,255,0.28) inset, 0 8px 26px rgba(139, 92, 246, 0.55);
}
.btn-primary:active  { transform: translateY(1px); }
.btn-primary:disabled { opacity: .5; cursor: not-allowed; filter: none; }
```

Shimmer, glow и transitions не трогаем — только hex-ы `#e63570 → #7c3aed`, `#a78bfa → #b8a1fb`, и glow `rgba(230,53,112, …) → rgba(139,92,246, …)`.

### 2.8 — `.nav-item-active` (строки 176-179)

```css
.nav-item-active {
  @apply text-ink-1 bg-surface-2;
  box-shadow: inset 2px 0 0 #8b5cf6;       /* was #e63570 */
}
```

### 2.9 — `.field-attention-pulse` (добавлено недавно, желтый pulse)

Оставляем как есть. Желтый — semantic attention, не brand.

### 2.10 — `.pill-*` status pills

Оставляем как есть — они semantic (green/amber/red/blue/violet), не brand. `pill-skip` уже фиолетовый.

**Сложность**: M (несколько блоков, но никакой логики — только hex-ы).

---

## Stage 3 — Заменить `gradient-text` на плоский `text-ink-1`

Гренд-text больше нет. Три места:

| Файл | Строка | Заменить |
|---|---|---|
| `web/src/pages/Overview.tsx` | ~14-16 | `gradient-text` → `text-ink-1` |
| `web/src/pages/LaunchPage.tsx` | (one h1) | то же |
| `web/src/components/Sidebar.tsx` | logo caption | то же |
| `web/src/pages/ProjectDetail.tsx` | `gradient-text-burgundy` h1 | `text-ink-1` |

**Сложность**: S.

---

## Stage 4 — `HealthHero.tsx` hard-coded pink gradient

Файл: `web/src/components/HealthHero.tsx:~155`.

Сейчас: `card-hero !bg-none bg-[linear-gradient(180deg,rgba(248,113,113,0.10),rgba(198,48,107,0.04)_55%,transparent)]` — смешивает red-400 (`#f87171`) и pink (`#c6306b`). С новым flat-dark это не вяжется.

**Заменить**: убрать второй stop (pink), оставить только faint red washat top — для error-состояний hero:
```tsx
'card-hero !bg-none bg-[linear-gradient(180deg,rgba(248,113,113,0.10),transparent_70%)]'
```

Для ok-состояния — без градиента вообще, flat card. Проверю в файле и подкорректирую.

**Сложность**: S.

---

## Не трогаем

- Semantic colors (ok/warn/err/info/idle) — не меняем
- Surface / ink / line tokens — нейтральные
- Forms, ValidationContext, CodeMirror, routing, scroll architecture
- Компоненты, использующие только токены — перекрасятся сами
- `.pill-skip` — уже violet

---

## Critical files

| Файл | Что меняем | Stage |
|---|---|---|
| `web/tailwind.config.ts` | brand tokens (4 hex), gradients, shadows, ring | 1 |
| `web/src/styles/globals.css` | #root::before remove, selection, focus, gradient-text remove, card-hero flat, btn-primary flat, nav-item-active hex | 2 |
| `web/src/pages/Overview.tsx` | `gradient-text` → `text-ink-1` | 3 |
| `web/src/pages/LaunchPage.tsx` | то же | 3 |
| `web/src/components/Sidebar.tsx` | то же | 3 |
| `web/src/pages/ProjectDetail.tsx` | `gradient-text-burgundy` → `text-ink-1` | 3 |
| `web/src/components/HealthHero.tsx` | pink stop в hero gradient | 4 |

Всего: **~7 файлов**, суммарно ~40 строк кода.

---

## Verification

1. `preview_screenshot` Overview → h1 «Overview» плоский белый, no pink glow on logo anymore.
2. `preview_screenshot` ProjectDetail → h1 проекта плоский белый, нет burgundy band сверху, нет fog в углах. Tab underline под активной — violet. Save кнопка — solid violet, без shimmer/glow.
3. `preview_inspect .btn-primary` → `background-image` содержит violet-only stops (`#7c3aed` / `#b8a1fb`), нет pink hex в computed style. Hover по-прежнему сдвигает `background-position` → shimmer работает в cool-violet.
4. `preview_inspect #root::before` → **element does not exist** (or computed `content: none`).
5. `preview_inspect [required-asterisk]` (любой `*` в FieldRenderer) → `color: rgb(139, 92, 246)` (violet), а не pink.
6. `preview_inspect .nav-item-active` в Sidebar → `box-shadow: inset 2px 0px 0px rgb(139, 92, 246)`.
7. Сравнить визуально с RunPod скриншотом — плоско, один пурпур, ничего кровавого.
8. `npx tsc --noEmit` в `web/` — типы чистые (нет новых import'ов, только CSS/hex edits + className swap).
9. `npm run build` в `web/` — сборка проходит (tailwind JIT успешно резолвит новые значения).
10. Проверить focus-ring любого input — должен стать viol-400 (`#a78bfa`), а не pink.
