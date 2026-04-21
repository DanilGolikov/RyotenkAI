# RyotenkAI Frontend Guidelines

This document is the single source of truth for how we write UI in `web/`.
Read it before opening a PR that touches styling, component layout, or
design tokens. When in doubt, follow these rules — and when you break
one of them, explain why in the PR description.

Scope: **desktop dev tool**. Min supported width is **1024 CSS px**
(`DesktopOnlyGate`). Mobile is out of scope.

---

## Stack

- **React 18 + TypeScript**, Vite 5.
- **State:** TanStack Query v5 for server state. Local UI state via
  `useState`; reach for `useReducer` only when transitions have ≥4
  states. No Redux / Zustand / MobX.
- **Routing:** React Router v6. Deep-links use URL hash for in-page
  anchors (e.g. `#project:training.qlora.r`).
- **Forms:** hand-written controlled components. No `react-hook-form`.
  The shape is schema-driven — we render from a JSON Schema fetched
  from `/config/schema`.
- **Styling:** Tailwind CSS 3. Design tokens in
  `tailwind.config.ts`. No CSS-in-JS. Component-level classes go in
  `globals.css` under `@layer components` when they're reused in ≥3
  places or contain non-atomic recipes (gradients, shadows).
- **Icons:** SVG from `/src/components/icons/`. No emoji in UI chrome.
- **Editor:** CodeMirror 6 for YAML. Monaco is banned (10× bundle).

---

## Design tokens

All tokens live in `web/tailwind.config.ts`.

### Surfaces — flat, near-black, ≈3% L steps

| Token | Value | Use |
|---|---|---|
| `surface-0` | `#181b1f` | App canvas (under `body`) |
| `surface-1` | `#1f2226` | Sidebar, input backgrounds, nested-section well |
| `surface-2` | `#262a2f` | Cards, section containers |
| `surface-3` | `#2f3338` | Hover/selected state |
| `surface-4` | `#3a3e44` | Popovers, floating menus |

**Rule:** sibling levels of hierarchy swap surfaces, never colours.
Parent on `surface-2`, its expanded child on `surface-1` (reads as a
"well"), etc. Borders stay hairline (`line-1`).

### Lines — hairline borders

| Token | Value | Use |
|---|---|---|
| `line-1` | `#2c3036` | Default borders |
| `line-2` | `#3c4046` | Dialog borders, raised cards |

Never use brand colour as a static border. Reserved for active/focus
states.

### Ink — text colours

| Token | Value | Meaning | Contrast vs `surface-0` |
|---|---|---|---|
| `ink-1` | `#fafafa` | Primary text | ~17:1 |
| `ink-2` | `#d4d4d8` | Secondary text, labels | ~12:1 |
| `ink-3` | `#a1a1aa` | Captions | ~7:1 |
| `ink-4` | `#71717a` | Placeholders, disabled | ~4.5:1 (borderline — avoid for reading text) |

### Brand — burgundy → violet

| Token | Value | Use |
|---|---|---|
| `brand` | `#ed487f` | Primary CTAs, active nav rail, section accents, required `*` |
| `brand-strong` | `#f76398` | CTA hover |
| `brand-weak` | `#44182b` | Translucent bg for active nav items |
| `brand-alt` | `#b8a1fb` | Secondary brand — nested-section accents, chain arrows, chips for suggestions |

### Semantic

| Token | Meaning |
|---|---|
| `ok` / `warn` / `err` | Validation states |
| `info` | "Running now" / live — **always sky-blue** so it doesn't blur with burgundy |
| `idle` | "Not started" neutral grey |

---

## Brand-usage policy

The palette is "Grafana-minimal dark + burgundy-violet accent". **Brand
does not flood forms.** Use it as a dyhatelnyj accent, not as
decoration.

### ✅ Brand is allowed on

- `.btn-primary` CTAs (one primary per screen)
- Active nav states (sidebar item, active tab underline, active
  section in TocRail with `bg-brand-weak/50`)
- Focus ring (`ringColor: #f25088`)
- Selection (`::selection`)
- Required `*` marker
- Section-header accent lines (`border-l-[3px] border-brand`)
- Dirty status dot (pulsing `bg-brand`)
- Tinted gradients ≤ 15% alpha on hero card and header bands
- Glow flashes on success moments (preset applied, validate ok) —
  1 s duration cap, always fade out
- Strategies chain phase badges and inter-phase arrows
- Chips suggesting LoRA baselines (`border-brand bg-brand-weak/50`
  for "selected", `border-line-2` for unselected)

### ❌ Brand is forbidden on

- `<input>` / `<select>` / `<textarea>` backgrounds (use `surface-1`)
- Card fills, except exactly one `.card-hero` per screen
- Table cell/row fills
- Dialog borders (use `line-2`)
- Default hover states (use `surface-3`)
- Non-state-bearing chrome (e.g. "just to brighten things up")

### Alpha guidance

- **≤ 10%** — background tints (section header glow, hero wash)
- **15–20%** — active/hover tints on pills and nav items
- **25–40%** — error/focus rings
- **100%** — text, icons, static accent lines only

---

## Typography

- Sans: **Inter**. Mono: **JetBrains Mono** (values, ids, YAML, paths).
- Root scale: `html { font-size: 113% }`. `1rem ≈ 18.08px`. This is
  intentional — ink density is tuned for it. Don't lower without a
  palette review.
- Custom breakpoints in `tailwind.config.ts` divide default Tailwind
  thresholds by 1.13 so class names still read the familiar way
  (`md:` ≈ 768 CSS px).

### Size tokens (target shape — codemod pending)

| Role | Size | Example |
|---|---|---|
| `caption` / `text-2xs` | 11 px | Hints, footnotes |
| `body` / `text-xs` | 13 px | Form labels, field values |
| `h3` / `text-sm` | 15 px | Card titles, section headers |
| `h2` / `text-lg` | 18 px | Hero headers, project names |

Avoid inline `text-[13px]` — pick the closest token.

---

## Form patterns

### LabelledRow (the default row)

All scalar fields render through `LabelledRow`. Layout:

```
[ *required | label | ? ] | [ input/combobox/select ]  [ <LinkIcon> ]
  ~220 px fixed                 fluid, max 640 px        hover-only
```

Rules:

- Required marker = `*` in a fixed-width slot, brand colour, **always
  in the same grid column** regardless of label length.
- `(required)` suffix is `sr-only` — accessible, not painted.
- `<label htmlFor={id}>` binds to the input's `id` (derive from dotted
  path — `cfg-${path}`).
- Help `?` button uses per-field `aria-label` ("Help for Model name"),
  not a generic one.
- Tooltip flips to the opposite corner when it would overflow the
  viewport (`HelpTooltip` measures + swaps placement).

### Focus signal

When any input inside a row is focused, the label column gets
`bg-brand-weak/10` + `border-brand/30` via
`group-focus-within/row:*`. Don't wire focus state manually —
reuse the group.

### Error state

- Input: `ring-2 ring-err/40 border-err bg-err/[0.03]`
- Label pill: `border-err ring-1 ring-err/30`
- Below row: `text-err font-mono` inline message
- `aria-invalid="true"` on the wrapping div, `aria-describedby` points
  to the error message id

---

## Buttons

| Class | When |
|---|---|
| `.btn-primary` | **One** primary CTA per screen (Save / Run). Shimmer gradient is intentional — users track it visually as the "active lever". |
| `.btn-ghost` | Secondary actions with equal weight (Cancel, Compare with last run) |
| Outline (`border-line-1`) | Neutral / tertiary actions (Validate) |
| `.btn-danger-ghost` | Destructive that needs confirmation (rare, behind modal) |

Never style a button with `bg-brand` directly — use `.btn-primary` so
the shimmer and shadow stay consistent.

---

## Icons

- Source: `/src/components/icons/` only.
- Attrs: `stroke="currentColor"`, `aria-hidden="true"`. Size at call
  site via Tailwind classes (`w-3 h-3`, `w-4 h-4`).
- Never emoji (`🔗`, `✓`, `▸`) in UI chrome. Emoji is acceptable
  in pure user-generated content (descriptions, run names).

Exception: decorative glyphs as SVG path — e.g. the chain arrow
between strategy phases is an inline SVG in `ArrayField.tsx` because
it needs a brand-gradient stroke.

---

## Accessibility checklist (per-component)

- [ ] Every `<input>`/`<select>`/`<textarea>` has an `id` and a
      matching `<label htmlFor>`.
- [ ] Unique `aria-label` on every icon-only button (generic
      "Button" / "Close" is NOT enough — include what it closes).
- [ ] Decorative SVG/emoji have `aria-hidden="true"`.
- [ ] `:focus-visible` is never suppressed. Default ring is brand.
- [ ] Error messages wired via `aria-describedby` to the input.
- [ ] Color is never the only signal — pair with text or icon.
- [ ] Contrast ≥ 4.5:1 for body text, 3:1 for large text.
- [ ] Click-outside dismiss AND `Escape` for every floating element
      (dropdown, tooltip, modal). Use `useClickOutside`.

---

## Custom field renderers

Register in `FieldRenderer.tsx → CUSTOM_FIELD_RENDERERS` keyed by the
dotted config path. Path with numeric indices normalises to `*`:

```ts
// training.strategies.*.strategy_type
CUSTOM_FIELD_RENDERERS['training.strategies.*.strategy_type'] = StrategyTypeField
```

Signature for scalar renderers:

```ts
type CustomFieldRenderer = (props: {
  value: unknown
  onChange: (next: unknown) => void
  onFocus?: () => void
  onBlur?: () => void
}) => JSX.Element
```

For array / object renderers that need the full schema node or
sibling-field access, prefer extending `ArrayField` or
`CollapsibleCard` (via `headerExtra` / `bodyExtra` props) instead of
forking the renderer entrypoint. The `CHAIN_PATHS` set in
`ArrayField.tsx` shows the pattern.

---

## Visual hierarchy recipes

### Section accent

```tsx
<header className="relative pb-3 pl-3 border-l-[3px] border-brand
                   border-b border-b-line-1 rounded-tl-sm
                   bg-gradient-to-r from-brand/[0.08] via-transparent to-transparent">
  <h3 className="text-lg font-semibold text-ink-1">{label}</h3>
</header>
```

### Nested well

```tsx
// Parent card: bg-surface-2
// Open child:  bg-surface-1 border-l-2 border-l-brand-alt/50
// Header tint: bg-gradient-to-r from-brand-alt/[0.12] via-transparent
```

### Success flash (mount + fade)

```tsx
<div className={`transition-shadow duration-[1500ms] ${
  flash ? 'shadow-[0_0_20px_rgba(237,72,127,0.45)]' : 'shadow-none'
}`}>
```

Set `flash=true` on an edge trigger (preset applied, validation
turned green). Auto-reset after 0.8–2.5 s.

---

## Don'ts (common mistakes)

- Don't add `text-[13px]` — use a token.
- Don't paint brand on an input `<textarea>`, even "just a touch".
- Don't add a second `.card-hero` to the same screen.
- Don't ship a dropdown without click-outside + Escape.
- Don't reach for emoji when an SVG from `/icons/` exists.
- Don't invent a new token without removing an existing one.
- Don't use `window.confirm` — open a real modal with a diff (see
  `PresetPreviewModal`).
