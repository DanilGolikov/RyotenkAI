# `src/stories/` + colocated `*.stories.tsx`

Storybook (Phase 6) живёт в `web/.storybook/`, stories разрешено держать
двумя способами:

1. **Колокация** (предпочтительно) — рядом с компонентом:
   `web/src/components/StatusPill.stories.tsx`. Используем для всех новых
   stories; так проще держать историю в синхроне с кодом.
2. **Этот каталог** (`web/src/stories/`) — для общих демонстраций
   (icon set demo, design tokens, цветовая палитра), не привязанных к
   одному компоненту.

Глобальный `stories` glob в [main.ts](../../.storybook/main.ts) ловит
оба варианта: `../src/**/*.stories.tsx`.

## Naming convention

- Файл: `<Component>.stories.tsx`.
- `Meta.title` повторяет логическую иерархию: `Components/StatusPill`,
  `Modals/DeleteProjectModal`, `Project/StalePluginsBanner`.
- Каждый story — отдельный named export (`Default`, `Loading`, `Empty`,
  `Error`, `LongList`). Стараемся держать минимум три варианта:
  Default + edge case + ошибка/пустое состояние.

## CSF3 vs MDX

- **CSF3** (TypeScript object literals — `Meta`, `StoryObj`) —
  дефолт. 95% историй прекрасно ложится сюда и не требует MDX.
- **MDX** — оставляем для "design system" страниц с длинным
  пояснительным текстом и встроенными embed-ами. У нас пока нет таких,
  но `main.ts` glob их подхватит, когда появятся.

## Минимум для нового story

```tsx
import type { Meta, StoryObj } from '@storybook/react'
import { MyComponent } from './MyComponent'

const meta: Meta<typeof MyComponent> = {
  title: 'Components/MyComponent',
  component: MyComponent,
}
export default meta
type Story = StoryObj<typeof MyComponent>

export const Default: Story = { args: { ... } }
export const Empty: Story = { args: { ... } }
export const Error: Story = { args: { ... } }
```

## Visual regression

Каждая story автоматически становится снэпшотом для
[lost-pixel](../../.lost-pixel.config.ts). На Phase 6 visual diffs —
**non-blocking warning** (см. Decision 7 в плане): первые 6 месяцев
baseline-ы стабилизируются, merge разрешён даже при diff. После 6
месяцев — promotion to blocking.

Маскируем волатильные регионы (timestamps, dynamic IDs) через
`data-lostpixel-mask="true"` на DOM-элементе либо явный CSS-селектор в
`lost-pixel.config.ts`.
