/**
 * Storybook 8 configuration — Vite builder, React framework.
 *
 * Phase 6 ships a minimal manual setup (no `storybook init`) so the
 * repo stays clean and the bundle is predictable. Five representative
 * stories live under `src/**\/*.stories.tsx`; we deliberately do not
 * try to cover every component in this phase (Decision 7 marks visual
 * regression as low priority — infrastructure first, breadth later).
 *
 * Addons are intentionally minimal:
 *  - `@storybook/addon-essentials` gives us controls, docs, viewport,
 *    backgrounds, measure, outline — everything 90% of stories use.
 *  - `msw-storybook-addon` wires the same MSW handlers used by Vitest
 *    component tests, so stories that fetch can be deterministic.
 */

import type { StorybookConfig } from '@storybook/react-vite'

const config: StorybookConfig = {
  framework: {
    name: '@storybook/react-vite',
    options: {},
  },
  stories: [
    '../src/**/*.stories.@(ts|tsx|mdx)',
  ],
  addons: [
    '@storybook/addon-essentials',
    'msw-storybook-addon',
  ],
  docs: {
    autodocs: 'tag',
  },
  // Vite final tweak — the SPA aliases `@` to `src/`; mirror that for
  // stories so imports like `@/components/Foo` keep working.
  async viteFinal(viteConfig) {
    const path = await import('node:path')
    viteConfig.resolve = viteConfig.resolve ?? {}
    viteConfig.resolve.alias = {
      ...(viteConfig.resolve.alias ?? {}),
      '@': path.resolve(__dirname, '../src'),
    }
    return viteConfig
  },
  typescript: {
    check: false,
    reactDocgen: 'react-docgen',
  },
}

export default config
