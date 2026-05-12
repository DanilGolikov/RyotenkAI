/**
 * Global Storybook preview wiring.
 *
 *  - Loads the same Tailwind stylesheet the SPA uses, so stories
 *    visually match production.
 *  - Boots `msw-storybook-addon` once with a default worker.
 *  - Exposes a `theme` global with `light` / `dark` so stories can be
 *    flipped via the toolbar; data-theme attribute is what the design
 *    system already uses in `index.html`.
 *  - Wraps every story in a `MemoryRouter` so components that read
 *    `useNavigate` / `<Link>` don't crash.
 */

import React, { useEffect } from 'react'
import type { Preview } from '@storybook/react'
import { MemoryRouter } from 'react-router-dom'
import { initialize, mswLoader } from 'msw-storybook-addon'

import '../src/styles/globals.css'

// Initialise MSW once. The addon is no-op when no handlers are
// supplied by the story.
initialize({
  onUnhandledRequest: 'bypass',
})

function ThemeDecorator({
  theme,
  children,
}: {
  theme: 'light' | 'dark'
  children: React.ReactNode
}) {
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    return () => {
      document.documentElement.removeAttribute('data-theme')
    }
  }, [theme])
  return <>{children}</>
}

const preview: Preview = {
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    backgrounds: {
      default: 'app',
      values: [
        { name: 'app', value: '#0d0e10' },
        { name: 'light', value: '#fafafa' },
      ],
    },
  },
  globalTypes: {
    theme: {
      name: 'Theme',
      description: 'Light / dark mode toggle',
      defaultValue: 'dark',
      toolbar: {
        icon: 'paintbrush',
        items: [
          { value: 'light', title: 'Light' },
          { value: 'dark', title: 'Dark' },
        ],
        dynamicTitle: true,
      },
    },
  },
  loaders: [mswLoader],
  decorators: [
    (Story, context) => (
      <ThemeDecorator theme={context.globals.theme as 'light' | 'dark'}>
        <MemoryRouter>
          <div className="p-6 min-h-[200px]">
            <Story />
          </div>
        </MemoryRouter>
      </ThemeDecorator>
    ),
  ],
  tags: ['autodocs'],
}

export default preview
