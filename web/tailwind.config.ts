import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#0f1115',
          raised: '#171a21',
          muted: '#1f232c',
        },
        accent: {
          DEFAULT: '#56c2a1',
          muted: '#3b8a70',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'Menlo', 'SFMono-Regular', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config
