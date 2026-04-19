import type { Config } from 'tailwindcss'

/**
 * Dashboard palette — "brighter dark + burgundy→violet brand".
 *   - Surfaces lifted so the UI doesn't feel gloomy (~L 18% → 28%),
 *     neutral zinc (no hue tint in chrome).
 *   - Brand: burgundy → violet gradient. Used only on CTAs, logo,
 *     card-hero, active nav rule, focus ring, selection.
 *   - info (running / live) = sky-blue — distinct from the warm brand
 *     so "currently running" and "launch" never blur together.
 *   - Steady 4% L steps so elevation reads without shadows.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Surfaces — zinc, but starting ~L 18% (v2 was 12%) so the app
        // looks like "soft charcoal", not a void.
        'surface-0': '#18181b',   // app canvas    (zinc-900)
        'surface-1': '#1e1e22',   // sidebar
        'surface-2': '#27272a',   // cards         (zinc-800)
        'surface-3': '#323238',   // hover / selected
        'surface-4': '#3f3f46',   // popover       (zinc-700)

        // Borders stronger so they read on the lifted surfaces
        'line-1': '#2f2f34',
        'line-2': '#4a4a52',

        // Text — ink-2 bumped to zinc-300 so body copy stays crisp
        // against the now-lighter surface-2.
        'ink-1': '#fafafa',       // primary
        'ink-2': '#d4d4d8',       // secondary (zinc-300) — 10.2:1 on s-0
        'ink-3': '#a1a1aa',       // captions  (zinc-400)
        'ink-4': '#71717a',       // placeholders / disabled

        // Brand — burgundy → violet. Used only on CTAs, logo, active
        // nav, focus ring, card-hero wash. `brand-alt` is the gradient
        // partner (violet).
        'brand':        '#d6305f',   // primary CTA / active
        'brand-strong': '#ea4a78',   // hover
        'brand-weak':   '#3a1522',   // translucent bg

        'brand-alt':    '#8b5cf6',   // violet — gradient partner

        // Semantic status — running/live lives on SKY so it never merges
        // with the burgundy brand. ok/warn/err tuned for dark mode.
        'ok':    '#4ade80',   // green-400
        'warn':  '#f59e0b',   // amber-500
        'err':   '#f87171',   // red-400
        'info':  '#60a5fa',   // blue-400  (used for running/live)
        'idle':  '#71717a',
      },
      backgroundImage: {
        'gradient-brand':
          'linear-gradient(135deg, #d6305f 0%, #8b5cf6 100%)', // burgundy → violet
        'gradient-brand-soft':
          'linear-gradient(135deg, rgba(214,48,95,0.18) 0%, rgba(139,92,246,0.18) 100%)',
        'gradient-sidebar':
          'linear-gradient(180deg, #1e1e22 0%, #18181b 85%)',
      },
      boxShadow: {
        'glow-brand':    '0 0 24px rgba(214, 48, 95, 0.38)',
        'card':          '0 1px 0 rgba(255,255,255,0.04) inset, 0 6px 18px rgba(0,0,0,0.35)',
        'inset-accent':  'inset 2px 0 0 #d6305f',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      ringColor: {
        DEFAULT: '#ea4a78',
      },
      keyframes: {
        pulse_ring: {
          '0%':   { boxShadow: '0 0 0 0 rgba(96,165,250,0.55)' },
          '100%': { boxShadow: '0 0 0 8px rgba(96,165,250,0)' },
        },
      },
      animation: {
        'pulse-ring': 'pulse_ring 1.6s ease-out infinite',
      },
    },
  },
  plugins: [],
} satisfies Config
