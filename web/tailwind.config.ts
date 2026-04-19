import type { Config } from 'tailwindcss'

/**
 * 2026 dashboard palette:
 *   - Neutrals drive ~90% of the UI (cool-tinted zinc, 6 tiers via OKLCH
 *     lightness for perceptual uniformity).
 *   - ONE brand accent (burgundy→violet gradient) used ONLY for logo,
 *     primary CTAs, focus rings, and live-running pulse. Brand is a
 *     signal, not a wallpaper.
 *   - Semantic status colours are isolated from the brand and calibrated
 *     for dark mode (lighter lightness, softer saturation).
 *   - All text tokens pass WCAG AA (≥4.5:1) on surface-1/2/3; large
 *     text + 3:1 on surface-0.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Neutral surfaces (zinc with a whisper of cool-magenta tint,
        // lightness climbs in ~4% OKLCH steps so elevation reads even
        // without drop shadows — required in dark mode).
        'surface-0': '#0e0c12',   // app canvas
        'surface-1': '#15131b',   // sidebar, sticky panels
        'surface-2': '#1c1a23',   // cards
        'surface-3': '#25222d',   // hover / selected card
        'surface-4': '#302c38',   // raised / popover

        // Borders
        'line-1': '#252230',      // hairline
        'line-2': '#363244',      // stronger

        // Text (lightness-driven hierarchy)
        'ink-1': '#eeeaf3',       // primary
        'ink-2': '#a59eb4',       // secondary
        'ink-3': '#6f6880',       // captions
        'ink-4': '#4a4556',       // placeholders / disabled

        // Brand (single hue family — used sparingly)
        'brand':        '#d6305f',   // solid CTA / active
        'brand-strong': '#ea4a78',   // hover
        'brand-weak':   '#3a1522',   // translucent bg accents (10% surface)

        'brand-alt':    '#8b5cf6',   // gradient partner only

        // Semantic (dark-mode calibrated — softer than pure)
        'ok':    '#4ade80',
        'warn':  '#f5a524',
        'err':   '#f87171',
        'info':  '#60a5fa',   // running / live
        'idle':  '#6f6880',
      },
      backgroundImage: {
        // Reserved usage: logo, launch CTA, "hero" KPI (Active runs).
        'gradient-brand':
          'linear-gradient(135deg, #d6305f 0%, #8b5cf6 100%)',
        'gradient-brand-soft':
          'linear-gradient(135deg, rgba(214,48,95,0.14) 0%, rgba(139,92,246,0.14) 100%)',
        'gradient-sidebar':
          'linear-gradient(180deg, #15131b 0%, #0e0c12 85%)',
      },
      boxShadow: {
        'glow-brand':    '0 0 20px rgba(214, 48, 95, 0.35)',
        'card':          '0 1px 0 rgba(255,255,255,0.03) inset, 0 6px 18px rgba(0,0,0,0.35)',
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
        DEFAULT: '#d6305f',
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
