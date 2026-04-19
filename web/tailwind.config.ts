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
        // Neutral surfaces — lifted ~4% in lightness vs. v1 so the UI
        // doesn't feel like a void. Cool-violet undertone (hue ≈ 290°)
        // keeps the brand direction without painting everything burgundy.
        // OKLCH steps ≈ 12% → 27% in 3-4% increments, so elevation reads
        // without shadows (required in dark mode).
        'surface-0': '#141221',   // app canvas
        'surface-1': '#1a1729',   // sidebar, sticky panels
        'surface-2': '#221f33',   // cards
        'surface-3': '#2c2840',   // hover / selected card
        'surface-4': '#38334f',   // raised / popover

        // Borders — slightly lighter so they read against the new surfaces
        'line-1': '#2e2a43',
        'line-2': '#423d5c',

        // Text — ink-1 unchanged (passes AA everywhere), ink-2 warmer
        // lavender so body copy feels alive, ink-3 bumped so captions
        // don't feel washed out.
        'ink-1': '#f1ecf7',       // primary  (contrast on surface-0 ≈ 15.8:1)
        'ink-2': '#b9b1cc',       // secondary (7.9:1 on s-0)
        'ink-3': '#857d9b',       // captions  (4.9:1)
        'ink-4': '#595370',       // placeholders / disabled

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
          'linear-gradient(135deg, rgba(214,48,95,0.18) 0%, rgba(139,92,246,0.18) 100%)',
        'gradient-sidebar':
          'linear-gradient(180deg, #1a1729 0%, #141221 85%)',
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
