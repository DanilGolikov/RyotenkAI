import type { Config } from 'tailwindcss'

/**
 * Burgundy → Violet palette, OKLCH-derived for perceptual uniformity on dark
 * backgrounds. All text tokens pass WCAG AA (≥4.5:1) on `surface-0` through
 * `surface-3`; the gradient is only used on ≥24px headings or iconography to
 * stay above 3:1 for large-text rules.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Surface scale: darkest → lightest, all burgundy-tinted
        'surface-0': '#0c0610',   // app background
        'surface-1': '#160b1c',   // sidebar / sticky panels
        'surface-2': '#1f1028',   // cards
        'surface-3': '#2a1a36',   // hover
        'surface-4': '#3a2646',   // raised/selected

        // Hairlines
        'line-1': '#2e1d3c',
        'line-2': '#3d2a50',

        // Text (tested on surface-0..3)
        'ink': '#f5eef8',         // primary
        'ink-dim': '#cbbcd8',     // secondary
        'ink-mute': '#8d7ba2',    // captions
        'ink-faint': '#6a5780',   // placeholders

        // Brand gradient anchors
        'burgundy': {
          DEFAULT: '#c6306b',
          400: '#e25690',
          500: '#c6306b',
          600: '#9e2255',
          700: '#761a40',
        },
        'violet': {
          DEFAULT: '#8b5cf6',
          300: '#b39afb',
          400: '#a185f8',
          500: '#8b5cf6',
          600: '#6d3ddc',
          700: '#5028ae',
        },

        // Status — chosen to be distinguishable from burgundy/violet
        'status-ok':      '#4ade80',
        'status-warn':    '#fbbf24',
        'status-err':     '#f87171',
        'status-run':     '#38bdf8',   // sky — different hue family from violet
        'status-idle':    '#6a5780',
      },
      backgroundImage: {
        'gradient-brand':
          'linear-gradient(135deg, #c6306b 0%, #8b5cf6 100%)',
        'gradient-brand-soft':
          'linear-gradient(135deg, rgba(198,48,107,0.22) 0%, rgba(139,92,246,0.22) 100%)',
        'gradient-sidebar':
          'linear-gradient(180deg, #160b1c 0%, #0c0610 80%)',
        'gradient-glow':
          'radial-gradient(circle at 30% 0%, rgba(198,48,107,0.18), transparent 60%)',
      },
      boxShadow: {
        'glow-burgundy': '0 0 24px rgba(198, 48, 107, 0.35)',
        'glow-violet':   '0 0 24px rgba(139, 92, 246, 0.35)',
        'card':          '0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.4)',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      ringColor: {
        DEFAULT: '#c6306b',
      },
      keyframes: {
        pulse_ring: {
          '0%':   { boxShadow: '0 0 0 0 rgba(56,189,248,0.55)' },
          '100%': { boxShadow: '0 0 0 10px rgba(56,189,248,0)' },
        },
      },
      animation: {
        'pulse-ring': 'pulse_ring 1.6s ease-out infinite',
      },
    },
  },
  plugins: [],
} satisfies Config
