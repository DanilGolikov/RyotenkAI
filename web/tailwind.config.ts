import type { Config } from 'tailwindcss'

/**
 * Dashboard palette — Grafana-minimal dark + burgundy→violet brand.
 *   - Surfaces flat, near-black with a faint cool undertone (modelled
 *     after Grafana's #181B1F base). Steady ~3% L steps so elevation
 *     still reads without shadows or tinted backgrounds.
 *   - Borders are muted off-white greys (low alpha), so structure comes
 *     from hairline rules rather than filled chrome.
 *   - Brand (burgundy → violet) is preserved but reserved for CTAs,
 *     logo, active nav rule, focus ring, selection — no longer bleeds
 *     into inputs or section cards.
 *   - info (running / live) stays sky-blue so "currently running" never
 *     blurs into the warm brand.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      // Custom breakpoints tuned to the global 113% root font-size
      // (see globals.css — `html { font-size: 113% }`). Default
      // Tailwind breakpoints are rem-based, so a 113% root would make
      // `md:` fire at ~868 CSS px instead of the conventional 768.
      // We restore the conventional CSS-px thresholds by dividing the
      // default rem values by 1.13 and rounding. Result: class names
      // still read the familiar way (`md:` ≈ 768 CSS px) regardless of
      // the ink-density scale choice.
      screens: {
        sm: '566px',   // default 640 / 1.13
        md: '680px',   // default 768 / 1.13
        lg: '907px',   // default 1024 / 1.13
        xl: '1133px',  // default 1280 / 1.13
        '2xl': '1359px', // default 1536 / 1.13
      },
      colors: {
        // Surfaces — near-black canvas (~L 11%) with ~3% L lifts per step.
        // No hue tint — chrome reads as neutral dark, closer to Grafana.
        'surface-0': '#181b1f',   // app canvas
        'surface-1': '#1f2226',   // sidebar / inputs
        'surface-2': '#262a2f',   // cards / hover-of-input
        'surface-3': '#2f3338',   // hover / selected
        'surface-4': '#3a3e44',   // popover

        // Borders — hairline off-white greys. Lower contrast than before
        // so cards feel flat; relies on surface elevation + hover to
        // communicate grouping.
        'line-1': '#2c3036',
        'line-2': '#3c4046',

        // Text — ink-2 bumped to zinc-300 so body copy stays crisp
        // against the now-lighter surface-2.
        'ink-1': '#fafafa',       // primary
        'ink-2': '#d4d4d8',       // secondary (zinc-300) — 10.2:1 on s-0
        'ink-3': '#a1a1aa',       // captions  (zinc-400)
        'ink-4': '#71717a',       // placeholders / disabled

        // Brand — burgundy → violet, brighter + punchier so the gradient
        // reads as vibrant signal rather than moody.
        'brand':        '#ed487f',   // primary CTA / active (↑ L + chroma)
        'brand-strong': '#f76398',   // hover
        'brand-weak':   '#44182b',   // translucent bg

        'brand-alt':    '#b8a1fb',   // violet-300 (↑ lightness)

        // Semantic status — running/live lives on SKY so it never merges
        // with the burgundy brand. ok/warn/err tuned for dark mode.
        'ok':    '#4ade80',   // green-400
        'warn':  '#f59e0b',   // amber-500
        'err':   '#f87171',   // red-400
        'info':  '#60a5fa',   // blue-400  (used for running/live)
        'idle':  '#71717a',
      },
      backgroundImage: {
        // Reserved for logo only now — do NOT use on sections or hero.
        'gradient-brand':
          'linear-gradient(135deg, #e63570 0%, #a78bfa 100%)',
        'gradient-brand-soft':
          'linear-gradient(135deg, rgba(230,53,112,0.26) 0%, rgba(167,139,250,0.22) 100%)',
      },
      boxShadow: {
        'glow-brand':    '0 0 28px rgba(230, 53, 112, 0.48)',
        'card':          '0 1px 0 rgba(255,255,255,0.04) inset, 0 6px 18px rgba(0,0,0,0.35)',
        'inset-accent':  'inset 2px 0 0 #e63570',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      ringColor: {
        DEFAULT: '#f25088',
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
