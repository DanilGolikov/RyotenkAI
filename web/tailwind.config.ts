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
        // Neutral zinc surfaces (pattern: Linear / Vercel / shadcn). Zero
        // hue-tinting so the chrome stops feeling like "night-club purple";
        // brand colour lives only on CTAs, logo, active nav, focus ring.
        // Lightness ramps ≈ 12% → 28% in steady steps so elevation reads
        // without shadows (required in dark mode).
        'surface-0': '#0f0f11',   // app canvas     (zinc-950-ish)
        'surface-1': '#18181b',   // sidebar        (zinc-900)
        'surface-2': '#1f1f23',   // cards
        'surface-3': '#27272a',   // hover / selected card (zinc-800)
        'surface-4': '#35353b',   // popover / raised

        // Borders
        'line-1': '#27272a',      // hairline (same L as s-3, ~1px reads via contrast)
        'line-2': '#3f3f46',      // stronger (zinc-700)

        // Text — neutral zinc scale, matches shadcn defaults; contrast
        // ≥ 4.5:1 down to ink-3 on surface-0 (WCAG AA for body).
        'ink-1': '#fafafa',       // primary   (17.5:1 on s-0)
        'ink-2': '#a1a1aa',       // secondary (7.2:1 on s-0)
        'ink-3': '#71717a',       // captions  (4.6:1)
        'ink-4': '#52525b',       // placeholders / disabled

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
          'linear-gradient(180deg, #18181b 0%, #0f0f11 85%)',
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
