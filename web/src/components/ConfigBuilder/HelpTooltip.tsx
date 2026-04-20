/**
 * Tiny CSS-only "?" icon with a hover / focus tooltip. Uses a *named*
 * Tailwind group ("help") so the tooltip only reacts to the icon — not to
 * any outer ``.group`` ancestor such as ProjectCard or FieldAnchor that
 * would otherwise trigger ``group-hover`` here by accident.
 *
 * Renders nothing when ``text`` is empty so callers can pass raw schema
 * descriptions freely.
 */
export function HelpTooltip({ text }: { text?: string }) {
  if (!text) return null
  return (
    <span className="relative group/help inline-flex">
      <button
        type="button"
        aria-label="Field help"
        className="w-4 h-4 rounded-full border border-line-2 text-ink-3 text-[0.6rem] font-semibold flex items-center justify-center cursor-help hover:text-ink-1 hover:border-brand-alt transition focus:outline-none focus:border-brand-alt"
      >
        ?
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 w-72 -translate-x-1/2 rounded-md border border-line-2 bg-surface-4 px-3 py-2 text-[0.7rem] leading-snug text-ink-1 opacity-0 shadow-card transition-opacity duration-150 group-hover/help:opacity-100 group-focus-within/help:opacity-100"
      >
        {text}
      </span>
    </span>
  )
}
