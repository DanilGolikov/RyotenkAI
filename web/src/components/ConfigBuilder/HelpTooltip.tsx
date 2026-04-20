/**
 * Tiny CSS-only "?" icon with a hover tooltip. Renders nothing when
 * ``text`` is empty so callers can pass raw schema descriptions freely.
 */
export function HelpTooltip({ text }: { text?: string }) {
  if (!text) return null
  return (
    <span className="relative group inline-flex">
      <span
        tabIndex={0}
        aria-label="Field help"
        className="w-4 h-4 rounded-full border border-line-2 text-ink-3 text-[0.6rem] font-semibold flex items-center justify-center cursor-help hover:text-ink-1 hover:border-brand-alt transition"
      >
        ?
      </span>
      <span
        role="tooltip"
        className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 w-72 -translate-x-1/2 rounded-md border border-line-2 bg-surface-4 px-3 py-2 text-[0.7rem] leading-snug text-ink-1 opacity-0 shadow-card transition group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {text}
      </span>
    </span>
  )
}
