import { useEffect, useRef, useState } from 'react'

/**
 * Click-to-toggle "?" help icon. Opens on click, closes on: another click
 * on the icon, Escape, or any click outside the popover. Intentionally
 * not hover-triggered so users can read the tooltip without worrying
 * about overshoot, and so nested groups (ProjectCard, FieldAnchor) can't
 * open it accidentally.
 *
 * Renders nothing when ``text`` is empty.
 */
export function HelpTooltip({ text }: { text?: string }) {
  const [open, setOpen] = useState(false)
  const wrapperRef = useRef<HTMLSpanElement | null>(null)

  useEffect(() => {
    if (!open) return
    function onDocClick(e: MouseEvent) {
      if (!wrapperRef.current) return
      if (!wrapperRef.current.contains(e.target as Node)) setOpen(false)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  if (!text) return null

  return (
    <span ref={wrapperRef} className="relative inline-flex">
      <button
        type="button"
        aria-label="Field help"
        aria-expanded={open}
        onClick={(e) => {
          e.preventDefault()
          e.stopPropagation()
          setOpen((v) => !v)
        }}
        className={[
          'w-4 h-4 rounded-full border text-[0.6rem] font-semibold flex items-center justify-center cursor-pointer transition focus:outline-none',
          open
            ? 'border-brand-alt text-ink-1 bg-brand-alt/15'
            : 'border-line-2 text-ink-3 hover:text-ink-1 hover:border-brand-alt',
        ].join(' ')}
      >
        ?
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute left-0 top-full z-30 mt-2 w-80 max-w-[calc(100vw-2rem)] rounded-md border border-line-2 bg-surface-4 px-3 py-2 text-[0.75rem] leading-snug text-ink-1 shadow-card whitespace-normal break-words"
        >
          {text}
        </span>
      )}
    </span>
  )
}
