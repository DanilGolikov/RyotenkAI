import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useClickOutside } from '../../hooks/useClickOutside'

/**
 * Click-to-toggle "?" help icon. Opens on click, closes on: another click
 * on the icon, Escape, or any click outside the popover. Intentionally
 * not hover-triggered so users can read the tooltip without worrying
 * about overshoot, and so nested groups (ProjectCard, FieldAnchor) can't
 * open it accidentally.
 *
 * Flip logic: after render we measure the bubble's bounding rect and
 * snap it to whichever corner keeps it on-screen. Viewport-relative so
 * it also handles narrow cards inside the Config form.
 *
 * Renders nothing when ``text`` is empty.
 */
type Placement = 'bl' | 'br' | 'tl' | 'tr'

export function HelpTooltip({ text, label = 'Field help' }: { text?: string; label?: string }) {
  const [open, setOpen] = useState(false)
  const [placement, setPlacement] = useState<Placement>('bl')
  const wrapperRef = useRef<HTMLSpanElement | null>(null)
  const bubbleRef = useRef<HTMLSpanElement | null>(null)

  useClickOutside(wrapperRef, open, () => setOpen(false))

  // Snap placement to whichever corner keeps the bubble in-viewport.
  // `useLayoutEffect` so the first paint already uses the correct
  // corner — no flash of the flipped tooltip.
  useLayoutEffect(() => {
    if (!open || !bubbleRef.current) return
    const b = bubbleRef.current.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight
    const overflowRight = b.right > vw - 8
    const overflowBottom = b.bottom > vh - 8
    setPlacement(
      overflowBottom ? (overflowRight ? 'tr' : 'tl') : overflowRight ? 'br' : 'bl',
    )
  }, [open])

  // Recompute on resize while open.
  useEffect(() => {
    if (!open) return
    function remeasure() {
      if (!bubbleRef.current) return
      const b = bubbleRef.current.getBoundingClientRect()
      const vw = window.innerWidth
      const vh = window.innerHeight
      const overflowRight = b.right > vw - 8
      const overflowBottom = b.bottom > vh - 8
      setPlacement(
        overflowBottom ? (overflowRight ? 'tr' : 'tl') : overflowRight ? 'br' : 'bl',
      )
    }
    window.addEventListener('resize', remeasure)
    return () => window.removeEventListener('resize', remeasure)
  }, [open])

  if (!text) return null

  // Placement class — each key maps corner anchor to wrapper-relative
  // offset. "bl" = below-left (default), "tr" = above-right (flipped
  // against both edges).
  const placementCls: Record<Placement, string> = {
    bl: 'left-0 top-full mt-2',
    br: 'right-0 top-full mt-2',
    tl: 'left-0 bottom-full mb-2',
    tr: 'right-0 bottom-full mb-2',
  }

  return (
    <span ref={wrapperRef} className="relative inline-flex">
      <button
        type="button"
        aria-label={label}
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
          ref={bubbleRef}
          role="tooltip"
          className={`absolute z-30 w-80 max-w-[calc(100vw-2rem)] rounded-md border border-line-2 bg-surface-4 px-3 py-2 text-[0.75rem] leading-snug text-ink-1 shadow-card whitespace-normal break-words ${placementCls[placement]}`}
        >
          {text}
        </span>
      )}
    </span>
  )
}
