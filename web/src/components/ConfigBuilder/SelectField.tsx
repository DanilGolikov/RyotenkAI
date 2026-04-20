import { useEffect, useRef, useState } from 'react'

export interface SelectOption {
  value: string
  label?: string
}

interface Props {
  value: string
  options: SelectOption[]
  onChange: (next: string) => void
  placeholder?: string
  /** Append a leading empty ("—") row so users can clear the field. */
  allowEmpty?: boolean
  /** Trigger width/padding class. Matches INPUT_BASE sizing via caller. */
  triggerClassName?: string
  /** Extra class applied to the trigger wrapper (`relative inline-block`). */
  wrapperClassName?: string
  /** Extra row rendered after the options (e.g. "+ Add new"). Rendered
   * inside the listbox but styled separately so it reads as an action,
   * not a selectable value. Closes the menu on click. */
  footer?: React.ReactNode
  /** Focus lifecycle — forwarded to the trigger button so the parent
   *  validation context can track focus entry/exit. */
  onFocus?: () => void
  onBlur?: () => void
}

const TRIGGER_BASE =
  'h-8 rounded bg-surface-1 border border-line-1 px-2.5 pr-7 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors flex items-center justify-between gap-2'

/**
 * Custom listbox dropdown. Mirrors native ``<select>`` semantics (value,
 * onChange, keyboard) but lets us paint the selected row with the
 * burgundy→violet brand accent bar on the left and a soft brand-tinted
 * background. Keyboard: ArrowUp/Down moves the cursor, Enter commits,
 * Escape closes. Outside click closes.
 */
export function SelectField({
  value,
  options,
  onChange,
  placeholder = '—',
  allowEmpty = false,
  triggerClassName = 'w-auto min-w-[160px]',
  wrapperClassName = '',
  footer,
  onFocus,
  onBlur,
}: Props) {
  const [open, setOpen] = useState(false)
  const [cursor, setCursor] = useState(0)
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const triggerRef = useRef<HTMLButtonElement | null>(null)

  const items: SelectOption[] = allowEmpty
    ? [{ value: '', label: '—' }, ...options]
    : options

  const selectedLabel = (() => {
    const hit = items.find((o) => o.value === value)
    if (hit) return hit.label ?? hit.value
    return value || placeholder
  })()

  // Seed cursor on the currently selected value whenever the menu opens.
  // Kept in its own effect so subsequent cursor changes (ArrowUp/Down)
  // don't re-seed back to the selected value mid-navigation.
  useEffect(() => {
    if (!open) return
    const idx = items.findIndex((o) => o.value === value)
    setCursor(idx >= 0 ? idx : 0)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  useEffect(() => {
    if (!open) return
    function onDocClick(e: MouseEvent) {
      if (!wrapperRef.current) return
      if (!wrapperRef.current.contains(e.target as Node)) setOpen(false)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setOpen(false)
        triggerRef.current?.focus()
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        setCursor((c) => Math.min(c + 1, items.length - 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setCursor((c) => Math.max(c - 1, 0))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        const opt = items[cursor]
        if (opt) {
          onChange(opt.value)
          setOpen(false)
          triggerRef.current?.focus()
        }
      }
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, cursor])

  const isEmpty = value === '' || value === undefined || value === null
  return (
    <div ref={wrapperRef} className={`relative inline-block ${wrapperClassName}`}>
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        onFocus={onFocus}
        onBlur={onBlur}
        className={`${TRIGGER_BASE} ${triggerClassName}`}
      >
        <span className={isEmpty ? 'text-ink-4' : ''}>{selectedLabel}</span>
        <span aria-hidden className="text-ink-3 text-[10px]">▾</span>
      </button>
      {open && (
        <div
          role="listbox"
          className="absolute z-30 left-0 mt-1 min-w-full rounded border border-line-2 bg-surface-1 shadow-card overflow-hidden py-0.5"
        >
          <ul className="list-none m-0 p-0">
            {items.map((opt, idx) => {
              const selected = opt.value === value
              const active = cursor === idx
              return (
                <li
                  key={opt.value || '__empty__'}
                  role="option"
                  aria-selected={selected}
                  onMouseEnter={() => setCursor(idx)}
                  onMouseDown={(e) => {
                    // Prevent the trigger's mousedown outside-handler from
                    // closing before the click lands on the option.
                    e.preventDefault()
                  }}
                  onClick={() => {
                    onChange(opt.value)
                    setOpen(false)
                    triggerRef.current?.focus()
                  }}
                  className={[
                    'relative pl-3 pr-3 py-1.5 text-[13px] font-mono cursor-pointer',
                    selected
                      ? 'text-ink-1 bg-gradient-brand-soft shadow-inset-accent'
                      : active
                        ? 'bg-surface-2 text-ink-1'
                        : 'text-ink-2',
                  ].join(' ')}
                >
                  {opt.label ?? opt.value}
                </li>
              )
            })}
          </ul>
          {footer && (
            <div
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => {
                setOpen(false)
                triggerRef.current?.focus()
              }}
              className="border-t border-line-1"
            >
              {footer}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
