import { useEffect, useRef, useState } from 'react'

interface Props {
  value: unknown
  onChange: (next: unknown) => void
  /** Forwarded to the input so the surrounding validation context can
   *  track focus for the status pill + debounced re-validate. */
  onFocus?: () => void
  onBlur?: () => void
}

interface HFModel {
  id: string
  downloads?: number
  likes?: number
  pipeline_tag?: string
}

const HF_SEARCH_URL = 'https://huggingface.co/api/models'
const DEBOUNCE_MS = 300
const MAX_RESULTS = 15
const MAX_RETRIES = 2

/**
 * Free-typing combobox backed by the public Hugging Face models API
 * (``https://huggingface.co/api/models?search=...``). No token — public
 * models only, which is all we need for picking a base.
 *
 * UX contract:
 *   - User types → debounced fetch (300 ms) → suggestions appear.
 *   - User can still type anything; the input IS the value. Suggestions
 *     are a convenience, not a gate.
 *   - Network failures are silent — two retries with backoff, then the
 *     field degrades to a plain text input. We never show a network
 *     error for this kind of optional autocomplete; it would add noise
 *     and the user still has a working input.
 *   - In-flight requests are aborted on every new keystroke.
 */
export function HFModelField({ value, onChange, onFocus, onBlur }: Props) {
  const current = typeof value === 'string' ? value : ''
  const [suggestions, setSuggestions] = useState<HFModel[]>([])
  const [open, setOpen] = useState(false)
  const [cursor, setCursor] = useState(-1)
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const focusedRef = useRef(false)

  // Debounced fetch with in-flight abort + silent retry. Fires on any
  // non-empty query so suggestions are primed by the time the user
  // opens the menu — programmatic focus and initial value both work.
  useEffect(() => {
    const q = current.trim()
    if (!q) {
      setSuggestions([])
      return
    }

    const timer = window.setTimeout(async () => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      let attempt = 0
      let lastErr: unknown = null
      while (attempt <= MAX_RETRIES) {
        try {
          const res = await fetch(
            `${HF_SEARCH_URL}?search=${encodeURIComponent(q)}&limit=${MAX_RESULTS}&sort=downloads&direction=-1`,
            { signal: controller.signal },
          )
          if (!res.ok) throw new Error(`HF ${res.status}`)
          const data: HFModel[] = await res.json()
          if (!controller.signal.aborted) {
            setSuggestions(Array.isArray(data) ? data : [])
            setCursor(-1)
          }
          return
        } catch (exc) {
          if ((exc as Error)?.name === 'AbortError') return
          lastErr = exc
          attempt += 1
          if (attempt > MAX_RETRIES) break
          // Exponential backoff: 250 / 500 ms.
          await new Promise((r) => setTimeout(r, 250 * 2 ** (attempt - 1)))
        }
      }
      // Silent failure — leave suggestions empty. Input still works.
      if (lastErr) setSuggestions([])
    }, DEBOUNCE_MS)

    return () => {
      window.clearTimeout(timer)
    }
  }, [current])

  // Outside-click + Escape close.
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

  function pick(id: string) {
    onChange(id)
    setOpen(false)
    inputRef.current?.blur()
  }

  function onInputKey(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open || suggestions.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setCursor((c) => Math.min(c + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setCursor((c) => Math.max(c - 1, 0))
    } else if (e.key === 'Enter') {
      if (cursor >= 0 && cursor < suggestions.length) {
        e.preventDefault()
        pick(suggestions[cursor].id)
      }
    }
  }

  return (
    <div ref={wrapperRef} className="relative w-[640px] max-w-full">
      <input
        ref={inputRef}
        type="text"
        role="combobox"
        aria-expanded={open && suggestions.length > 0}
        aria-autocomplete="list"
        value={current}
        onChange={(e) => {
          onChange(e.target.value)
          setOpen(true)
        }}
        onFocus={() => {
          focusedRef.current = true
          if (current.trim()) setOpen(true)
          onFocus?.()
        }}
        onBlur={() => {
          focusedRef.current = false
          // Defer so clicks on suggestions register before close.
          window.setTimeout(() => setOpen(false), 120)
          onBlur?.()
        }}
        onKeyDown={onInputKey}
        placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct"
        className="h-8 w-full rounded bg-surface-1 border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors"
      />
      {open && suggestions.length > 0 && (
        <ul
          role="listbox"
          className="absolute z-30 left-0 right-0 mt-1 rounded border border-line-2 bg-surface-1 shadow-card overflow-hidden py-0.5 max-h-80 overflow-y-auto"
        >
          {suggestions.map((m, idx) => {
            const active = cursor === idx
            const selected = m.id === current
            return (
              <li
                key={m.id}
                role="option"
                aria-selected={selected}
                onMouseEnter={() => setCursor(idx)}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(m.id)}
                className={[
                  'relative flex items-baseline gap-2 px-3 py-1.5 text-[13px] font-mono cursor-pointer',
                  selected
                    ? 'text-ink-1 bg-gradient-brand-soft'
                    : active
                      ? 'bg-surface-2 text-ink-1'
                      : 'text-ink-2',
                ].join(' ')}
              >
                <span className="truncate flex-1">{m.id}</span>
                {typeof m.downloads === 'number' && (
                  <span className="text-[0.6rem] text-ink-3 shrink-0">
                    ↓ {formatCount(m.downloads)}
                  </span>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}
