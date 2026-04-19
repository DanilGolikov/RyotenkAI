import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCommandPalette } from '../hooks/useCommandPalette'
import { useRuns } from '../api/hooks/useRuns'
import { StatusPill } from './StatusPill'

export function CommandPalette() {
  const cmdk = useCommandPalette()
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const navigate = useNavigate()
  const { data } = useRuns()
  const inputRef = useRef<HTMLInputElement | null>(null)

  // Global hotkey binding
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const mod = event.metaKey || event.ctrlKey
      if (mod && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        cmdk.toggle()
      }
      if (event.key === 'Escape' && cmdk.isOpen) {
        cmdk.setOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cmdk])

  useEffect(() => {
    if (cmdk.isOpen) {
      setTimeout(() => inputRef.current?.focus(), 10)
      setQuery('')
      setCursor(0)
    }
  }, [cmdk.isOpen])

  const results = useMemo(() => {
    const runs = data ? Object.values(data.groups).flat() : []
    const q = query.trim().toLowerCase()
    const base = [
      {
        kind: 'nav' as const,
        id: 'overview',
        label: 'Go to Overview',
        subtitle: 'dashboard',
        href: '/',
      },
      {
        kind: 'nav' as const,
        id: 'runs',
        label: 'Go to Runs',
        subtitle: 'run list + detail',
        href: '/runs',
      },
      {
        kind: 'nav' as const,
        id: 'launch',
        label: 'Launch new run',
        subtitle: 'start pipeline',
        href: '/launch',
      },
    ]
    const runItems = runs.map((run) => ({
      kind: 'run' as const,
      id: run.run_id,
      label: run.run_id,
      subtitle: `${run.config_name} · ${run.created_at}`,
      href: `/runs/${encodeURIComponent(run.run_id)}`,
      status: run.status,
    }))
    const all = [...base, ...runItems]
    if (!q) return all.slice(0, 12)
    return all.filter((item) =>
      item.label.toLowerCase().includes(q) || (item.subtitle ?? '').toLowerCase().includes(q),
    ).slice(0, 20)
  }, [data, query])

  useEffect(() => {
    setCursor(0)
  }, [query])

  if (!cmdk.isOpen) return null

  function activate(idx: number) {
    const item = results[idx]
    if (!item) return
    navigate(item.href)
    cmdk.setOpen(false)
  }

  return (
    <div
      role="dialog"
      aria-modal
      className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh] bg-black/60 backdrop-blur-sm"
      onClick={() => cmdk.setOpen(false)}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        className="w-full max-w-xl mx-4 rounded-xl bg-surface-1 border border-line-2 shadow-card overflow-hidden"
      >
        <div className="flex items-center gap-2 px-4 h-12 border-b border-line-1">
          <svg className="w-4 h-4 text-ink-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="7" />
            <path d="M21 21l-4.3-4.3" />
          </svg>
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search runs, go to page, launch…"
            className="flex-1 bg-transparent outline-none text-sm placeholder:text-ink-4"
            onKeyDown={(event) => {
              if (event.key === 'ArrowDown') {
                event.preventDefault()
                setCursor((c) => Math.min(results.length - 1, c + 1))
              } else if (event.key === 'ArrowUp') {
                event.preventDefault()
                setCursor((c) => Math.max(0, c - 1))
              } else if (event.key === 'Enter') {
                event.preventDefault()
                activate(cursor)
              }
            }}
          />
          <kbd className="kbd">esc</kbd>
        </div>
        <ul className="max-h-[50vh] overflow-auto py-2">
          {results.length === 0 && (
            <li className="px-4 py-6 text-center text-sm text-ink-3">no matches</li>
          )}
          {results.map((item, idx) => (
            <li key={`${item.kind}:${item.id}`}>
              <button
                type="button"
                onClick={() => activate(idx)}
                onMouseEnter={() => setCursor(idx)}
                className={`w-full flex items-center gap-3 px-4 py-2 text-left text-sm ${
                  idx === cursor ? 'bg-surface-3 text-ink' : 'text-ink-2'
                }`}
              >
                <span className={`w-1.5 h-5 rounded ${idx === cursor ? 'bg-gradient-brand' : 'bg-transparent'}`} />
                <span className="flex-1 truncate">{item.label}</span>
                {item.kind === 'run' && <StatusPill status={item.status} compact />}
                <span className="text-ink-4 text-xs truncate max-w-[40%]">{item.subtitle}</span>
              </button>
            </li>
          ))}
        </ul>
        <div className="px-4 py-2 text-2xs text-ink-3 border-t border-line-1 flex gap-4">
          <span className="flex gap-1 items-center"><kbd className="kbd">↑</kbd><kbd className="kbd">↓</kbd> navigate</span>
          <span className="flex gap-1 items-center"><kbd className="kbd">↵</kbd> open</span>
          <span className="flex gap-1 items-center"><kbd className="kbd">esc</kbd> close</span>
        </div>
      </div>
    </div>
  )
}
