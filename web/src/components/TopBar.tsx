import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { qk } from '../api/queryKeys'
import type { HealthStatus } from '../api/types'
import { useCommandPalette } from '../hooks/useCommandPalette'

export function TopBar() {
  const { data: health } = useQuery({
    queryKey: qk.health(),
    queryFn: () => api.get<HealthStatus>('/health'),
    refetchInterval: 15_000,
  })
  const cmdk = useCommandPalette()

  return (
    <header className="sticky top-0 z-20 backdrop-blur bg-surface-0/80 border-b border-line-1">
      <div className="h-14 px-5 flex items-center gap-4">
        <button
          type="button"
          onClick={() => cmdk.setOpen(true)}
          className="flex-1 max-w-[520px] h-9 px-3 flex items-center gap-2 rounded-md text-sm text-ink-3 bg-surface-1 border border-line-2 hover:border-ink-3 hover:text-ink-1 transition"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="7" />
            <path d="M21 21l-4.3-4.3" />
          </svg>
          <span>Search runs, launch…</span>
          <span className="ml-auto flex gap-0.5">
            <kbd className="kbd">⌘</kbd><kbd className="kbd">K</kbd>
          </span>
        </button>

        <div className="ml-auto flex items-center gap-4 text-2xs text-ink-3">
          {health && (
            <div className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${health.status === 'ok' ? 'bg-ok' : 'bg-err'}`} />
              <span className="font-mono truncate max-w-[360px]">{health.runs_dir}</span>
            </div>
          )}
          <span className="text-ink-4 font-mono">{health?.version ?? ''}</span>
        </div>
      </div>
    </header>
  )
}
