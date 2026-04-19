import { Link, Outlet, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { HealthStatus } from '../api/types'
import { qk } from '../api/queryKeys'

export function Layout() {
  const location = useLocation()
  const { data: health } = useQuery({
    queryKey: qk.health(),
    queryFn: () => api.get<HealthStatus>('/health'),
    refetchInterval: 15_000,
  })

  return (
    <div className="min-h-full flex flex-col">
      <header className="border-b border-surface-muted bg-surface-raised px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link to="/" className="text-accent font-semibold text-lg">RyotenkAI</Link>
          <nav className="flex gap-4 text-sm text-gray-400">
            <Link to="/" className={location.pathname === '/' ? 'text-gray-100' : ''}>Runs</Link>
          </nav>
        </div>
        {health && (
          <div className="text-xs text-gray-500">
            <span className={`status-dot ${health.status === 'ok' ? 'status-completed' : 'status-failed'}`} />
            {health.runs_dir}
            <span className="ml-3">{health.version}</span>
          </div>
        )}
      </header>
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
