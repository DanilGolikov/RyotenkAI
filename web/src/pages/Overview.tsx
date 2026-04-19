import { Link } from 'react-router-dom'
import { KpiStrip } from '../components/KpiStrip'
import { RunRow } from '../components/RunRow'
import { useKpis } from '../api/hooks/useKpis'
import { Card, EmptyState, SectionHeader, Spinner } from '../components/ui'

export function Overview() {
  const { data, isLoading, error } = useKpis()

  const activeRuns = data?.flat.filter((r) => r.status === 'running') ?? []
  const recentRuns = (data?.flat ?? []).slice(0, 8)

  return (
    <div className="p-5 space-y-5 max-w-[1400px]">
      <section className="space-y-1">
        <h1 className="text-2xl font-semibold gradient-text">Overview</h1>
        <p className="text-xs text-ink-mute">
          Pipeline fleet at a glance — live activity, recent history, and health.
        </p>
      </section>

      <KpiStrip />

      {error ? (
        <div className="text-xs text-status-err bg-status-err/10 border border-status-err/30 px-3 py-2 rounded">
          {(error as Error).message}
        </div>
      ) : null}

      <section className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-4">
        <Card>
          <SectionHeader
            title="Live runs"
            subtitle={activeRuns.length === 0 ? 'nothing running right now' : `${activeRuns.length} active`}
            action={<Link to="/runs" className="text-2xs text-violet-300 hover:text-violet-400">all runs →</Link>}
          />
          {isLoading && <div className="text-sm text-ink-mute flex gap-2 items-center"><Spinner /> loading</div>}
          {!isLoading && activeRuns.length === 0 && (
            <EmptyState
              title="Idle"
              hint="launch a run from the Launch page or ⌘K"
              action={
                <Link to="/launch" className="btn-primary inline-flex">
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M5 12h14" /><path d="M12 5l7 7-7 7" />
                  </svg>
                  Start a run
                </Link>
              }
            />
          )}
          <ul className="space-y-1.5">
            {activeRuns.map((run) => (
              <li key={run.run_id}>
                <Link to={`/runs/${encodeURIComponent(run.run_id)}`} className="block">
                  <RunRow run={run} selected={false} onSelect={() => undefined} />
                </Link>
              </li>
            ))}
          </ul>
        </Card>

        <Card>
          <SectionHeader
            title="Recent"
            subtitle="most recent runs across all groups"
            action={<Link to="/runs" className="text-2xs text-violet-300 hover:text-violet-400">open runs →</Link>}
          />
          {isLoading && <div className="text-sm text-ink-mute flex gap-2 items-center"><Spinner /> loading</div>}
          {!isLoading && recentRuns.length === 0 && <EmptyState title="No history yet" />}
          <ul className="space-y-1.5">
            {recentRuns.map((run) => (
              <li key={`r:${run.run_id}`}>
                <Link to={`/runs/${encodeURIComponent(run.run_id)}`} className="block">
                  <RunRow run={run} selected={false} onSelect={() => undefined} />
                </Link>
              </li>
            ))}
          </ul>
        </Card>
      </section>
    </div>
  )
}
