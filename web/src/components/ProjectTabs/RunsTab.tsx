import { Link } from 'react-router-dom'
import { useProjectRuns } from '../../api/hooks/useProjects'
import type { ProjectRunEntry } from '../../api/types'

/**
 * Runs tab — surfaces the project's launched-run ledger.
 *
 * Backed by ``GET /projects/{id}/runs`` (Step 6 of Variant 1). Each
 * row links to the run-details page; status / actor / config-hash
 * breadcrumbs surface as compact metadata so the user can identify a
 * run at a glance without drilling in.
 *
 * Empty / loading / error states are intentionally minimal — this is
 * an informational tab, not a transactional surface; an empty state
 * for a fresh project is the steady-state, not a problem.
 */
export function RunsTab({ projectId }: { projectId: string }) {
  const query = useProjectRuns(projectId)

  if (query.isLoading) {
    return <div className="text-xs text-ink-3">Loading runs…</div>
  }

  if (query.isError) {
    return (
      <div className="text-xs text-rose-600">
        Failed to load runs: {(query.error as Error).message}
      </div>
    )
  }

  const runs = query.data?.runs ?? []

  if (runs.length === 0) {
    return (
      <div className="text-xs text-ink-3 space-y-2">
        <div>No runs yet.</div>
        <div className="text-ink-4">
          Launch a pipeline with this project selected and it'll show
          up here automatically.
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="text-2xs uppercase tracking-wider text-ink-4 font-medium">
        {runs.length} run{runs.length === 1 ? '' : 's'}
      </div>
      <ul className="space-y-1.5">
        {runs.map((run) => (
          <RunRow key={`${run.run_id}-${run.started_at}`} run={run} />
        ))}
      </ul>
    </div>
  )
}

function RunRow({ run }: { run: ProjectRunEntry }) {
  return (
    <li className="rounded-md border border-line-1 hover:border-line-2 transition-colors bg-surface-1">
      <Link
        to={`/runs/${encodeURIComponent(run.run_id)}`}
        className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 text-xs"
      >
        <StatusBadge status={run.status} />
        <code className="font-mono text-ink-1 truncate">{run.run_id}</code>
        <span className="text-ink-4 font-mono text-2xs">
          {formatStartedAt(run.started_at)}
        </span>
        {run.actor && (
          <span className="text-ink-3 text-2xs">
            by <span className="text-ink-2">{run.actor}</span>
          </span>
        )}
        {run.config_version_hash && (
          <span
            className="text-ink-4 font-mono text-2xs"
            title={`config_version_hash: ${run.config_version_hash}`}
          >
            #{run.config_version_hash.slice(0, 7)}
          </span>
        )}
      </Link>
    </li>
  )
}

function StatusBadge({ status }: { status: string }) {
  const palette = STATUS_PALETTE[status] ?? STATUS_PALETTE.unknown
  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded text-2xs font-medium ${palette}`}
    >
      {status}
    </span>
  )
}

const STATUS_PALETTE: Record<string, string> = {
  running: 'bg-amber-50 text-amber-800 border border-amber-200',
  completed: 'bg-emerald-50 text-emerald-800 border border-emerald-200',
  failed: 'bg-rose-50 text-rose-800 border border-rose-200',
  interrupted: 'bg-slate-50 text-slate-800 border border-slate-200',
  pending: 'bg-sky-50 text-sky-800 border border-sky-200',
  unknown: 'bg-surface-2 text-ink-3 border border-line-1',
}

function formatStartedAt(iso: string): string {
  // Defensive: malformed strings render as-is rather than crashing.
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}
