import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useRun } from '../api/hooks/useRun'
import { useInterrupt } from '../api/hooks/useLaunch'
import { useDeleteRun } from '../api/hooks/useDelete'
import { LaunchModal } from '../components/LaunchModal'
import { StatusBadge } from '../components/StatusBadge'
import { formatDateTime, formatDuration } from '../lib/format'

export function RunDetail() {
  const params = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const runId = params.runId ? decodeURIComponent(params.runId) : undefined
  const { data, error, isLoading } = useRun(runId)
  const interruptMut = useInterrupt(runId ?? '')
  const deleteMut = useDeleteRun()
  const [launchOpen, setLaunchOpen] = useState(false)

  if (isLoading) return <div className="p-6 text-gray-400">loading…</div>
  if (error) return <div className="p-6 text-rose-400">{(error as Error).message}</div>
  if (!data || !runId) return null

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-xs text-gray-500 mb-1">
            <Link to="/" className="hover:text-gray-300">runs</Link> / {runId}
          </div>
          <h1 className="text-xl flex items-center gap-3"><StatusBadge status={data.status} />{data.logical_run_id}</h1>
          <div className="text-xs text-gray-500 mt-1">config: {data.config_path || '—'}</div>
        </div>
        <div className="flex gap-2">
          {data.root_mlflow_run_id && data.mlflow_runtime_tracking_uri && (
            <a
              href={`${data.mlflow_runtime_tracking_uri}/#/experiments/0/runs/${data.root_mlflow_run_id}`}
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1.5 text-sm border border-surface-muted rounded text-gray-300 hover:text-accent"
            >
              Open MLflow
            </a>
          )}
          <button
            type="button"
            onClick={() => setLaunchOpen(true)}
            className="px-3 py-1.5 text-sm bg-accent text-surface rounded"
          >
            Launch
          </button>
          {data.is_locked && (
            <button
              type="button"
              onClick={() => interruptMut.mutate()}
              className="px-3 py-1.5 text-sm border border-amber-400 text-amber-400 rounded"
            >
              Interrupt
            </button>
          )}
          <button
            type="button"
            disabled={data.is_locked}
            onClick={async () => {
              if (!confirm(`Delete ${runId} and its MLflow runs?`)) return
              await deleteMut.mutateAsync(runId)
              navigate('/')
            }}
            className="px-3 py-1.5 text-sm border border-rose-500 text-rose-400 rounded disabled:opacity-50"
          >
            Delete
          </button>
        </div>
      </div>

      <section className="bg-surface-raised rounded border border-surface-muted">
        <h2 className="px-4 py-2 border-b border-surface-muted text-sm text-gray-400">Attempts</h2>
        {data.attempts.length === 0 ? (
          <div className="px-4 py-6 text-sm text-gray-500">no attempts yet</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wider text-gray-500 border-b border-surface-muted">
              <tr>
                <th className="px-4 py-2">#</th>
                <th className="px-4 py-2">Status</th>
                <th className="px-4 py-2">Action</th>
                <th className="px-4 py-2">Started</th>
                <th className="px-4 py-2">Finished</th>
                <th className="px-4 py-2">Duration</th>
              </tr>
            </thead>
            <tbody>
              {data.attempts.map((attempt) => {
                const duration = attempt.completed_at && attempt.started_at
                  ? (new Date(attempt.completed_at).getTime() - new Date(attempt.started_at).getTime()) / 1000
                  : null
                return (
                  <tr key={attempt.attempt_id} className="border-b border-surface-muted/50 hover:bg-surface">
                    <td className="px-4 py-2 text-accent">
                      <Link to={`/runs/${encodeURIComponent(runId)}/attempts/${attempt.attempt_no}`}>{attempt.attempt_no}</Link>
                    </td>
                    <td className="px-4 py-2"><StatusBadge status={attempt.status} /></td>
                    <td className="px-4 py-2 text-gray-400">{attempt.effective_action}</td>
                    <td className="px-4 py-2 text-gray-400">{formatDateTime(attempt.started_at)}</td>
                    <td className="px-4 py-2 text-gray-400">{formatDateTime(attempt.completed_at)}</td>
                    <td className="px-4 py-2 text-gray-400">{formatDuration(duration)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </section>

      <LaunchModal
        runId={runId}
        open={launchOpen}
        onClose={() => setLaunchOpen(false)}
        defaultMode={data.attempts.length === 0 ? 'new_run' : 'resume'}
        defaultConfigPath={data.config_abspath}
      />
    </div>
  )
}
