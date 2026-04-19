import { Link, useParams } from 'react-router-dom'
import { useAttempt, useStages } from '../api/hooks/useAttempt'
import { StagesTable } from '../components/StagesTable'
import { StatusBadge } from '../components/StatusBadge'
import { LogPanel } from '../components/LogPanel'
import { formatDuration } from '../lib/format'

export function AttemptDetail() {
  const params = useParams<{ runId: string; attemptNo: string }>()
  const runId = params.runId ? decodeURIComponent(params.runId) : undefined
  const attemptNo = params.attemptNo ? Number(params.attemptNo) : undefined
  const attempt = useAttempt(runId, attemptNo)
  const stages = useStages(runId, attemptNo)

  if (!runId || !attemptNo) return null

  return (
    <div className="p-6 space-y-6">
      <div className="text-xs text-gray-500">
        <Link to="/" className="hover:text-gray-300">runs</Link>
        {' / '}
        <Link to={`/runs/${encodeURIComponent(runId)}`} className="hover:text-gray-300">{runId}</Link>
        {' / attempt '}
        {attemptNo}
      </div>

      {attempt.error && <div className="text-rose-400 text-sm">{(attempt.error as Error).message}</div>}
      {attempt.data && (
        <div className="flex items-center gap-4">
          <StatusBadge status={attempt.data.status} />
          <span className="text-sm text-gray-400">{attempt.data.effective_action}</span>
          <span className="text-sm text-gray-500">duration: {formatDuration(attempt.data.duration_seconds)}</span>
          {attempt.data.error && <span className="text-sm text-rose-400 truncate">{attempt.data.error}</span>}
        </div>
      )}

      <section className="bg-surface-raised rounded border border-surface-muted">
        <h2 className="px-4 py-2 border-b border-surface-muted text-sm text-gray-400">Stages</h2>
        <StagesTable stages={stages.data?.stages ?? []} />
      </section>

      <section>
        <h2 className="text-sm text-gray-400 mb-2">Logs</h2>
        <LogPanel runId={runId} attemptNo={attemptNo} enabled={true} />
      </section>
    </div>
  )
}
