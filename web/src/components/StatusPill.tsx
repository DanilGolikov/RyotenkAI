import type { Status } from '../api/types'

const CLS: Record<Status, string> = {
  completed:   'pill pill-ok',
  running:     'pill pill-run',
  failed:      'pill pill-err',
  interrupted: 'pill pill-warn',
  stale:       'pill pill-idle',
  skipped:     'pill pill-skip',
  pending:     'pill pill-idle',
  unknown:     'pill pill-idle',
}

const LABEL: Record<Status, string> = {
  completed: 'completed',
  running: 'running',
  failed: 'failed',
  interrupted: 'interrupted',
  stale: 'stale',
  skipped: 'skipped',
  pending: 'pending',
  unknown: 'unknown',
}

const DOT: Record<Status, string> = {
  completed:   'bg-status-ok',
  running:     'bg-status-run',
  failed:      'bg-status-err',
  interrupted: 'bg-status-warn',
  stale:       'bg-ink-faint',
  skipped:     'bg-violet-400',
  pending:     'bg-ink-faint',
  unknown:     'bg-ink-faint',
}

export function StatusPill({
  status,
  compact = false,
}: {
  status: Status
  compact?: boolean
}) {
  return (
    <span className={CLS[status]}>
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${DOT[status]} ${status === 'running' ? 'animate-pulse' : ''}`} />
      {!compact && LABEL[status]}
    </span>
  )
}
