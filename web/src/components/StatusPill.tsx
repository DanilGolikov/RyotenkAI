import type { Status } from '../api/types'

const CLS: Record<Status, string> = {
  completed:   'pill pill-ok',
  running:     'pill pill-info',
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
  completed:   'bg-ok',
  running:     'bg-info',
  failed:      'bg-err',
  interrupted: 'bg-warn',
  stale:       'bg-idle',
  skipped:     'bg-brand-alt',
  pending:     'bg-idle',
  unknown:     'bg-idle',
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
