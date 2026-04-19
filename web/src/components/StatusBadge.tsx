import type { Status } from '../api/types'

const LABEL: Record<Status, string> = {
  pending: 'pending',
  running: 'running',
  completed: 'completed',
  failed: 'failed',
  interrupted: 'interrupted',
  stale: 'stale',
  skipped: 'skipped',
  unknown: 'unknown',
}

export function StatusBadge({ status }: { status: Status }) {
  return (
    <span className="inline-flex items-center text-xs">
      <span className={`status-dot status-${status}`} />
      {LABEL[status] ?? status}
    </span>
  )
}
