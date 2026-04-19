import type { RunSummary } from '../api/types'
import { StatusPill } from './StatusPill'
import { formatDuration } from '../lib/format'

const STATUS_BAR: Record<string, string> = {
  completed:   'bg-status-ok',
  running:     'bg-gradient-brand',
  failed:      'bg-status-err',
  interrupted: 'bg-status-warn',
  skipped:     'bg-violet-400',
  stale:       'bg-ink-faint',
  pending:     'bg-ink-faint',
  unknown:     'bg-ink-faint',
}

export function RunRow({
  run,
  selected,
  onSelect,
}: {
  run: RunSummary
  selected: boolean
  onSelect: (runId: string) => void
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(run.run_id)}
      className={[
        'w-full text-left flex gap-3 items-center px-3 py-2.5 rounded-md border transition',
        'relative overflow-hidden',
        selected
          ? 'bg-surface-3 border-burgundy-400/50 shadow-glow-burgundy'
          : 'bg-surface-2 border-line-1 hover:border-violet-400 hover:bg-surface-3',
      ].join(' ')}
    >
      <span className={`absolute left-0 top-0 bottom-0 w-[3px] ${STATUS_BAR[run.status] ?? 'bg-ink-faint'}`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`font-medium text-sm truncate ${selected ? 'text-ink' : 'text-ink-dim'}`}>
            {run.run_id}
          </span>
          <StatusPill status={run.status} compact />
        </div>
        <div className="flex gap-3 mt-1 text-2xs text-ink-mute">
          <span className="truncate max-w-[180px]">{run.config_name}</span>
          <span>· {run.attempts} attempt{run.attempts === 1 ? '' : 's'}</span>
          {run.duration_seconds != null && <span>· {formatDuration(run.duration_seconds)}</span>}
        </div>
      </div>
      <div className="text-2xs text-ink-mute whitespace-nowrap">{run.created_at}</div>
    </button>
  )
}
