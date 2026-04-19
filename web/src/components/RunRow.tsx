import type { RunSummary } from '../api/types'
import { StatusPill } from './StatusPill'
import { formatDuration } from '../lib/format'

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
        selected
          ? 'bg-surface-3 border-line-2'
          : 'bg-surface-1 border-line-1 hover:bg-surface-2 hover:border-line-2',
      ].join(' ')}
      style={selected ? { boxShadow: 'inset 2px 0 0 #d6305f' } : undefined}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`font-medium text-sm truncate ${selected ? 'text-ink-1' : 'text-ink-2'}`}>
            {run.run_id}
          </span>
          <StatusPill status={run.status} compact />
        </div>
        <div className="flex gap-3 mt-1 text-2xs text-ink-3">
          <span className="truncate max-w-[180px]">{run.config_name}</span>
          <span>· {run.attempts} attempt{run.attempts === 1 ? '' : 's'}</span>
          {run.duration_seconds != null && <span>· {formatDuration(run.duration_seconds)}</span>}
        </div>
      </div>
      <div className="text-2xs text-ink-3 whitespace-nowrap">{run.created_at}</div>
    </button>
  )
}
