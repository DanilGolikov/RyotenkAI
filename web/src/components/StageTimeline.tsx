import type { StageRun, Status } from '../api/types'
import { formatDuration } from '../lib/format'

// Segment fills — semantic only. No brand colour here.
const SEG_BG: Record<Status, string> = {
  completed:   'bg-ok/50',
  running:     'bg-info/70',
  failed:      'bg-err/65',
  interrupted: 'bg-warn/60',
  skipped:     'bg-brand-alt/40',
  stale:       'bg-idle/40',
  pending:     'bg-surface-3',
  unknown:     'bg-surface-3',
}

const DOT_BG: Record<Status, string> = {
  completed:   'bg-ok',
  running:     'bg-info',
  failed:      'bg-err',
  interrupted: 'bg-warn',
  skipped:     'bg-brand-alt',
  stale:       'bg-idle',
  pending:     'bg-idle',
  unknown:     'bg-idle',
}

export function StageTimeline({
  stages,
  onSelect,
  selected,
}: {
  stages: StageRun[]
  onSelect?: (stageName: string) => void
  selected?: string | null
}) {
  if (!stages.length) return null

  return (
    <div className="space-y-3">
      <div className="flex items-stretch gap-[3px]">
        {stages.map((stage) => {
          const isRunning = stage.status === 'running'
          const isSelected = selected === stage.stage_name
          return (
            <button
              key={stage.stage_name}
              type="button"
              onClick={() => onSelect?.(stage.stage_name)}
              title={`${stage.stage_name} · ${stage.status}`}
              className={[
                'group relative flex-1 h-8 rounded-md overflow-hidden transition border',
                isSelected ? 'border-brand' : 'border-line-1 hover:border-line-2',
              ].join(' ')}
            >
              <div className={`absolute inset-0 ${SEG_BG[stage.status]}`}>
                {isRunning && (
                  <div className="absolute inset-0 bg-info/30 animate-pulse" />
                )}
              </div>
              <div className="relative flex items-center h-full px-2 text-2xs text-ink-1/90">
                <span className="truncate">{stage.stage_name}</span>
              </div>
            </button>
          )
        })}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-2xs text-ink-3">
        {stages.map((stage) => (
          <div key={`leg:${stage.stage_name}`} className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${DOT_BG[stage.status]}`} />
            <span className="text-ink-2">{stage.stage_name}</span>
            <span>· {stage.status}</span>
            {stage.duration_seconds != null && (
              <span className="text-ink-4">· {formatDuration(stage.duration_seconds)}</span>
            )}
            {stage.mode_label && stage.mode_label !== 'executed' && (
              <span className="text-brand-alt">· {stage.mode_label}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
