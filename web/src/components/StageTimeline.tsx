import type { StageRun, Status } from '../api/types'
import { formatDuration } from '../lib/format'

const SEG_BG: Record<Status, string> = {
  completed:   'bg-status-ok/60',
  running:     'bg-status-run/80',
  failed:      'bg-status-err/80',
  interrupted: 'bg-status-warn/70',
  skipped:     'bg-violet-500/50',
  stale:       'bg-ink-faint/40',
  pending:     'bg-surface-4',
  unknown:     'bg-surface-4',
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
    <div className="space-y-2">
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
                'group relative flex-1 h-8 rounded-md overflow-hidden transition',
                'border',
                isSelected ? 'border-burgundy-400 ring-2 ring-burgundy-400/40' : 'border-line-1 hover:border-violet-400',
              ].join(' ')}
            >
              <div className={`absolute inset-0 ${SEG_BG[stage.status]}`}>
                {isRunning && (
                  <div className="absolute inset-0 bg-gradient-brand opacity-50 animate-pulse" />
                )}
              </div>
              <div className="relative flex items-center h-full px-2 text-2xs text-ink/90">
                <span className="truncate">{stage.stage_name}</span>
              </div>
            </button>
          )
        })}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-2xs text-ink-mute">
        {stages.map((stage) => (
          <div key={`leg:${stage.stage_name}`} className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${SEG_BG[stage.status]}`} />
            <span className="text-ink-dim">{stage.stage_name}</span>
            <span>· {stage.status}</span>
            {stage.duration_seconds != null && (
              <span className="text-ink-faint">· {formatDuration(stage.duration_seconds)}</span>
            )}
            {stage.mode_label && stage.mode_label !== 'executed' && (
              <span className="text-violet-300">· {stage.mode_label}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
