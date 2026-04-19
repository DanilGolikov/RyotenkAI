import type { StageRun, Status } from '../api/types'
import { STATUS_LABELS } from '../lib/statusConstants'
import { STATUS_TEXT_CLASS } from './StatusPill'
import { formatDuration } from '../lib/format'

// Segment fills — semantic only. No brand colour here.
const SEG_BG: Record<Status, string> = {
  completed:   'bg-ok/35',
  running:     'bg-info/55',
  failed:      'bg-err/50',
  interrupted: 'bg-warn/45',
  skipped:     'bg-brand-alt/35',
  stale:       'bg-idle/35',
  pending:     'bg-surface-3',
  unknown:     'bg-surface-3',
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
              title={`${stage.stage_name} · ${STATUS_LABELS[stage.status]}`}
              className={[
                'group relative flex-1 h-14 rounded-md overflow-hidden transition border',
                isSelected ? 'border-brand' : 'border-line-1 hover:border-line-2',
              ].join(' ')}
            >
              <div className={`absolute inset-0 ${SEG_BG[stage.status]}`}>
                {isRunning && (
                  <div className="absolute inset-0 bg-info/30 animate-pulse" />
                )}
              </div>
              <div className="relative flex h-full flex-col items-center justify-center gap-0.5 px-2 text-center">
                <span className="text-2xs text-ink-1/90 truncate max-w-full">
                  {stage.stage_name}
                </span>
                <span
                  className={`text-2xs font-medium tracking-wide ${STATUS_TEXT_CLASS[stage.status]}`}
                >
                  {STATUS_LABELS[stage.status]}
                </span>
                {stage.duration_seconds != null && (
                  <span className="text-[0.6rem] leading-none text-ink-3/80 mt-0.5 truncate max-w-full">
                    {formatDuration(stage.duration_seconds)}
                    {stage.mode_label && stage.mode_label !== 'executed' && (
                      <span className="text-brand-alt"> · {stage.mode_label}</span>
                    )}
                  </span>
                )}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
