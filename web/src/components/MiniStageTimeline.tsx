import type { StageRun, Status } from '../api/types'

const SEG_BG: Record<Status, string> = {
  completed:   'bg-ok/60',
  running:     'bg-info/80',
  failed:      'bg-err/75',
  interrupted: 'bg-warn/70',
  skipped:     'bg-brand-alt/50',
  stale:       'bg-idle/40',
  pending:     'bg-surface-3',
  unknown:     'bg-surface-3',
}

const SIZE = {
  micro: 'h-[3px]',   // inline with status row
  mini:  'h-[6px]',   // under status row
} as const

export function MiniStageTimeline({
  stages,
  variant = 'mini',
  max = 12,
}: {
  stages: StageRun[]
  variant?: keyof typeof SIZE
  max?: number
}) {
  if (!stages.length) return null
  const slice = stages.slice(0, max)
  return (
    <div className={`flex items-stretch gap-[2px] ${SIZE[variant]}`} role="img" aria-label="Stage progress">
      {slice.map((stage) => {
        const running = stage.status === 'running'
        return (
          <div
            key={stage.stage_name}
            title={`${stage.stage_name} · ${stage.status}`}
            className={[
              'flex-1 rounded-[1.5px] overflow-hidden',
              SEG_BG[stage.status],
            ].join(' ')}
          >
            {running && (
              <div className="h-full w-full bg-info/50 animate-pulse" />
            )}
          </div>
        )
      })}
    </div>
  )
}
