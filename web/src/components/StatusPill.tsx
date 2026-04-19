import type { Status } from '../api/types'

const PILL_CLS: Record<Status, string> = {
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

const ICON_PROPS = {
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2.4,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
}

function StatusIcon({ status, className }: { status: Status; className: string }) {
  switch (status) {
    case 'completed':
      // check-mark circle
      return (
        <svg className={className} {...ICON_PROPS}>
          <circle cx="8" cy="8" r="6.5" fill="currentColor" opacity="0.3" stroke="none" />
          <path d="M5.2 8.2l2 2 3.6-4" />
        </svg>
      )
    case 'failed':
      // x circle
      return (
        <svg className={className} {...ICON_PROPS}>
          <circle cx="8" cy="8" r="6.5" fill="currentColor" opacity="0.3" stroke="none" />
          <path d="M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8" />
        </svg>
      )
    case 'running':
      // play triangle (pulses via parent dot)
      return (
        <svg className={`${className} animate-pulse`} {...ICON_PROPS}>
          <circle cx="8" cy="8" r="6.5" fill="currentColor" opacity="0.32" stroke="none" />
          <path d="M6.6 5.4l4 2.6-4 2.6z" fill="currentColor" stroke="none" />
        </svg>
      )
    case 'interrupted':
      // pause bars
      return (
        <svg className={className} {...ICON_PROPS}>
          <circle cx="8" cy="8" r="6.5" fill="currentColor" opacity="0.3" stroke="none" />
          <path d="M6.5 5.5v5M9.5 5.5v5" />
        </svg>
      )
    case 'skipped':
      // skip-forward
      return (
        <svg className={className} {...ICON_PROPS}>
          <circle cx="8" cy="8" r="6.5" fill="currentColor" opacity="0.3" stroke="none" />
          <path d="M5.5 5.5l3.2 2.5-3.2 2.5z" fill="currentColor" stroke="none" />
          <path d="M10.5 5.5v5" />
        </svg>
      )
    case 'stale':
      // dashed circle
      return (
        <svg className={className} {...ICON_PROPS}>
          <circle cx="8" cy="8" r="5.5" strokeDasharray="2 2" />
        </svg>
      )
    case 'pending':
    case 'unknown':
    default:
      // open circle
      return (
        <svg className={className} {...ICON_PROPS}>
          <circle cx="8" cy="8" r="5.5" />
        </svg>
      )
  }
}

export function StatusPill({
  status,
  compact = false,
}: {
  status: Status
  compact?: boolean
}) {
  return (
    <span className={PILL_CLS[status]} title={compact ? LABEL[status] : undefined}>
      <StatusIcon status={status} className="w-3 h-3" />
      {!compact && LABEL[status]}
    </span>
  )
}
