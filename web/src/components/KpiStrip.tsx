import type { ReactNode } from 'react'
import { useKpis } from '../api/hooks/useKpis'
import { formatDuration } from '../lib/format'

type Tone = 'default' | 'ok' | 'warn' | 'err' | 'info'

function Kpi({
  label,
  value,
  hint,
  tone = 'default',
  hero = false,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: Tone
  hero?: boolean
}) {
  const valueTone =
    tone === 'ok' ? 'text-ok' :
    tone === 'warn' ? 'text-warn' :
    tone === 'err' ? 'text-err' :
    tone === 'info' ? 'text-info' :
    'text-ink-1'
  return (
    <div className={`${hero ? 'card-hero' : 'card'} px-4 py-3 min-w-0`}>
      <div className="text-2xs uppercase tracking-wider text-ink-3">{label}</div>
      <div className={`text-2xl font-semibold mt-1 tabular-nums ${valueTone}`}>{value}</div>
      {hint && <div className="text-2xs text-ink-3 mt-1">{hint}</div>}
    </div>
  )
}

export function KpiStrip() {
  const { data } = useKpis()
  if (!data) {
    return (
      <div className="grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="card px-4 py-3 h-[78px] animate-pulse bg-surface-2" />
        ))}
      </div>
    )
  }
  const successPct = data.successRate == null ? '—' : `${Math.round(data.successRate * 100)}%`
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-3">
      <Kpi
        label="Active runs"
        value={data.activeRuns}
        tone={data.activeRuns > 0 ? 'info' : 'default'}
        hint={data.activeRuns > 0 ? 'live' : 'idle'}
        hero
      />
      <Kpi
        label="Total runs"
        value={data.totalRuns}
        hint={data.latest ? `latest ${data.latest.created_at}` : '—'}
      />
      <Kpi
        label="Failures · 24h"
        value={data.failuresLast24h}
        tone={data.failuresLast24h > 0 ? 'err' : 'default'}
      />
      <Kpi
        label="Success rate"
        value={successPct}
        tone={
          data.successRate != null && data.successRate < 0.5
            ? 'warn'
            : data.successRate != null && data.successRate >= 0.8
              ? 'ok'
              : 'default'
        }
      />
      <Kpi
        label="Avg duration"
        value={data.avgDurationSec ? formatDuration(data.avgDurationSec) : '—'}
      />
    </div>
  )
}
