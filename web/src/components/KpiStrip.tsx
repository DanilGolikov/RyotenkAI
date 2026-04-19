import type { ReactNode } from 'react'
import { useKpis } from '../api/hooks/useKpis'
import { formatDuration } from '../lib/format'

function Kpi({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: 'default' | 'ok' | 'warn' | 'err' | 'run'
}) {
  const valueTone =
    tone === 'ok' ? 'text-status-ok' :
    tone === 'warn' ? 'text-status-warn' :
    tone === 'err' ? 'text-status-err' :
    tone === 'run' ? 'text-status-run' :
    'text-ink'
  return (
    <div className="card gradient-border px-4 py-3 min-w-[170px] flex-1 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-brand-soft opacity-0 hover:opacity-100 transition pointer-events-none" />
      <div className="relative">
        <div className="text-2xs uppercase tracking-wider text-ink-mute">{label}</div>
        <div className={`text-2xl font-semibold mt-1 ${valueTone}`}>{value}</div>
        {hint && <div className="text-2xs text-ink-mute mt-1">{hint}</div>}
      </div>
    </div>
  )
}

export function KpiStrip() {
  const { data } = useKpis()
  if (!data) {
    return (
      <div className="flex gap-3">
        <div className="card px-4 py-3 flex-1 min-w-[170px] h-[78px] animate-pulse bg-surface-2" />
        <div className="card px-4 py-3 flex-1 min-w-[170px] h-[78px] animate-pulse bg-surface-2" />
        <div className="card px-4 py-3 flex-1 min-w-[170px] h-[78px] animate-pulse bg-surface-2" />
        <div className="card px-4 py-3 flex-1 min-w-[170px] h-[78px] animate-pulse bg-surface-2" />
      </div>
    )
  }
  const successPct = data.successRate == null ? '—' : `${Math.round(data.successRate * 100)}%`
  return (
    <div className="flex gap-3 flex-wrap">
      <Kpi
        label="Active runs"
        value={data.activeRuns}
        tone={data.activeRuns > 0 ? 'run' : 'default'}
        hint={data.activeRuns > 0 ? 'live' : 'idle'}
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
        tone={data.successRate != null && data.successRate < 0.5 ? 'warn' : data.successRate != null && data.successRate >= 0.8 ? 'ok' : 'default'}
      />
      <Kpi
        label="Avg duration"
        value={data.avgDurationSec ? formatDuration(data.avgDurationSec) : '—'}
      />
    </div>
  )
}
