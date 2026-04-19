import { useMemo } from 'react'
import { useRuns } from './useRuns'
import type { RunSummary } from '../types'

export interface Kpi {
  // Core counts
  activeRuns: number
  totalRuns: number

  // Monitoring metrics
  runsToday: number
  failuresLast24h: number
  failures24h: RunSummary[]
  successRate7d: number | null
  medianDuration7dSec: number | null

  // Derived refs
  flat: RunSummary[]
  latest: RunSummary | null
  mostRecentTerminal: RunSummary | null

  // Legacy / kept for any other callers
  successRate: number | null
  avgDurationSec: number | null
}

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'interrupted', 'stale'])

function flatten(groups: Record<string, RunSummary[]>): RunSummary[] {
  const rows: RunSummary[] = []
  for (const group of Object.values(groups)) rows.push(...group)
  return rows
}

function toTs(value: string | null | undefined): number | null {
  if (!value) return null
  const t = Date.parse(value)
  if (Number.isNaN(t)) return null
  return t / 1000
}

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

export function useKpis(): { data: Kpi | null; isLoading: boolean; error: unknown } {
  const { data, isLoading, error } = useRuns()
  const kpi = useMemo<Kpi | null>(() => {
    if (!data) return null
    const flat = flatten(data.groups)
    const nowSec = Date.now() / 1000
    const last24h = nowSec - 24 * 3600
    const last7d = nowSec - 7 * 24 * 3600

    if (flat.length === 0) {
      return {
        activeRuns: 0,
        totalRuns: 0,
        runsToday: 0,
        failuresLast24h: 0,
        failures24h: [],
        successRate7d: null,
        medianDuration7dSec: null,
        flat,
        latest: null,
        mostRecentTerminal: null,
        successRate: null,
        avgDurationSec: null,
      }
    }

    const sorted = [...flat].sort((a, b) => b.created_ts - a.created_ts)

    // Prefer started_at for "today" — when the run actually began running.
    const startedOrCreated = (r: RunSummary) => toTs(r.started_at) ?? r.created_ts
    const runsToday = sorted.filter((r) => startedOrCreated(r) >= last24h).length

    const terminals = sorted.filter((r) => TERMINAL_STATUSES.has(r.status))
    const completedAll = terminals.filter((r) => r.status === 'completed')

    // 7-day windows use completed_at as the "end time" when available.
    const terminals7d = terminals.filter((r) => (toTs(r.completed_at) ?? r.created_ts) >= last7d)
    const completed7d = terminals7d.filter((r) => r.status === 'completed')
    const durations7d = terminals7d
      .map((r) => r.duration_seconds)
      .filter((d): d is number => typeof d === 'number' && d > 0)

    const failures24hList = sorted.filter(
      (r) => r.status === 'failed' && (toTs(r.completed_at) ?? r.created_ts) >= last24h,
    )
    const mostRecentTerminal = sorted.find((r) => TERMINAL_STATUSES.has(r.status)) ?? null

    // Legacy fields
    const durationsAll = sorted
      .map((r) => r.duration_seconds)
      .filter((d): d is number => typeof d === 'number' && d > 0)
    const successRateAll = terminals.length === 0 ? null : completedAll.length / terminals.length

    return {
      activeRuns: sorted.filter((r) => r.status === 'running').length,
      totalRuns: sorted.length,
      runsToday,
      failuresLast24h: failures24hList.length,
      failures24h: failures24hList,
      successRate7d: terminals7d.length === 0 ? null : completed7d.length / terminals7d.length,
      medianDuration7dSec: median(durations7d),
      flat: sorted,
      latest: sorted[0] ?? null,
      mostRecentTerminal,
      successRate: successRateAll,
      avgDurationSec: durationsAll.length === 0 ? null : durationsAll.reduce((a, b) => a + b, 0) / durationsAll.length,
    }
  }, [data])

  return { data: kpi, isLoading, error }
}
