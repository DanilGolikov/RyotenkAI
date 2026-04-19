import { useMemo } from 'react'
import { useRuns } from './useRuns'
import type { RunSummary } from '../types'

export interface Kpi {
  activeRuns: number
  totalRuns: number
  failuresLast24h: number
  successRate: number | null
  avgDurationSec: number | null
  flat: RunSummary[]
  latest: RunSummary | null
}

function flatten(groups: Record<string, RunSummary[]>): RunSummary[] {
  const rows: RunSummary[] = []
  for (const group of Object.values(groups)) rows.push(...group)
  return rows
}

export function useKpis(): { data: Kpi | null; isLoading: boolean; error: unknown } {
  const { data, isLoading, error } = useRuns()
  const kpi = useMemo<Kpi | null>(() => {
    if (!data) return null
    const flat = flatten(data.groups)
    if (flat.length === 0) {
      return {
        activeRuns: 0,
        totalRuns: 0,
        failuresLast24h: 0,
        successRate: null,
        avgDurationSec: null,
        flat,
        latest: null,
      }
    }
    const sorted = [...flat].sort((a, b) => b.created_ts - a.created_ts)
    const nowSec = Date.now() / 1000
    const last24hCutoff = nowSec - 24 * 3600
    const last24h = sorted.filter((r) => r.created_ts >= last24hCutoff)
    const terminals = sorted.filter((r) =>
      ['completed', 'failed', 'interrupted'].includes(r.status),
    )
    const completed = terminals.filter((r) => r.status === 'completed')
    const durations = sorted.map((r) => r.duration_seconds).filter((d): d is number => typeof d === 'number' && d > 0)
    return {
      activeRuns: sorted.filter((r) => r.status === 'running').length,
      totalRuns: sorted.length,
      failuresLast24h: last24h.filter((r) => r.status === 'failed').length,
      successRate: terminals.length === 0 ? null : completed.length / terminals.length,
      avgDurationSec: durations.length === 0 ? null : durations.reduce((a, b) => a + b, 0) / durations.length,
      flat: sorted,
      latest: sorted[0] ?? null,
    }
  }, [data])

  return { data: kpi, isLoading, error }
}
