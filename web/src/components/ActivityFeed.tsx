import { forwardRef, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { useKpis } from '../api/hooks/useKpis'
import { useRun } from '../api/hooks/useRun'
import { qk } from '../api/queryKeys'
import type { RunSummary, StagesResponse } from '../api/types'
import { formatDuration, timeAgo } from '../lib/format'
import { MiniStageTimeline } from './MiniStageTimeline'
import { StatusPill } from './StatusPill'
import { Card, EmptyState, SectionHeader, Spinner } from './ui'

const CAP = 6

function priority(run: RunSummary, recentFailIds: Set<string>): number {
  if (run.status === 'running') return 0
  if (run.status === 'failed' && recentFailIds.has(run.run_id)) return 1 // recent failure → surface, but below live runs
  if (run.status === 'failed') return 2
  if (run.status === 'interrupted') return 3
  if (run.status === 'pending') return 4
  if (run.status === 'stale') return 5
  if (run.status === 'completed') return 6
  if (run.status === 'skipped') return 7
  return 8
}

function sortForActivity(runs: RunSummary[], recentFailIds: Set<string>): RunSummary[] {
  return [...runs].sort((a, b) => {
    const pa = priority(a, recentFailIds)
    const pb = priority(b, recentFailIds)
    if (pa !== pb) return pa - pb
    return b.created_ts - a.created_ts
  })
}

function truncate(text: string, max = 160): string {
  if (text.length <= max) return text
  return `${text.slice(0, max - 1).trimEnd()}…`
}

function RunningStages({ runId }: { runId: string }) {
  const { data: run } = useRun(runId)
  const attemptNo = run?.running_attempt_no ?? null

  const stagesQuery = useQuery({
    queryKey: attemptNo ? qk.stages(runId, attemptNo) : ['activity-stages-disabled'],
    queryFn: () =>
      api.get<StagesResponse>(
        `/runs/${encodeURIComponent(runId)}/attempts/${attemptNo!}/stages`,
      ),
    enabled: !!attemptNo,
    refetchInterval: 3000,
  })

  const stages = stagesQuery.data?.stages ?? []
  if (stages.length === 0) return null
  const runningIdx = stages.findIndex((s) => s.status === 'running')
  const label =
    runningIdx >= 0 ? `stage ${runningIdx + 1}/${stages.length} · ${stages[runningIdx].stage_name}` : null

  return (
    <div className="mt-2 space-y-1">
      <MiniStageTimeline stages={stages} variant="mini" />
      {label && <div className="text-2xs text-ink-3">{label}</div>}
    </div>
  )
}

function ActivityRow({ run }: { run: RunSummary }) {
  const href = `/runs/${encodeURIComponent(run.run_id)}`
  const isRunning = run.status === 'running'
  const isFailed = run.status === 'failed'

  const elapsed = isRunning
    ? timeAgo(run.started_at ?? run.completed_at ?? null) || 'just started'
    : run.completed_at
      ? timeAgo(run.completed_at)
      : ''

  return (
    <Link
      to={href}
      className="block rounded-md border border-line-1 bg-surface-1 hover:bg-surface-2 hover:border-line-2 transition px-3 py-2.5"
    >
      <div className="flex items-center gap-2 min-w-0">
        <StatusPill status={run.status} compact />
        <span className="font-medium text-sm text-ink-1 truncate flex-1 min-w-0">{run.run_id}</span>
        <span className="text-2xs text-ink-3 whitespace-nowrap">{elapsed}</span>
      </div>
      <div className="mt-1 flex items-center gap-3 text-2xs text-ink-3 min-w-0">
        <span className="font-mono truncate max-w-[260px]">{run.config_name}</span>
        {run.duration_seconds != null && !isRunning && (
          <span>· {formatDuration(run.duration_seconds)}</span>
        )}
        {run.attempts > 1 && <span>· {run.attempts} attempts</span>}
      </div>
      {isFailed && run.error && (
        <div className="mt-1 text-2xs font-mono text-err/90 truncate">
          {truncate(run.error)}
        </div>
      )}
      {isRunning && <RunningStages runId={run.run_id} />}
    </Link>
  )
}

export const ActivityFeed = forwardRef<HTMLElement>(function ActivityFeed(_props, ref) {
  const { data, isLoading, error } = useKpis()
  const rows = useMemo(() => {
    if (!data) return []
    const recentFailIds = new Set(data.failures24h.map((r) => r.run_id))
    return sortForActivity(data.flat, recentFailIds).slice(0, CAP)
  }, [data])

  return (
    <section ref={ref} id="activity" className="scroll-mt-20">
      <Card padding="p-0">
        <div className="px-4 pt-3">
          <SectionHeader
            title="Activity"
            subtitle={
              data && data.flat.length > 0
                ? `showing ${rows.length} of ${data.flat.length}`
                : undefined
            }
            action={
              <Link to="/runs" className="text-2xs text-ink-2 hover:text-ink-1">
                all runs →
              </Link>
            }
          />
        </div>
        <div className="px-3 pb-3 space-y-1.5">
          {error ? (
            <div className="px-3 py-4 text-sm text-err">{(error as Error).message}</div>
          ) : isLoading && rows.length === 0 ? (
            <div className="px-3 py-4 text-sm text-ink-3 flex items-center gap-2">
              <Spinner /> loading
            </div>
          ) : rows.length === 0 ? (
            <EmptyState title="No runs yet" hint="start one from Launch or ⌘K" />
          ) : (
            rows.map((run) => <ActivityRow key={run.run_id} run={run} />)
          )}
        </div>
      </Card>
    </section>
  )
})
