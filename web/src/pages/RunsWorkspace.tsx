import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useRuns } from '../api/hooks/useRuns'
import { useRun } from '../api/hooks/useRun'
import { useAttempt, useStages } from '../api/hooks/useAttempt'
import { useInterrupt } from '../api/hooks/useLaunch'
import { useDeleteRun } from '../api/hooks/useDelete'
import { LaunchModal } from '../components/LaunchModal'
import { LogDock } from '../components/LogDock'
import { RunRow } from '../components/RunRow'
import { StageTimeline } from '../components/StageTimeline'
import { StatusPill } from '../components/StatusPill'
import { Card, EmptyState, SectionHeader, Spinner } from '../components/ui'
import { formatDateTime, formatDuration } from '../lib/format'
import type { RunSummary } from '../api/types'

type StatusFilter = 'all' | 'running' | 'failed' | 'completed'

export function RunsWorkspace() {
  const params = useParams<{ runId?: string; attemptNo?: string }>()
  const navigate = useNavigate()
  const { data: runs, isLoading: runsLoading } = useRuns()
  const [filter, setFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')

  const flat = useMemo(() => {
    if (!runs) return []
    return Object.values(runs.groups).flat()
  }, [runs])

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase()
    return flat.filter((run) => {
      if (statusFilter !== 'all') {
        if (statusFilter === 'running' && run.status !== 'running') return false
        if (statusFilter === 'failed' && run.status !== 'failed') return false
        if (statusFilter === 'completed' && run.status !== 'completed') return false
      }
      if (!q) return true
      return (
        run.run_id.toLowerCase().includes(q) ||
        run.config_name.toLowerCase().includes(q)
      )
    })
  }, [flat, filter, statusFilter])

  const selectedRunId = params.runId ? decodeURIComponent(params.runId) : undefined

  // Auto-select first run when none chosen yet
  useEffect(() => {
    if (!selectedRunId && filtered.length > 0) {
      navigate(`/runs/${encodeURIComponent(filtered[0].run_id)}`, { replace: true })
    }
  }, [selectedRunId, filtered, navigate])

  return (
    <div className="h-[calc(100vh-56px)] grid grid-cols-[360px_1fr] min-h-0">
      <RunsPanel
        runs={filtered}
        total={flat.length}
        loading={runsLoading}
        filter={filter}
        onFilter={setFilter}
        statusFilter={statusFilter}
        onStatusFilter={setStatusFilter}
        selectedRunId={selectedRunId}
        onSelect={(id) => navigate(`/runs/${encodeURIComponent(id)}`)}
      />
      <div className="min-w-0 overflow-auto">
        {selectedRunId ? (
          <RunDetailPanel runId={selectedRunId} />
        ) : (
          <div className="h-full flex items-center justify-center text-ink-mute text-sm">
            {runsLoading ? 'loading runs…' : 'no run selected'}
          </div>
        )}
      </div>
    </div>
  )
}

function RunsPanel({
  runs,
  total,
  loading,
  filter,
  onFilter,
  statusFilter,
  onStatusFilter,
  selectedRunId,
  onSelect,
}: {
  runs: RunSummary[]
  total: number
  loading: boolean
  filter: string
  onFilter: (v: string) => void
  statusFilter: StatusFilter
  onStatusFilter: (v: StatusFilter) => void
  selectedRunId?: string
  onSelect: (runId: string) => void
}) {
  const filters: { id: StatusFilter; label: string }[] = [
    { id: 'all',       label: `all · ${total}` },
    { id: 'running',   label: 'live' },
    { id: 'failed',    label: 'failed' },
    { id: 'completed', label: 'ok' },
  ]
  return (
    <aside className="flex flex-col min-h-0 border-r border-line-1 bg-surface-0/40">
      <div className="p-3 border-b border-line-1 space-y-2">
        <input
          value={filter}
          onChange={(event) => onFilter(event.target.value)}
          placeholder="filter runs…"
          className="w-full bg-surface-2 border border-line-2 rounded-md px-3 py-1.5 text-sm placeholder:text-ink-faint focus:border-burgundy-400 focus:outline-none"
        />
        <div className="flex gap-1.5 flex-wrap">
          {filters.map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => onStatusFilter(f.id)}
              className={[
                'px-2 py-0.5 rounded-full text-2xs border transition',
                statusFilter === f.id
                  ? 'bg-gradient-brand text-white border-transparent'
                  : 'border-line-2 text-ink-mute hover:text-ink hover:border-violet-400',
              ].join(' ')}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-2 space-y-1.5">
        {loading && runs.length === 0 && (
          <div className="text-sm text-ink-mute p-4 flex items-center gap-2"><Spinner /> loading</div>
        )}
        {!loading && runs.length === 0 && (
          <EmptyState title="no runs match" hint="adjust filters or start one from Launch" />
        )}
        {runs.map((run) => (
          <RunRow
            key={run.run_id}
            run={run}
            selected={run.run_id === selectedRunId}
            onSelect={onSelect}
          />
        ))}
      </div>
    </aside>
  )
}

function RunDetailPanel({ runId }: { runId: string }) {
  const { data: run, isLoading, error } = useRun(runId)
  const navigate = useNavigate()
  const params = useParams<{ attemptNo?: string }>()
  const [launchOpen, setLaunchOpen] = useState(false)
  const interruptMut = useInterrupt(runId)
  const deleteMut = useDeleteRun()

  const attemptNo = useMemo(() => {
    if (params.attemptNo) return Number(params.attemptNo)
    if (run?.running_attempt_no) return run.running_attempt_no
    if (run && run.attempts.length > 0) return run.attempts[run.attempts.length - 1].attempt_no
    return null
  }, [params.attemptNo, run])

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center text-ink-mute text-sm">
        <Spinner /> loading {runId}
      </div>
    )
  }
  if (error) return <div className="p-6 text-status-err text-sm">{(error as Error).message}</div>
  if (!run) return null

  const mlflowHref =
    run.root_mlflow_run_id && run.mlflow_runtime_tracking_uri
      ? `${run.mlflow_runtime_tracking_uri}/#/experiments/0/runs/${run.root_mlflow_run_id}`
      : null

  return (
    <div className="p-5 space-y-5 max-w-[1400px]">
      <section className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <StatusPill status={run.status} />
            <h1 className="text-xl font-semibold text-ink truncate">{run.logical_run_id}</h1>
          </div>
          <div className="mt-1 text-xs text-ink-mute flex gap-4 flex-wrap">
            <span>config: <span className="font-mono text-ink-dim">{run.config_path || '—'}</span></span>
            <span>attempts: <span className="text-ink-dim">{run.attempts.length}</span></span>
            {run.is_locked && run.lock_pid != null && (
              <span className="text-status-run">pid {run.lock_pid}</span>
            )}
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          {mlflowHref && (
            <a href={mlflowHref} target="_blank" rel="noreferrer" className="btn-ghost">
              MLflow ↗
            </a>
          )}
          {run.is_locked && (
            <button
              type="button"
              onClick={() => interruptMut.mutate()}
              className="btn-danger-ghost"
            >
              Interrupt
            </button>
          )}
          <button
            type="button"
            disabled={run.is_locked}
            onClick={async () => {
              if (!confirm(`Delete ${runId} (and linked MLflow runs)?`)) return
              await deleteMut.mutateAsync(runId)
              navigate('/runs')
            }}
            className="btn-danger-ghost disabled:opacity-40"
          >
            Delete
          </button>
          <button type="button" onClick={() => setLaunchOpen(true)} className="btn-primary">
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 12h14" /><path d="M12 5l7 7-7 7" />
            </svg>
            Launch
          </button>
        </div>
      </section>

      <section className="card p-4">
        <SectionHeader
          title="Attempts"
          subtitle={run.attempts.length === 0 ? 'no attempts yet — launch the pipeline' : undefined}
        />
        {run.attempts.length === 0 ? (
          <EmptyState title="No attempts" hint="pipeline never ran for this run" />
        ) : (
          <div className="flex gap-2 overflow-x-auto pb-1">
            {run.attempts.map((attempt) => {
              const active = attempt.attempt_no === attemptNo
              return (
                <button
                  key={attempt.attempt_id}
                  type="button"
                  onClick={() => navigate(`/runs/${encodeURIComponent(runId)}/attempts/${attempt.attempt_no}`)}
                  className={[
                    'min-w-[150px] shrink-0 text-left p-3 rounded-md border transition',
                    active
                      ? 'border-burgundy-400/70 bg-surface-3 shadow-glow-burgundy'
                      : 'border-line-1 bg-surface-2 hover:border-violet-400 hover:bg-surface-3',
                  ].join(' ')}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs text-ink-mute">#{attempt.attempt_no}</span>
                    <StatusPill status={attempt.status} compact />
                  </div>
                  <div className="text-2xs text-ink-mute mt-1">{attempt.effective_action}</div>
                  <div className="text-2xs text-ink-faint mt-2">
                    {formatDateTime(attempt.started_at)}
                    {attempt.completed_at && (
                      <> · {formatDuration(
                        (new Date(attempt.completed_at).getTime() - new Date(attempt.started_at).getTime()) / 1000,
                      )}</>
                    )}
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </section>

      {attemptNo && (
        <AttemptSection runId={runId} attemptNo={attemptNo} />
      )}

      <LaunchModal
        runId={runId}
        open={launchOpen}
        onClose={() => setLaunchOpen(false)}
        defaultMode={run.attempts.length === 0 ? 'new_run' : 'resume'}
        defaultConfigPath={run.config_abspath}
      />
    </div>
  )
}

function AttemptSection({ runId, attemptNo }: { runId: string; attemptNo: number }) {
  const attempt = useAttempt(runId, attemptNo)
  const stages = useStages(runId, attemptNo)
  const [selectedStage, setSelectedStage] = useState<string | null>(null)
  const stageList = stages.data?.stages ?? []
  const selected = selectedStage
    ? stageList.find((s) => s.stage_name === selectedStage)
    : null
  const attemptRunning = attempt.data?.status === 'running'

  return (
    <section className="space-y-4">
      <Card>
        <SectionHeader
          title={<>Stages · attempt {attemptNo}</>}
          subtitle={attempt.data ? `duration ${formatDuration(attempt.data.duration_seconds)}` : undefined}
        />
        {stageList.length === 0 ? (
          <EmptyState title="stages pending" hint="waiting for attempt to start" />
        ) : (
          <StageTimeline
            stages={stageList}
            selected={selectedStage}
            onSelect={(name) => setSelectedStage((v) => (v === name ? null : name))}
          />
        )}
        {selected && (
          <div className="mt-4 rounded-md border border-line-2 bg-surface-1 p-3 text-xs space-y-1">
            <div className="flex items-center gap-2">
              <StatusPill status={selected.status} compact />
              <span className="font-medium text-ink">{selected.stage_name}</span>
              {selected.mode_label && <span className="text-violet-300">· {selected.mode_label}</span>}
              {selected.duration_seconds != null && (
                <span className="text-ink-mute">· {formatDuration(selected.duration_seconds)}</span>
              )}
            </div>
            {selected.started_at && (
              <div className="text-ink-mute">started {formatDateTime(selected.started_at)}</div>
            )}
            {selected.completed_at && (
              <div className="text-ink-mute">finished {formatDateTime(selected.completed_at)}</div>
            )}
            {selected.error && (
              <div className="text-status-err font-mono whitespace-pre-wrap mt-1">{selected.error}</div>
            )}
          </div>
        )}
      </Card>

      <LogDock runId={runId} attemptNo={attemptNo} enabled={attemptRunning || true} />
    </section>
  )
}
