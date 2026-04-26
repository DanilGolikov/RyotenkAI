/**
 * Phase 7.2 — Live Training MVP page.
 *
 * Mounted at ``/runs/:runId/live``. Polls the FastAPI control-plane proxy
 * (``src/api/routers/jobs.py``) every couple of seconds for status + new
 * events, and offers a single Stop button. No charts, no WebSockets,
 * no fancy filters — just enough to show the user what the in-pod
 * runner is doing.
 *
 * Design notes
 * ------------
 * - Browser cannot open SSH tunnels itself; the FastAPI server opens a
 *   short-lived tunnel per poll and proxies to the runner. ~1 s of
 *   extra latency per poll is acceptable for an MVP polling UI.
 * - Events are kept client-side in a growing list. We carry the cursor
 *   (``since``) ourselves so each poll fetches only the new slice.
 * - Stop confirmation is a simple ``window.confirm`` — sufficient for
 *   v1; a richer modal can come later if we ever need a grace-time
 *   slider.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'

import {
  type JobEvent,
  type JobSnapshot,
  type JobState,
  useJobEvents,
  useJobStatus,
  useStopJob,
} from '../api/hooks/useJob'
import { Card, EmptyState, SectionHeader, Spinner } from '../components/ui'

// --------------------------------------------------------------------
// Status pill — small, compact, hand-rolled because the global
// ``StatusPill`` is typed against pipeline status (no ``preparing`` etc).
// --------------------------------------------------------------------

const STATE_PILL: Record<JobState, string> = {
  preparing: 'pill pill-info',
  running: 'pill pill-info',
  stopping: 'pill pill-warn',
  completed: 'pill pill-ok',
  failed: 'pill pill-err',
  cancelled: 'pill pill-warn',
}

const STATE_LABEL: Record<JobState, string> = {
  preparing: 'Preparing',
  running: 'Running',
  stopping: 'Stopping',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

function StatePill({ state }: { state: JobState }) {
  return <span className={STATE_PILL[state]}>{STATE_LABEL[state]}</span>
}

// --------------------------------------------------------------------
// Page
// --------------------------------------------------------------------

const TERMINAL_STATES: ReadonlySet<JobState> = new Set([
  'completed',
  'failed',
  'cancelled',
])

export function JobLivePage() {
  const { runId } = useParams<{ runId: string }>()
  const decodedRunId = runId ? decodeURIComponent(runId) : undefined

  const status = useJobStatus(decodedRunId, undefined)
  const snapshot: JobSnapshot | undefined = status.data?.snapshot
  const submission = status.data?.submission
  const isTerminal = snapshot ? TERMINAL_STATES.has(snapshot.state) : false

  // Cursor + accumulator for events.
  const [since, setSince] = useState(0)
  const [events, setEvents] = useState<JobEvent[]>([])
  const eventsQuery = useJobEvents(decodedRunId, undefined, since, !isTerminal)

  // Whenever the events poll returns, append to the local list and
  // bump the cursor. We dedupe on offset to be safe against retries.
  const lastIngestedRef = useRef<number>(-1)
  useEffect(() => {
    const data = eventsQuery.data
    if (!data) return
    if (data.events.length > 0) {
      const lastOffset = data.events[data.events.length - 1].offset
      if (lastOffset !== lastIngestedRef.current) {
        lastIngestedRef.current = lastOffset
        setEvents((prev) => mergeEvents(prev, data.events))
      }
    }
    if (data.next_since > since) {
      setSince(data.next_since)
    }
  }, [eventsQuery.data, since])

  const stopMut = useStopJob(decodedRunId, undefined)

  if (!decodedRunId) {
    return (
      <div className="p-5">
        <EmptyState title="No run selected" hint="URL is missing :runId." />
      </div>
    )
  }

  return (
    <div className="p-5 space-y-5 max-w-[1100px]">
      <section className="space-y-1">
        <h1 className="text-2xl font-semibold text-ink-1">Live training</h1>
        <p className="text-xs text-ink-3 font-mono">{decodedRunId}</p>
      </section>

      {/* Status row */}
      <Card>
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            {status.isLoading && !status.data && <Spinner />}
            {snapshot && <StatePill state={snapshot.state} />}
            {snapshot?.reason && (
              <span className="text-xs text-ink-2 truncate">{snapshot.reason}</span>
            )}
          </div>

          <button
            type="button"
            className="btn-danger-ghost"
            disabled={!snapshot || isTerminal || stopMut.isPending}
            onClick={() => {
              if (!window.confirm('Send graceful stop to the runner?')) return
              stopMut.mutate(undefined)
            }}
          >
            {stopMut.isPending ? 'Stopping…' : 'Stop'}
          </button>
        </div>

        {status.error && (
          <div className="mt-3 text-xs text-err bg-err/10 border border-err/30 px-3 py-2 rounded">
            {(status.error as Error).message}
          </div>
        )}
        {stopMut.error && (
          <div className="mt-3 text-xs text-err bg-err/10 border border-err/30 px-3 py-2 rounded">
            Stop failed: {(stopMut.error as Error).message}
          </div>
        )}
      </Card>

      {/* Submission metadata */}
      {submission && (
        <Card>
          <SectionHeader
            title="Submission"
            subtitle="Where the runner lives. Read-only."
          />
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs font-mono">
            <DLRow label="Job ID" value={submission.job_id} />
            <DLRow label="Provider" value={submission.provider_name} />
            <DLRow label="Pod ID" value={submission.pod_id ?? '—'} />
            <DLRow
              label="SSH"
              value={`${submission.ssh_username}@${submission.ssh_host}:${submission.ssh_port}`}
            />
            <DLRow label="Created" value={submission.created_at_iso} />
            {snapshot && <DLRow label="Sequence" value={String(snapshot.sequence)} />}
          </dl>
        </Card>
      )}

      {/* Events */}
      <Card>
        <SectionHeader
          title="Events"
          subtitle={
            isTerminal
              ? 'Run finished — events frozen.'
              : `Polling every 2 s · ${events.length} buffered`
          }
        />
        {events.length === 0 ? (
          <EmptyState
            title="No events yet"
            hint={
              status.isLoading
                ? 'Connecting to runner…'
                : 'The runner has not emitted any events yet.'
            }
          />
        ) : (
          <ol className="text-2xs font-mono space-y-0.5 max-h-[420px] overflow-auto">
            {events
              .slice()
              .reverse()
              .map((event) => (
                <EventLine key={event.offset} event={event} />
              ))}
          </ol>
        )}

        {eventsQuery.data?.error && (
          <div className="mt-3 text-2xs text-warn">
            Stream warning ({eventsQuery.data.error.code}):{' '}
            {eventsQuery.data.error.message}
          </div>
        )}
      </Card>
    </div>
  )
}

// --------------------------------------------------------------------
// Subcomponents
// --------------------------------------------------------------------

function DLRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-ink-3">{label}</dt>
      <dd className="text-ink-1 truncate" title={value}>
        {value}
      </dd>
    </>
  )
}

function EventLine({ event }: { event: JobEvent }) {
  const summary = useMemo(() => summarizePayload(event.payload), [event.payload])
  return (
    <li className="grid grid-cols-[3rem_8rem_1fr] gap-2 px-1 py-0.5 hover:bg-surface-3/40 rounded">
      <span className="text-ink-3 text-right">{event.offset}</span>
      <span className="text-info">{event.kind}</span>
      <span className="text-ink-2 truncate" title={summary}>
        {summary}
      </span>
    </li>
  )
}

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------

/**
 * Merge two event lists, deduping on ``offset``. Polls usually arrive
 * in strict order but a retry can replay the previous slice — guard
 * against duplicates so the list stays clean.
 */
function mergeEvents(prev: JobEvent[], incoming: JobEvent[]): JobEvent[] {
  if (incoming.length === 0) return prev
  const seen = new Set(prev.map((event) => event.offset))
  const additions = incoming.filter((event) => !seen.has(event.offset))
  if (additions.length === 0) return prev
  return [...prev, ...additions]
}

/** One-line preview of an event payload. Matches the CLI's ``_format_event_line``. */
function summarizePayload(payload: Record<string, unknown> | undefined): string {
  if (!payload) return ''
  const keys = Object.keys(payload)
  if (keys.length === 0) return ''
  // Priority fields the user likely wants to see first.
  const priority = ['loss', 'step', 'epoch', 'gpu_util_percent', 'reason']
  const ordered = [
    ...priority.filter((key) => keys.includes(key)),
    ...keys.filter((key) => !priority.includes(key)),
  ]
  return ordered
    .slice(0, 4)
    .map((key) => `${key}=${formatValue(payload[key])}`)
    .join('  ')
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(4)
  }
  if (typeof value === 'string') return value
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  try {
    const json = JSON.stringify(value)
    return json.length > 24 ? json.slice(0, 24) + '…' : json
  } catch {
    return '[…]'
  }
}
