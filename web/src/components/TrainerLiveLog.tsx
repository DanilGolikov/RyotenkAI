import { useEffect, useMemo, useRef, useState } from 'react'

import { useTrainerEventStream } from '../api/hooks/useTrainerEventStream'
import { Card, SectionHeader } from './ui'

type StreamFilter = 'all' | 'stdout' | 'stderr'

type Props = {
  runId: string
  attemptNo: number
  enabled?: boolean
}

/**
 * Live trainer-stdout/stderr panel for the active-run page.
 *
 * Wraps :func:`useTrainerEventStream` and filters down to
 * ``trainer_log`` events. Auto-scrolls to the tail of the buffer
 * unless the user has manually scrolled up (the
 * ``stickyToBottom`` heuristic).
 *
 * Visible controls:
 * - stream filter (all / stdout / stderr)
 * - pause / resume auto-scroll toggle
 * - manual reconnect button
 * - status pill (connected / connecting / closed)
 */
export function TrainerLiveLog({ runId, attemptNo, enabled = true }: Props) {
  const { events, connected, terminalReason, offset, phase, error, reconnect } =
    useTrainerEventStream(runId, attemptNo, enabled)

  const [filter, setFilter] = useState<StreamFilter>('all')
  const [autoScroll, setAutoScroll] = useState(true)
  const containerRef = useRef<HTMLOListElement>(null)

  const trainerLogs = useMemo(() => {
    const result: Array<{ offset: number; kind: string; line: string }> = []
    for (const ev of events) {
      if (ev.kind !== 'trainer_log') continue
      const payload = ev.payload || {}
      const line = typeof payload.line === 'string' ? payload.line : ''
      const streamKind =
        typeof payload.kind === 'string' ? payload.kind : 'stdout'
      if (filter !== 'all' && streamKind !== filter) continue
      if (!line) continue
      result.push({ offset: ev.offset, kind: streamKind, line })
    }
    return result
  }, [events, filter])

  // Auto-scroll to bottom when new lines arrive AND we're "sticky"
  // (auto-scroll on AND user is currently near the bottom).
  useEffect(() => {
    if (!autoScroll) return
    const el = containerRef.current
    if (!el) return
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight
    // If user scrolled up >100px, leave them alone.
    if (distance < 200) {
      el.scrollTop = el.scrollHeight
    }
  }, [trainerLogs, autoScroll])

  const statusPill = (() => {
    if (terminalReason) return { label: `done (${terminalReason})`, cls: 'bg-ok-bg text-ok' }
    if (connected) return { label: `live · ${phase ?? '…'}`, cls: 'bg-ok-bg text-ok' }
    if (error) return { label: 'error', cls: 'bg-err-bg text-err' }
    return { label: 'connecting…', cls: 'bg-warn-bg text-warn' }
  })()

  return (
    <Card>
      <SectionHeader
        title="Trainer log (live)"
        subtitle={
          <span className="font-mono">
            offset={offset} · events={events.length} · {trainerLogs.length} lines visible
          </span>
        }
        action={
          <div className="flex items-center gap-2">
            <span
              className={`pill text-2xs px-2 py-0.5 rounded ${statusPill.cls}`}
              data-testid="trainer-log-status"
            >
              {statusPill.label}
            </span>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as StreamFilter)}
              className="text-2xs border border-line-2 rounded px-1.5 py-0.5 bg-surface-3"
              aria-label="Filter trainer log streams"
            >
              <option value="all">all</option>
              <option value="stdout">stdout</option>
              <option value="stderr">stderr</option>
            </select>
            <button
              type="button"
              onClick={() => setAutoScroll((v) => !v)}
              className="text-2xs border border-line-2 rounded px-2 py-0.5 hover:bg-surface-3"
              aria-pressed={!autoScroll}
            >
              {autoScroll ? 'pause' : 'follow'}
            </button>
            <button
              type="button"
              onClick={reconnect}
              className="text-2xs border border-line-2 rounded px-2 py-0.5 hover:bg-surface-3"
              disabled={connected}
            >
              reconnect
            </button>
          </div>
        }
      />

      {trainerLogs.length === 0 ? (
        <div className="text-2xs text-ink-3 px-3 py-8 text-center">
          {connected || phase
            ? 'Waiting for trainer output…'
            : terminalReason
            ? 'Run ended without trainer output.'
            : 'Connecting to runner…'}
        </div>
      ) : (
        <ol
          ref={containerRef}
          className="text-2xs font-mono space-y-0.5 max-h-[480px] overflow-auto bg-surface-3 rounded p-2"
          data-testid="trainer-log-list"
        >
          {trainerLogs.map((entry) => (
            <li
              key={entry.offset}
              className={`whitespace-pre-wrap break-words ${
                entry.kind === 'stderr' ? 'text-warn' : 'text-ink-2'
              }`}
            >
              <span className="text-ink-3 mr-2">#{entry.offset}</span>
              {entry.line}
            </li>
          ))}
        </ol>
      )}
    </Card>
  )
}
