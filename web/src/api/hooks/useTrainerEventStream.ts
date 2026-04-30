import { useEffect, useRef, useState } from 'react'
import { wsUrl } from '../client'

/**
 * One event as published by the in-pod runner and relayed by the
 * Mac-side WS endpoint. Matches the pod-side ``JournalRecord`` /
 * ``EventResponse`` shape exactly.
 */
export type RunnerEvent = {
  v?: number
  offset: number
  ts?: string
  kind: string
  payload: Record<string, unknown>
}

export type TrainerEventStreamState = {
  events: RunnerEvent[]
  connected: boolean
  /** Set once the server sends an ``eof`` frame; further reconnect attempts stop. */
  terminalReason: string | null
  /** Highest offset seen so far. Reconnect uses ``offset + 1`` as ``since=``. */
  offset: number
  /** Most recent server-driven phase (``catchup`` or ``live``). */
  phase: 'catchup' | 'live' | null
  error: string | null
  reconnect: () => void
}

const MAX_BUFFER = 10_000

/**
 * Subscribe to the Mac-side WebSocket relay
 * (``/api/v1/runs/<id>/attempts/<n>/events/stream``) and accumulate
 * events into a bounded in-memory buffer.
 *
 * Reconnect strategy mirrors :func:`useLogStream`: exponential
 * backoff (0.5–8 s), driven by a `reconnectNonce` counter. On
 * reconnect we pass ``since=offset + 1`` so the Mac mirror reader
 * fills any gap before resuming the live tail.
 *
 * We stop reconnecting once the server sends ``eof`` — at that
 * point the run reached terminal state and there's no value in
 * keeping the WS open.
 */
export function useTrainerEventStream(
  runId: string | undefined,
  attemptNo: number | undefined,
  enabled: boolean,
): TrainerEventStreamState {
  const [events, setEvents] = useState<RunnerEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [terminalReason, setTerminalReason] = useState<string | null>(null)
  const [offset, setOffset] = useState(0)
  const [phase, setPhase] = useState<'catchup' | 'live' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const attemptRef = useRef(0)
  const offsetRef = useRef(0)
  const [reconnectNonce, setReconnectNonce] = useState(0)

  useEffect(() => {
    if (!enabled || !runId || !attemptNo) return
    if (terminalReason) return // run is done — no resubscribe

    setError(null)
    const since = offsetRef.current === 0 ? 0 : offsetRef.current + 1
    const path =
      `/runs/${encodeURIComponent(runId)}/attempts/${attemptNo}/events/stream` +
      `?since=${since}`
    const socket = new WebSocket(wsUrl(path))
    let closedByUs = false

    socket.onopen = () => {
      setConnected(true)
      attemptRef.current = 0 // reset backoff counter on successful open
    }

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as { type: string; [key: string]: unknown }
        if (msg.type === 'init') {
          if (msg.phase === 'catchup' || msg.phase === 'live') {
            setPhase(msg.phase)
          }
        } else if (msg.type === 'event') {
          const ev = msg.event as RunnerEvent | undefined
          if (!ev || typeof ev.offset !== 'number') return
          setEvents((prev) => {
            const appended = [...prev, ev]
            return appended.length > MAX_BUFFER
              ? appended.slice(appended.length - MAX_BUFFER)
              : appended
          })
          offsetRef.current = ev.offset
          setOffset(ev.offset)
        } else if (msg.type === 'eof') {
          const reason = typeof msg.reason === 'string' ? msg.reason : 'eof'
          setTerminalReason(reason)
          closedByUs = true
          socket.close()
        }
      } catch {
        /* ignore malformed frame */
      }
    }

    socket.onclose = () => {
      setConnected(false)
      if (closedByUs) return
      // Exponential backoff: 0.5, 1, 2, 4, 8, capped at 8 s. Same
      // schedule as useLogStream; consistent across the app.
      const attempt = attemptRef.current
      const delay = Math.min(8000, 500 * 2 ** attempt)
      attemptRef.current = attempt + 1
      setTimeout(() => setReconnectNonce((n) => n + 1), delay)
    }

    socket.onerror = () => setError('websocket error')

    return () => {
      closedByUs = true
      socket.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, attemptNo, enabled, reconnectNonce, terminalReason])

  return {
    events,
    connected,
    terminalReason,
    offset,
    phase,
    error,
    reconnect: () => {
      attemptRef.current = 0
      setReconnectNonce((n) => n + 1)
    },
  }
}
