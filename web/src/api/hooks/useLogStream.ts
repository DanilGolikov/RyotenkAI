import { useEffect, useRef, useState } from 'react'
import { wsUrl } from '../client'

export type LogStreamState = {
  lines: string[]
  connected: boolean
  terminalStatus: string | null
  offset: number
  error: string | null
  reconnect: () => void
}

const MAX_BUFFER = 10_000

export function useLogStream(
  runId: string | undefined,
  attemptNo: number | undefined,
  file: string,
  enabled: boolean,
): LogStreamState {
  const [lines, setLines] = useState<string[]>([])
  const [connected, setConnected] = useState(false)
  const [terminalStatus, setTerminalStatus] = useState<string | null>(null)
  const [offset, setOffset] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const attemptRef = useRef(0)
  const [reconnectNonce, setReconnectNonce] = useState(0)

  useEffect(() => {
    if (!enabled || !runId || !attemptNo) return
    setError(null)
    setTerminalStatus(null)

    const path = `/runs/${encodeURIComponent(runId)}/attempts/${attemptNo}/logs/stream?file=${encodeURIComponent(file)}`
    const socket = new WebSocket(wsUrl(path))
    let closedByUs = false

    socket.onopen = () => setConnected(true)
    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as { type: string; [key: string]: unknown }
        if (msg.type === 'init' && typeof msg.offset === 'number') {
          setOffset(msg.offset)
        } else if (msg.type === 'chunk' && Array.isArray(msg.lines)) {
          setLines((prev) => {
            const appended = [...prev, ...(msg.lines as string[])]
            return appended.length > MAX_BUFFER ? appended.slice(appended.length - MAX_BUFFER) : appended
          })
          if (typeof msg.offset === 'number') setOffset(msg.offset)
        } else if (msg.type === 'state' && typeof msg.status === 'string') {
          setTerminalStatus(msg.status as string)
        } else if (msg.type === 'eof') {
          closedByUs = true
          socket.close()
        }
      } catch {
        /* ignore malformed */
      }
    }
    socket.onclose = () => {
      setConnected(false)
      if (closedByUs) return
      // Exponential backoff: 0.5, 1, 2, 4, 8, capped at 8s.
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
  }, [runId, attemptNo, file, enabled, reconnectNonce])

  return {
    lines,
    connected,
    terminalStatus,
    offset,
    error,
    reconnect: () => {
      attemptRef.current = 0
      setReconnectNonce((n) => n + 1)
    },
  }
}
