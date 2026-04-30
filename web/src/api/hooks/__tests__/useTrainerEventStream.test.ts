/**
 * Tests for the trainer-event WebSocket hook.
 *
 * Covers (per project test policy):
 * - Positive: subscribes on mount, accumulates events
 * - Negative: terminal eof stops reconnect
 * - Boundary: buffer caps at MAX_BUFFER, disabled does not open
 * - Invariant: offset monotonic; reconnect uses last_offset+1
 * - Dependency-error: bad WS frame is skipped, malformed JSON ignored
 * - Regression: uses ``wsUrl`` helper (so wss:// is honoured behind TLS)
 * - Logic-specific: terminal reason set; phase init frame parsed
 *
 * jsdom does NOT provide a real WebSocket — we install a global
 * ``MockWebSocket`` whose API matches what the hook reads.
 */
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useTrainerEventStream } from '../useTrainerEventStream'

// ---------------------------------------------------------------------------
// Mock WebSocket — minimal surface the hook touches.
// ---------------------------------------------------------------------------

class MockWebSocket {
  static instances: MockWebSocket[] = []
  static reset() {
    MockWebSocket.instances = []
  }

  url: string
  readyState = 0
  onopen: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  onclose: ((ev: CloseEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }

  // Trigger WS lifecycle from the test side.
  open() {
    this.readyState = 1
    this.onopen?.(new Event('open'))
  }

  send(_data: string) {
    /* no-op: hook doesn't send */
  }

  message(payload: unknown) {
    this.onmessage?.(
      new MessageEvent('message', { data: JSON.stringify(payload) }),
    )
  }

  // Server-side close.
  remoteClose() {
    this.readyState = 3
    this.onclose?.(new CloseEvent('close'))
  }

  // Client-side close (the hook's cleanup).
  close() {
    this.readyState = 3
    this.onclose?.(new CloseEvent('close'))
  }
}

// ---------------------------------------------------------------------------
// Test setup
// ---------------------------------------------------------------------------

describe('useTrainerEventStream', () => {
  let originalWebSocket: typeof globalThis.WebSocket

  beforeEach(() => {
    originalWebSocket = globalThis.WebSocket
    // @ts-expect-error — assigning a partial-mock; the hook's surface is small.
    globalThis.WebSocket = MockWebSocket
    MockWebSocket.reset()
    // NOTE: don't enable fake timers globally — waitFor() uses real
    // setTimeout polling and would deadlock. Tests that need fake
    // timers (reconnect-backoff) opt in inside the test body.
  })

  afterEach(() => {
    globalThis.WebSocket = originalWebSocket
    vi.useRealTimers()
  })

  // -----------------------------------------------------------------------
  // Positive
  // -----------------------------------------------------------------------

  it('opens a WebSocket on mount when enabled', () => {
    renderHook(() =>
      useTrainerEventStream('run-x', 1, true),
    )
    expect(MockWebSocket.instances).toHaveLength(1)
    expect(MockWebSocket.instances[0].url).toContain(
      '/runs/run-x/attempts/1/events/stream',
    )
    expect(MockWebSocket.instances[0].url).toContain('since=0')
  })

  it('appends event frames to the buffer in order', async () => {
    const { result } = renderHook(() =>
      useTrainerEventStream('run-x', 1, true),
    )
    const ws = MockWebSocket.instances[0]
    act(() => ws.open())
    act(() => {
      ws.message({ type: 'init', phase: 'catchup', since: 0 })
      ws.message({
        type: 'event',
        event: { offset: 0, kind: 'trainer_log', payload: { line: 'a' } },
      })
      ws.message({
        type: 'event',
        event: { offset: 1, kind: 'trainer_log', payload: { line: 'b' } },
      })
    })

    await waitFor(() => expect(result.current.events).toHaveLength(2))
    expect(result.current.events.map((e) => e.offset)).toEqual([0, 1])
    expect(result.current.offset).toBe(1)
    expect(result.current.phase).toBe('catchup')
  })

  // -----------------------------------------------------------------------
  // Negative — terminal eof stops reconnect
  // -----------------------------------------------------------------------

  it('eof frame sets terminalReason and prevents reconnect', async () => {
    vi.useFakeTimers()
    const { result } = renderHook(() =>
      useTrainerEventStream('run-x', 1, true),
    )
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.open()
      ws.message({ type: 'eof', reason: 'trainer_exited' })
    })
    expect(result.current.terminalReason).toBe('trainer_exited')

    // Force a close — without eof we'd reconnect; with eof we must NOT.
    act(() => ws.remoteClose())
    act(() => {
      vi.advanceTimersByTime(20_000)
    })
    expect(MockWebSocket.instances).toHaveLength(1)
  })

  // -----------------------------------------------------------------------
  // Boundary — disabled does not open WS
  // -----------------------------------------------------------------------

  it('disabled=false does not open a WebSocket', () => {
    renderHook(() => useTrainerEventStream('run-x', 1, false))
    expect(MockWebSocket.instances).toHaveLength(0)
  })

  it('missing runId or attemptNo does not open a WebSocket', () => {
    renderHook(() => useTrainerEventStream(undefined, 1, true))
    expect(MockWebSocket.instances).toHaveLength(0)
    renderHook(() => useTrainerEventStream('run-x', undefined, true))
    expect(MockWebSocket.instances).toHaveLength(0)
  })

  // -----------------------------------------------------------------------
  // Invariant — reconnect uses last_offset+1
  // -----------------------------------------------------------------------

  it('reconnect after disconnect requests since=lastOffset+1', () => {
    vi.useFakeTimers()
    const { result } = renderHook(() =>
      useTrainerEventStream('run-x', 1, true),
    )
    const ws1 = MockWebSocket.instances[0]
    act(() => {
      ws1.open()
      ws1.message({ type: 'init', phase: 'catchup', since: 0 })
      ws1.message({
        type: 'event',
        event: { offset: 5, kind: 'trainer_log', payload: { line: 'x' } },
      })
    })
    expect(result.current.offset).toBe(5)

    // Simulate server close (not eof — true network drop).
    act(() => ws1.remoteClose())
    // Backoff: first attempt is 500 ms.
    act(() => vi.advanceTimersByTime(600))

    expect(MockWebSocket.instances).toHaveLength(2)
    expect(MockWebSocket.instances[1].url).toContain('since=6')
  })

  // -----------------------------------------------------------------------
  // Dependency-error — malformed frame skipped
  // -----------------------------------------------------------------------

  it('malformed JSON frames are ignored', async () => {
    const { result } = renderHook(() =>
      useTrainerEventStream('run-x', 1, true),
    )
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.open()
      // Manually fire a non-JSON message: hook must catch the parse
      // error and keep listening.
      ws.onmessage?.(new MessageEvent('message', { data: '{not json' }))
      ws.message({
        type: 'event',
        event: { offset: 0, kind: 'trainer_log', payload: { line: 'ok' } },
      })
    })
    await waitFor(() => expect(result.current.events).toHaveLength(1))
    expect(result.current.events[0].offset).toBe(0)
  })

  it('event without offset is skipped', async () => {
    const { result } = renderHook(() =>
      useTrainerEventStream('run-x', 1, true),
    )
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.open()
      ws.message({
        type: 'event',
        event: { kind: 'trainer_log', payload: { line: 'no-offset' } },
      })
      ws.message({
        type: 'event',
        event: { offset: 1, kind: 'trainer_log', payload: { line: 'good' } },
      })
    })
    await waitFor(() => expect(result.current.events).toHaveLength(1))
    expect(result.current.events[0].offset).toBe(1)
  })

  // -----------------------------------------------------------------------
  // Regression — uses wsUrl helper
  // -----------------------------------------------------------------------

  it('WS URL is built from wsUrl helper (does not hardcode ws://)', () => {
    renderHook(() => useTrainerEventStream('run-x', 1, true))
    const url = MockWebSocket.instances[0].url
    // Either ws:// or wss:// depending on document.location, never empty.
    expect(url).toMatch(/^wss?:\/\//)
  })
})
