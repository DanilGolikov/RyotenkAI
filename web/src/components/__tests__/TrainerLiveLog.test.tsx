/**
 * Tests for TrainerLiveLog component.
 *
 * The component is a thin shell around ``useTrainerEventStream`` —
 * tests mock the hook to avoid spinning up a real (or even a mock)
 * WebSocket. This keeps the component-level concerns isolated:
 *
 * - Positive: renders incoming trainer_log lines
 * - Negative: filters by stream kind
 * - Boundary: empty state when no lines
 * - Logic-specific: pause button toggles auto-scroll
 * - Regression: status pill reflects connection state
 */
import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import * as hookModule from '../../api/hooks/useTrainerEventStream'
import { TrainerLiveLog } from '../TrainerLiveLog'

type HookReturn = ReturnType<typeof hookModule.useTrainerEventStream>

function makeHookReturn(overrides: Partial<HookReturn> = {}): HookReturn {
  return {
    events: [],
    connected: true,
    terminalReason: null,
    offset: 0,
    phase: 'live',
    error: null,
    reconnect: vi.fn(),
    ...overrides,
  }
}

describe('TrainerLiveLog', () => {
  let useTrainerEventStreamSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    useTrainerEventStreamSpy = vi.spyOn(hookModule, 'useTrainerEventStream')
  })

  afterEach(() => {
    useTrainerEventStreamSpy.mockRestore()
  })

  // -----------------------------------------------------------------------
  // Positive
  // -----------------------------------------------------------------------

  it('renders trainer_log lines', () => {
    useTrainerEventStreamSpy.mockReturnValue(
      makeHookReturn({
        events: [
          { offset: 0, kind: 'trainer_log',
            payload: { kind: 'stdout', line: 'epoch 1 loss=0.5' } },
          { offset: 1, kind: 'trainer_log',
            payload: { kind: 'stdout', line: 'epoch 2 loss=0.3' } },
        ],
      }),
    )

    render(<TrainerLiveLog runId="r" attemptNo={1} />)

    expect(screen.getByText('epoch 1 loss=0.5')).toBeInTheDocument()
    expect(screen.getByText('epoch 2 loss=0.3')).toBeInTheDocument()
  })

  it('non-trainer-log events are filtered out of the visible list', () => {
    useTrainerEventStreamSpy.mockReturnValue(
      makeHookReturn({
        events: [
          { offset: 0, kind: 'health_snapshot',
            payload: { gpu_util_percent: 80 } },
          { offset: 1, kind: 'trainer_log',
            payload: { kind: 'stdout', line: 'visible' } },
        ],
      }),
    )

    render(<TrainerLiveLog runId="r" attemptNo={1} />)
    expect(screen.getByText('visible')).toBeInTheDocument()
    // Health snapshot payload must not show up as a line.
    expect(screen.queryByText(/gpu_util_percent/)).toBeNull()
  })

  // -----------------------------------------------------------------------
  // Boundary — empty state
  // -----------------------------------------------------------------------

  it('shows the waiting placeholder when no trainer_log events have arrived', () => {
    useTrainerEventStreamSpy.mockReturnValue(makeHookReturn({ events: [] }))
    render(<TrainerLiveLog runId="r" attemptNo={1} />)
    expect(screen.getByText(/Waiting for trainer output/)).toBeInTheDocument()
  })

  it('shows the run-ended placeholder when terminalReason set and no events', () => {
    useTrainerEventStreamSpy.mockReturnValue(
      makeHookReturn({
        connected: false,
        terminalReason: 'trainer_exited',
        phase: null,
      }),
    )
    render(<TrainerLiveLog runId="r" attemptNo={1} />)
    expect(screen.getByText(/Run ended without trainer output/)).toBeInTheDocument()
  })

  // -----------------------------------------------------------------------
  // Negative — stream filter
  // -----------------------------------------------------------------------

  it('selecting stderr filter hides stdout lines', () => {
    useTrainerEventStreamSpy.mockReturnValue(
      makeHookReturn({
        events: [
          { offset: 0, kind: 'trainer_log',
            payload: { kind: 'stdout', line: 'out-line' } },
          { offset: 1, kind: 'trainer_log',
            payload: { kind: 'stderr', line: 'err-line' } },
        ],
      }),
    )
    render(<TrainerLiveLog runId="r" attemptNo={1} />)
    fireEvent.change(
      screen.getByLabelText(/Filter trainer log streams/),
      { target: { value: 'stderr' } },
    )
    expect(screen.queryByText('out-line')).toBeNull()
    expect(screen.getByText('err-line')).toBeInTheDocument()
  })

  // -----------------------------------------------------------------------
  // Logic-specific — pause button
  // -----------------------------------------------------------------------

  it('pause button toggles to "follow" label', () => {
    useTrainerEventStreamSpy.mockReturnValue(makeHookReturn({ events: [] }))
    render(<TrainerLiveLog runId="r" attemptNo={1} />)
    const button = screen.getByRole('button', { name: /pause/i })
    fireEvent.click(button)
    expect(screen.getByRole('button', { name: /follow/i })).toBeInTheDocument()
  })

  // -----------------------------------------------------------------------
  // Regression — status pill
  // -----------------------------------------------------------------------

  it('status pill shows live phase when connected', () => {
    useTrainerEventStreamSpy.mockReturnValue(
      makeHookReturn({ connected: true, phase: 'live' }),
    )
    render(<TrainerLiveLog runId="r" attemptNo={1} />)
    expect(screen.getByTestId('trainer-log-status').textContent).toMatch(
      /live/,
    )
  })

  it('status pill shows done state on terminalReason', () => {
    useTrainerEventStreamSpy.mockReturnValue(
      makeHookReturn({
        connected: false,
        terminalReason: 'trainer_exited',
        phase: null,
      }),
    )
    render(<TrainerLiveLog runId="r" attemptNo={1} />)
    expect(screen.getByTestId('trainer-log-status').textContent).toMatch(
      /done/,
    )
  })

  it('reconnect button calls hook reconnect', () => {
    const reconnect = vi.fn()
    useTrainerEventStreamSpy.mockReturnValue(
      makeHookReturn({ connected: false, reconnect }),
    )
    render(<TrainerLiveLog runId="r" attemptNo={1} />)
    fireEvent.click(screen.getByRole('button', { name: /reconnect/i }))
    expect(reconnect).toHaveBeenCalled()
  })
})
