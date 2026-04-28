/**
 * Tests for ``RunsTab`` (Step 7 of Variant 1).
 *
 * The tab is a thin presentational layer over ``useProjectRuns`` —
 * we mock the hook directly to pin every state (loading / error /
 * empty / populated). React Router is wrapped because each row
 * is a ``<Link>``.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { RunsTab } from './RunsTab'
import * as projectsHook from '../../api/hooks/useProjects'
import type { ProjectRunEntry, ProjectRunsResponse } from '../../api/types'

function row(overrides: Partial<ProjectRunEntry> = {}): ProjectRunEntry {
  return {
    run_id: 'run-1',
    started_at: '2026-04-28T10:00:00Z',
    status: 'completed',
    ...overrides,
  } as ProjectRunEntry
}

function mockRuns(state: {
  data?: ProjectRunsResponse
  isLoading?: boolean
  isError?: boolean
  error?: Error
}) {
  vi.spyOn(projectsHook, 'useProjectRuns').mockReturnValue({
    data: state.data,
    isLoading: state.isLoading ?? false,
    isError: state.isError ?? false,
    error: state.error ?? null,
    // Other react-query fields the component doesn't read.
  } as unknown as ReturnType<typeof projectsHook.useProjectRuns>)
}

function renderTab() {
  return render(
    <MemoryRouter>
      <RunsTab projectId="p1" />
    </MemoryRouter>,
  )
}

describe('RunsTab', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  // -------------------------------------------------------------------
  // Loading / error states
  // -------------------------------------------------------------------

  it('shows a loading hint while the query is in-flight', () => {
    mockRuns({ isLoading: true })
    renderTab()
    expect(screen.getByText(/loading runs/i)).toBeInTheDocument()
  })

  it('shows an error message when the query fails', () => {
    mockRuns({ isError: true, error: new Error('500 Internal Server Error') })
    renderTab()
    expect(screen.getByText(/failed to load runs/i)).toBeInTheDocument()
    expect(screen.getByText(/500 internal server error/i)).toBeInTheDocument()
  })

  // -------------------------------------------------------------------
  // Empty state
  // -------------------------------------------------------------------

  it('shows an empty-state message when there are no runs', () => {
    mockRuns({ data: { runs: [] } })
    renderTab()
    expect(screen.getByText(/no runs yet/i)).toBeInTheDocument()
  })

  // -------------------------------------------------------------------
  // Populated list
  // -------------------------------------------------------------------

  it('renders one row per run with run_id and started_at', () => {
    mockRuns({
      data: {
        runs: [
          row({ run_id: 'run-A', started_at: '2026-04-28T10:00:00Z' }),
          row({ run_id: 'run-B', started_at: '2026-04-27T10:00:00Z' }),
        ],
      },
    })
    renderTab()

    expect(screen.getByText('2 runs')).toBeInTheDocument()
    const links = screen.getAllByRole('link')
    expect(links).toHaveLength(2)
    expect(within(links[0]!).getByText('run-A')).toBeInTheDocument()
    expect(within(links[1]!).getByText('run-B')).toBeInTheDocument()
  })

  it('uses the singular form when there is exactly one run', () => {
    mockRuns({ data: { runs: [row()] } })
    renderTab()
    expect(screen.getByText('1 run')).toBeInTheDocument()
  })

  it('renders status badges for each row', () => {
    mockRuns({
      data: {
        runs: [
          row({ run_id: 'r-running', status: 'running' }),
          row({ run_id: 'r-failed', status: 'failed' }),
        ],
      },
    })
    renderTab()
    expect(screen.getByText('running')).toBeInTheDocument()
    expect(screen.getByText('failed')).toBeInTheDocument()
  })

  it('links each row to /runs/<run_id>', () => {
    mockRuns({
      data: {
        runs: [row({ run_id: 'my-run-with/slash' })],
      },
    })
    renderTab()
    const link = screen.getByRole('link') as HTMLAnchorElement
    // run_id is URL-encoded in the href so a slash doesn't break the route.
    expect(link.getAttribute('href')).toBe(
      '/runs/my-run-with%2Fslash',
    )
  })

  it('shows the actor when present, omits when missing', () => {
    mockRuns({
      data: {
        runs: [
          row({ run_id: 'with-actor', actor: 'alice' }),
          row({ run_id: 'no-actor' }),
        ],
      },
    })
    renderTab()
    expect(screen.getByText('alice')).toBeInTheDocument()
    // The no-actor row simply doesn't have a "by" prefix — verify by
    // counting "by " occurrences (one for the alice row only).
    expect(screen.getAllByText(/^by$/).length).toBe(1)
  })

  it('shows a 7-char prefix of config_version_hash when present', () => {
    mockRuns({
      data: {
        runs: [
          row({
            run_id: 'with-hash',
            config_version_hash: 'abcdef1234567890',
          }),
        ],
      },
    })
    renderTab()
    expect(screen.getByText('#abcdef1')).toBeInTheDocument()
  })

  it('renders raw started_at when it is not a parseable date', () => {
    mockRuns({
      data: {
        runs: [row({ run_id: 'malformed-time', started_at: 'not-a-date' })],
      },
    })
    renderTab()
    expect(screen.getByText('not-a-date')).toBeInTheDocument()
  })
})
