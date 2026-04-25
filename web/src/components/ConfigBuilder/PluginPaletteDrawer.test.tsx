/**
 * Tests for ``PluginPaletteDrawer`` (PR17 / D1).
 *
 * The drawer's job is "let me find a plugin to drag onto the project."
 * Tests below pin behaviour the user relies on:
 *
 * - search filters by id / name / description, case-insensitive;
 * - reports plugins already attached are hidden (single-instance);
 * - reward plugins whose supported_strategies don't match the current
 *   project's strategies render disabled with an "incompatible" tooltip;
 * - clicking an info button fires the callback (the catalog browse
 *   path);
 * - kind groups expand on first match when the user types a query.
 *
 * The drawer queries useAllPlugins via react-query — we mock the hook
 * directly since the network surface itself is covered by integration
 * tests.
 */

import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { PluginPaletteDrawer } from './PluginPaletteDrawer'
import type { PluginKind, PluginManifest } from '../../api/types'

function manifest(
  id: string,
  kind: PluginKind,
  overrides: Partial<PluginManifest> = {},
): PluginManifest {
  return {
    schema_version: 4,
    id,
    name: id,
    version: '1.0.0',
    description: '',
    category: '',
    stability: 'stable',
    kind,
    supported_strategies: [],
    params_schema: {},
    thresholds_schema: {},
    suggested_params: {},
    suggested_thresholds: {},
    ...overrides,
  }
}

// useAllPlugins is the only external surface the drawer touches.
// Per-test override below supplies the byKind / loading / error
// contract; the dnd-kit useDraggable returns a stable mock so chip
// rendering doesn't require wrapping in DndContext.
const mockHookState = {
  byKind: {
    validation: [] as PluginManifest[],
    evaluation: [] as PluginManifest[],
    reward: [] as PluginManifest[],
    reports: [] as PluginManifest[],
  } as Record<PluginKind, PluginManifest[]>,
  isLoading: false,
  error: undefined as Error | undefined,
}

vi.mock('../../api/hooks/usePlugins', () => ({
  useAllPlugins: () => mockHookState,
}))

// Stub useDraggable to avoid pulling DndContext into every test —
// the palette's filtering / search / disabling logic doesn't touch
// the dnd lifecycle, only the chip rendering does (which we don't
// assert beyond presence here).
vi.mock('@dnd-kit/core', async (importActual) => {
  const actual = await importActual<typeof import('@dnd-kit/core')>()
  return {
    ...actual,
    useDraggable: () => ({
      attributes: {},
      listeners: {},
      setNodeRef: () => {},
      isDragging: false,
    }),
  }
})

function setMock(byKind: Partial<Record<PluginKind, PluginManifest[]>>): void {
  mockHookState.byKind = {
    validation: byKind.validation ?? [],
    evaluation: byKind.evaluation ?? [],
    reward: byKind.reward ?? [],
    reports: byKind.reports ?? [],
  }
}

function emptyAttached(): Record<PluginKind, Set<string>> {
  return {
    validation: new Set(),
    evaluation: new Set(),
    reward: new Set(),
    reports: new Set(),
  }
}

describe('PluginPaletteDrawer', () => {
  it('renders all four kind groups by default', () => {
    setMock({})
    render(
      <PluginPaletteDrawer
        attachedIdsByKind={emptyAttached()}
        activeStrategyTypes={new Set()}
      />,
    )
    // Group buttons stay collapsed by default but the labels are
    // always visible — assert each one is rendered.
    expect(screen.getByText('Validation')).toBeInTheDocument()
    expect(screen.getByText('Evaluation')).toBeInTheDocument()
    expect(screen.getByText('Reward')).toBeInTheDocument()
    expect(screen.getByText('Reports')).toBeInTheDocument()
  })

  it('respects onlyKinds — Datasets-tab usage hides the other groups', () => {
    setMock({})
    render(
      <PluginPaletteDrawer
        attachedIdsByKind={emptyAttached()}
        activeStrategyTypes={new Set()}
        onlyKinds={['validation']}
      />,
    )
    expect(screen.getByText('Validation')).toBeInTheDocument()
    expect(screen.queryByText('Evaluation')).not.toBeInTheDocument()
    expect(screen.queryByText('Reward')).not.toBeInTheDocument()
    expect(screen.queryByText('Reports')).not.toBeInTheDocument()
  })

  it('search filter expands matching groups and lists only matches', async () => {
    const user = userEvent.setup()
    setMock({
      validation: [
        manifest('min_samples', 'validation'),
        manifest('avg_length', 'validation'),
      ],
      evaluation: [manifest('cerebras_judge', 'evaluation')],
    })
    render(
      <PluginPaletteDrawer
        attachedIdsByKind={emptyAttached()}
        activeStrategyTypes={new Set()}
      />,
    )

    await user.type(screen.getByLabelText('Search palette'), 'min')

    // Validation auto-expanded → min_samples chip visible. avg_length
    // was filtered out. cerebras_judge (different kind) doesn't match.
    expect(screen.getByText('min_samples')).toBeInTheDocument()
    expect(screen.queryByText('avg_length')).not.toBeInTheDocument()
    expect(screen.queryByText('cerebras_judge')).not.toBeInTheDocument()
  })

  it('search is case-insensitive across id, name, and description', async () => {
    const user = userEvent.setup()
    setMock({
      validation: [
        manifest('foo', 'validation', {
          name: 'Foo',
          description: 'matches by description',
        }),
        manifest('Bar', 'validation', { name: 'BAR_PLUGIN' }),
      ],
    })
    render(
      <PluginPaletteDrawer
        attachedIdsByKind={emptyAttached()}
        activeStrategyTypes={new Set()}
      />,
    )

    await user.type(
      screen.getByLabelText('Search palette'),
      'DESCRIPTION',
    )
    expect(screen.getByText('foo')).toBeInTheDocument()
    expect(screen.queryByText('Bar')).not.toBeInTheDocument()
  })

  it('hides reports plugins already attached (single-instance kind)', async () => {
    const user = userEvent.setup()
    setMock({
      reports: [
        manifest('header', 'reports'),
        manifest('footer', 'reports'),
      ],
    })
    const attached = emptyAttached()
    attached.reports = new Set(['header'])
    render(
      <PluginPaletteDrawer
        attachedIdsByKind={attached}
        activeStrategyTypes={new Set()}
      />,
    )

    // Type query to expand the reports group; "header" is dropped from
    // the palette (already attached); "footer" remains.
    await user.type(screen.getByLabelText('Search palette'), 'er')
    expect(screen.queryByText('header')).not.toBeInTheDocument()
    expect(screen.getByText('footer')).toBeInTheDocument()
  })

  it('manual toggle opens / closes a kind group', async () => {
    const user = userEvent.setup()
    setMock({
      validation: [manifest('min_samples', 'validation')],
    })
    render(
      <PluginPaletteDrawer
        attachedIdsByKind={emptyAttached()}
        activeStrategyTypes={new Set()}
      />,
    )

    // Validation starts collapsed, so the chip is not in the DOM.
    expect(screen.queryByText('min_samples')).not.toBeInTheDocument()

    // Click the validation row to expand. ``aria-expanded`` swaps to
    // ``true``; the chip becomes visible.
    const button = screen.getByRole('button', { name: /validation/i })
    await user.click(button)
    expect(button).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText('min_samples')).toBeInTheDocument()
  })

  it('dims reward plugins whose supported_strategies do not overlap', async () => {
    const user = userEvent.setup()
    setMock({
      reward: [
        manifest('helixql_compiler_semantic', 'reward', {
          supported_strategies: ['grpo', 'sapo'],
        }),
      ],
    })
    render(
      <PluginPaletteDrawer
        attachedIdsByKind={emptyAttached()}
        activeStrategyTypes={new Set(['sft'])}
      />,
    )

    await user.type(screen.getByLabelText('Search palette'), 'helix')
    const chip = screen.getByLabelText(
      /Plugin helixql_compiler_semantic.*unavailable/,
    )
    expect(chip).toBeInTheDocument()
  })

  it('calls onInfoClick when the info button is clicked', async () => {
    const user = userEvent.setup()
    const onInfoClick = vi.fn()
    setMock({
      validation: [manifest('min_samples', 'validation')],
    })
    render(
      <PluginPaletteDrawer
        attachedIdsByKind={emptyAttached()}
        activeStrategyTypes={new Set()}
        onInfoClick={onInfoClick}
      />,
    )

    await user.type(screen.getByLabelText('Search palette'), 'min')
    await user.click(screen.getByLabelText(/Details for min_samples/i))
    expect(onInfoClick).toHaveBeenCalledTimes(1)
    expect(onInfoClick).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'min_samples' }),
    )
  })

  it('surfaces a loading state while the catalog query is pending', () => {
    mockHookState.isLoading = true
    setMock({})
    try {
      render(
        <PluginPaletteDrawer
          attachedIdsByKind={emptyAttached()}
          activeStrategyTypes={new Set()}
        />,
      )
      expect(screen.getByText(/Loading…/)).toBeInTheDocument()
    } finally {
      mockHookState.isLoading = false
    }
  })
})
