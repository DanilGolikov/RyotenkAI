/**
 * Tests for the ``KindSection`` block in ``PluginsTab`` (PR16 / D1).
 *
 * Scope: rendering + remove + configure callbacks + empty hint. The
 * DnD reorder behaviour itself is delegated to ``reorderInstances``
 * (covered by the pure-function suite in ``pluginInstances.test.ts``)
 * so this file doesn't simulate pointer drags — that path is brittle
 * under jsdom and the visible UX (a row gets removed / reordered) is
 * already covered by the underlying helper.
 *
 * KindSection requires a ``DndContext`` wrapper because it uses
 * ``useDroppable`` and ``useDndContext``. We provide a no-op context
 * so the hooks initialise cleanly.
 */

import { describe, expect, it, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DndContext } from '@dnd-kit/core'

import { KindSection } from './PluginsTab'
import type { PluginManifest } from '../../api/types'

function manifest(id: string, overrides: Partial<PluginManifest> = {}): PluginManifest {
  return {
    schema_version: 4,
    id,
    name: id,
    version: '1.0.0',
    description: '',
    category: '',
    stability: 'stable',
    kind: 'validation',
    supported_strategies: [],
    params_schema: {},
    thresholds_schema: {},
    suggested_params: {},
    suggested_thresholds: {},
    ...overrides,
  }
}

function renderInDnd(ui: React.ReactNode) {
  return render(<DndContext>{ui}</DndContext>)
}

describe('KindSection', () => {
  it('renders the label, help text, and instance count', () => {
    renderInDnd(
      <KindSection
        kind="validation"
        label="Validation"
        help="Pre-flight quality checks."
        sortable
        instances={[]}
        manifestById={new Map()}
        activeStrategyTypes={new Set()}
        onRemove={vi.fn()}
        onConfigure={vi.fn()}
        onInfo={vi.fn()}
      />,
    )
    expect(screen.getByText('Validation')).toBeInTheDocument()
    expect(screen.getByText(/Pre-flight quality checks/i)).toBeInTheDocument()
    // Empty list → instance count chip reads "0".
    expect(screen.getByText('0')).toBeInTheDocument()
  })

  it('renders one PluginInstanceRow per instance with the manifest name', () => {
    const byId = new Map<string, PluginManifest>([
      ['min_samples', manifest('min_samples', { name: 'Minimum Samples' })],
      ['avg_length', manifest('avg_length', { name: 'Average Length' })],
    ])
    renderInDnd(
      <KindSection
        kind="validation"
        label="Validation"
        help=""
        sortable
        instances={[
          { instanceId: 'min', pluginId: 'min_samples' },
          { instanceId: 'avg', pluginId: 'avg_length' },
        ]}
        manifestById={byId}
        activeStrategyTypes={new Set()}
        onRemove={vi.fn()}
        onConfigure={vi.fn()}
        onInfo={vi.fn()}
      />,
    )
    expect(screen.getByText('Minimum Samples')).toBeInTheDocument()
    expect(screen.getByText('Average Length')).toBeInTheDocument()
  })

  it('calls onRemove with the instance id when a Remove button is clicked', async () => {
    const user = userEvent.setup()
    const onRemove = vi.fn()
    const byId = new Map<string, PluginManifest>([
      ['min_samples', manifest('min_samples', { name: 'Minimum Samples' })],
    ])
    renderInDnd(
      <KindSection
        kind="validation"
        label="Validation"
        help=""
        sortable
        instances={[{ instanceId: 'min', pluginId: 'min_samples' }]}
        manifestById={byId}
        activeStrategyTypes={new Set()}
        onRemove={onRemove}
        onConfigure={vi.fn()}
        onInfo={vi.fn()}
      />,
    )
    // Find the row, then its remove control. PluginInstanceRow
    // typically renders the action buttons via aria-label / title;
    // find them by their visible text or by the X glyph.
    const removeBtn = screen.getByRole('button', { name: /remove/i })
    await user.click(removeBtn)
    expect(onRemove).toHaveBeenCalledWith('min')
  })

  it('calls onConfigure with the instance when its Configure handle fires', async () => {
    const user = userEvent.setup()
    const onConfigure = vi.fn()
    const byId = new Map<string, PluginManifest>([
      ['min_samples', manifest('min_samples', { name: 'Minimum Samples' })],
    ])
    renderInDnd(
      <KindSection
        kind="validation"
        label="Validation"
        help=""
        sortable
        instances={[{ instanceId: 'min', pluginId: 'min_samples' }]}
        manifestById={byId}
        activeStrategyTypes={new Set()}
        onRemove={vi.fn()}
        onConfigure={onConfigure}
        onInfo={vi.fn()}
      />,
    )
    await user.click(screen.getByRole('button', { name: /configure/i }))
    expect(onConfigure).toHaveBeenCalledWith(
      expect.objectContaining({ instanceId: 'min', pluginId: 'min_samples' }),
    )
  })

  it('renders the "Reset to defaults" action when onResetToDefaults is wired (reports kind)', async () => {
    const user = userEvent.setup()
    const onReset = vi.fn()
    renderInDnd(
      <KindSection
        kind="reports"
        label="Reports"
        help="Per-section ordering."
        sortable
        instances={[]}
        manifestById={new Map()}
        activeStrategyTypes={new Set()}
        onRemove={vi.fn()}
        onConfigure={vi.fn()}
        onInfo={vi.fn()}
        onResetToDefaults={onReset}
      />,
    )
    const reset = screen.getByRole('button', { name: /reset to defaults/i })
    await user.click(reset)
    expect(onReset).toHaveBeenCalledTimes(1)
  })

  it('hides "Reset to defaults" for kinds that have no defaults', () => {
    renderInDnd(
      <KindSection
        kind="validation"
        label="Validation"
        help=""
        sortable
        instances={[]}
        manifestById={new Map()}
        activeStrategyTypes={new Set()}
        onRemove={vi.fn()}
        onConfigure={vi.fn()}
        onInfo={vi.fn()}
      />,
    )
    expect(
      screen.queryByRole('button', { name: /reset to defaults/i }),
    ).not.toBeInTheDocument()
  })

  it('falls back to the plugin id when the manifest has gone missing from the catalog', () => {
    // No manifest in the map for ``ghost_plugin`` — this is the
    // "stale reference still in YAML, plugin folder removed" path.
    // The row must still render so the user can find + remove it.
    renderInDnd(
      <KindSection
        kind="validation"
        label="Validation"
        help=""
        sortable
        instances={[{ instanceId: 'main', pluginId: 'ghost_plugin' }]}
        manifestById={new Map()}
        activeStrategyTypes={new Set()}
        onRemove={vi.fn()}
        onConfigure={vi.fn()}
        onInfo={vi.fn()}
      />,
    )
    // Some surface text references the plugin id so the row is
    // identifiable. We don't pin which slot — just that "ghost_plugin"
    // is in the DOM.
    expect(within(document.body).getByText(/ghost_plugin/)).toBeInTheDocument()
  })
})
