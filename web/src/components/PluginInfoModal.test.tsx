/**
 * Tests for ``PluginInfoModal`` (PR16 / D1).
 *
 * The modal is read-only — it surfaces every meaningful manifest
 * field so plugin authors who land here from the catalog know what
 * they're getting before adding to a project. The tests below pin:
 *
 * - identity row (id / kind / version / category / stability) renders
 *   the manifest verbatim;
 * - description, params_schema, thresholds_schema each surface;
 * - reward plugins show the supported_strategies chip, others don't;
 * - close button + Escape key both fire the onClose callback.
 */

import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { PluginInfoModal } from './PluginInfoModal'
import type { PluginManifest } from '../api/types'

function manifest(overrides: Partial<PluginManifest> = {}): PluginManifest {
  return {
    schema_version: 4,
    id: 'demo_plugin',
    name: 'Demo Plugin',
    version: '1.2.3',
    description: 'Does demo things.',
    category: 'basic',
    stability: 'stable',
    kind: 'evaluation',
    supported_strategies: [],
    params_schema: {},
    thresholds_schema: {},
    suggested_params: {},
    suggested_thresholds: {},
    ...overrides,
  }
}

describe('PluginInfoModal', () => {
  it('renders the identity row from the manifest', () => {
    render(<PluginInfoModal plugin={manifest()} onClose={vi.fn()} />)
    expect(screen.getByText('Demo Plugin')).toBeInTheDocument()
    expect(screen.getByText('demo_plugin')).toBeInTheDocument()
    expect(screen.getByText('evaluation')).toBeInTheDocument()
    expect(screen.getByText('v1.2.3')).toBeInTheDocument()
    expect(screen.getByText('basic')).toBeInTheDocument()
    expect(screen.getByText('stable')).toBeInTheDocument()
  })

  it('falls back to id when name is empty', () => {
    render(
      <PluginInfoModal
        plugin={manifest({ name: '', id: 'fallback_id' })}
        onClose={vi.fn()}
      />,
    )
    // Title and id-code chip both reduce to "fallback_id".
    expect(screen.getAllByText('fallback_id').length).toBeGreaterThan(0)
  })

  it('renders the description block when present', () => {
    render(<PluginInfoModal plugin={manifest()} onClose={vi.fn()} />)
    expect(screen.getByText('Does demo things.')).toBeInTheDocument()
  })

  it('lists supported_strategies for reward plugins', () => {
    render(
      <PluginInfoModal
        plugin={manifest({
          kind: 'reward',
          supported_strategies: ['grpo', 'sapo'],
        })}
        onClose={vi.fn()}
      />,
    )
    expect(screen.getByText(/Compatible strategies/i)).toBeInTheDocument()
    expect(screen.getByText('grpo')).toBeInTheDocument()
    expect(screen.getByText('sapo')).toBeInTheDocument()
  })

  it('does not render the strategies block for non-reward plugins', () => {
    render(<PluginInfoModal plugin={manifest()} onClose={vi.fn()} />)
    expect(
      screen.queryByText(/Compatible strategies/i),
    ).not.toBeInTheDocument()
  })

  it('shows params_schema entries with type + description', () => {
    render(
      <PluginInfoModal
        plugin={manifest({
          params_schema: {
            type: 'object',
            properties: {
              timeout_seconds: {
                type: 'integer',
                description: 'How long to wait per call.',
                default: 30,
              },
            },
            required: [],
            additionalProperties: false,
          } as unknown as Record<string, unknown>,
        })}
        onClose={vi.fn()}
      />,
    )
    expect(screen.getByText('timeout_seconds')).toBeInTheDocument()
    expect(screen.getByText('How long to wait per call.')).toBeInTheDocument()
  })

  it('shows the empty-hint for plugins without thresholds', () => {
    render(<PluginInfoModal plugin={manifest()} onClose={vi.fn()} />)
    expect(
      screen.getByText(/This plugin accepts no parameters./i),
    ).toBeInTheDocument()
  })

  it('calls onClose when the close button is clicked', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(<PluginInfoModal plugin={manifest()} onClose={onClose} />)
    // The close button is the only ``button`` element in the modal —
    // the close-icon only renders inside it. ``getByText`` is fine
    // since "close" appears once at the top-right.
    await user.click(screen.getByText('close'))
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onClose on Escape', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(<PluginInfoModal plugin={manifest()} onClose={onClose} />)
    await user.keyboard('{Escape}')
    // Escape may fire onClose more than once when both the keydown
    // listener and the click-outside helper observe it; what matters
    // for UX is that *some* call happened.
    expect(onClose).toHaveBeenCalled()
  })
})
