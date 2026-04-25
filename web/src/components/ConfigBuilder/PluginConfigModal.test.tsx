/**
 * Tests for ``PluginConfigModal`` (PR15 / D1).
 *
 * The Configure modal is the user's main surface for setting per-
 * instance params and thresholds, plus the kind-specific toggles
 * (``apply_to`` for validation, ``save_report`` for evaluation).
 * Tests below pin the contract callers depend on:
 *
 * - identity row reflects the manifest;
 * - dirty-tracking gates the Save button (no save until something
 *   changed);
 * - id-collision blocks save when the user renames into an existing id;
 * - reward broadcast hint shows when targets are non-empty;
 * - close button + onSave path call back correctly.
 *
 * The component pulls FieldRenderer / PluginEnvSection — both of
 * which work fine inside jsdom but require a non-empty schema to
 * render fields. We stick with empty schemas so the test focuses on
 * the modal's own logic, not FieldRenderer's.
 */

import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { PluginConfigModal } from './PluginConfigModal'
import type { PluginManifest } from '../../api/types'
import type { PluginInstanceDetails } from '../ProjectTabs/pluginInstances'

function manifest(
  overrides: Partial<PluginManifest> = {},
): PluginManifest {
  return {
    schema_version: 4,
    id: 'demo_plugin',
    name: 'Demo Plugin',
    version: '1.0.0',
    description: 'Does demo things.',
    category: '',
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

function details(
  overrides: Partial<PluginInstanceDetails> = {},
): PluginInstanceDetails {
  return {
    instanceId: 'judge',
    pluginId: 'demo_plugin',
    params: {},
    thresholds: {},
    ...overrides,
  }
}

describe('PluginConfigModal', () => {
  it('renders the manifest identity in the header', () => {
    render(
      <PluginConfigModal
        kind="evaluation"
        manifest={manifest()}
        initial={details()}
        takenInstanceIds={[]}
        onCancel={vi.fn()}
        onSave={vi.fn()}
      />,
    )
    expect(screen.getByText(/Configure Demo Plugin/i)).toBeInTheDocument()
    expect(screen.getByText('demo_plugin')).toBeInTheDocument()
    expect(screen.getByText('v1.0.0')).toBeInTheDocument()
    expect(screen.getByText('evaluation')).toBeInTheDocument()
  })

  it('disables Save until the user makes a change (no spurious dirty state)', () => {
    render(
      <PluginConfigModal
        kind="evaluation"
        manifest={manifest()}
        initial={details()}
        takenInstanceIds={[]}
        onCancel={vi.fn()}
        onSave={vi.fn()}
      />,
    )
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
  })

  it('enables Save once the instance id changes', async () => {
    const user = userEvent.setup()
    render(
      <PluginConfigModal
        kind="evaluation"
        manifest={manifest()}
        initial={details()}
        takenInstanceIds={[]}
        onCancel={vi.fn()}
        onSave={vi.fn()}
      />,
    )
    const idInput = screen.getByDisplayValue('judge')
    await user.clear(idInput)
    await user.type(idInput, 'judge_v2')
    expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled()
  })

  it('disables Save when the new instance id collides with another', async () => {
    const user = userEvent.setup()
    render(
      <PluginConfigModal
        kind="evaluation"
        manifest={manifest()}
        initial={details()}
        takenInstanceIds={['judge', 'other_instance']}
        onCancel={vi.fn()}
        onSave={vi.fn()}
      />,
    )
    const idInput = screen.getByDisplayValue('judge')
    await user.clear(idInput)
    await user.type(idInput, 'other_instance')
    expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled()
  })

  it('renders the reward broadcast hint when targets are non-empty', () => {
    render(
      <PluginConfigModal
        kind="reward"
        manifest={manifest({ kind: 'reward' })}
        initial={details({ pluginId: 'helixql_compiler_semantic' })}
        takenInstanceIds={[]}
        broadcastTargets={['grpo', 'sapo']}
        onCancel={vi.fn()}
        onSave={vi.fn()}
      />,
    )
    const hint = screen.getByTestId('reward-broadcast-hint')
    expect(hint).toBeInTheDocument()
    expect(hint).toHaveTextContent(/Applies to 2 strategies/i)
    expect(hint).toHaveTextContent(/grpo, sapo/)
  })

  it('hides the broadcast hint for non-reward kinds', () => {
    render(
      <PluginConfigModal
        kind="evaluation"
        manifest={manifest()}
        initial={details()}
        takenInstanceIds={[]}
        broadcastTargets={['grpo']}
        onCancel={vi.fn()}
        onSave={vi.fn()}
      />,
    )
    expect(
      screen.queryByTestId('reward-broadcast-hint'),
    ).not.toBeInTheDocument()
  })

  it('hides the broadcast hint when targets are empty', () => {
    render(
      <PluginConfigModal
        kind="reward"
        manifest={manifest({ kind: 'reward' })}
        initial={details()}
        takenInstanceIds={[]}
        broadcastTargets={[]}
        onCancel={vi.fn()}
        onSave={vi.fn()}
      />,
    )
    expect(
      screen.queryByTestId('reward-broadcast-hint'),
    ).not.toBeInTheDocument()
  })

  it('calls onSave with the edited details when the Save button fires', async () => {
    const user = userEvent.setup()
    const onSave = vi.fn().mockResolvedValue(undefined)
    const onCancel = vi.fn()
    render(
      <PluginConfigModal
        kind="evaluation"
        manifest={manifest()}
        initial={details()}
        takenInstanceIds={[]}
        onCancel={onCancel}
        onSave={onSave}
      />,
    )
    const idInput = screen.getByDisplayValue('judge')
    await user.clear(idInput)
    await user.type(idInput, 'judge_v2')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    expect(onSave).toHaveBeenCalledTimes(1)
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ instanceId: 'judge_v2' }),
    )
    // Modal closes itself after a successful save.
    expect(onCancel).toHaveBeenCalled()
  })

  it('renders the kind-specific instructions string', () => {
    render(
      <PluginConfigModal
        kind="reward"
        manifest={manifest({ kind: 'reward' })}
        initial={details()}
        takenInstanceIds={[]}
        onCancel={vi.fn()}
        onSave={vi.fn()}
      />,
    )
    // Reward instructions explicitly mention the propagation behaviour
    // — guards against accidentally swapping the strings between kinds.
    expect(
      screen.getByText(/apply to every matching training phase/i),
    ).toBeInTheDocument()
  })
})
