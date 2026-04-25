/**
 * Tests for ``StalePluginsBanner`` (PR14 frontend).
 *
 * Behaviour the UX leans on:
 * - hides itself when there are no stale entries (no banner noise on
 *   healthy configs);
 * - one Remove button per row, calling back with (kind, instance_id);
 * - busy prop disables every Remove so a fast-double-click can't race
 *   into two saves;
 * - per-row data-testid is stable so the rest of the test suite (and
 *   future Cypress flows) can target a specific entry without
 *   fragile selectors.
 */

import { describe, expect, it, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { StalePluginsBanner } from './StalePluginsBanner'
import type { StalePluginEntry } from '../../api/types'

function entry(overrides: Partial<StalePluginEntry> = {}): StalePluginEntry {
  return {
    plugin_kind: 'evaluation',
    plugin_name: 'ghost_eval',
    instance_id: 'judge',
    location: 'evaluation.evaluators.plugins[judge]',
    ...overrides,
  }
}

describe('StalePluginsBanner', () => {
  it('renders nothing when the entries list is empty', () => {
    const { container } = render(
      <StalePluginsBanner entries={[]} onRemove={vi.fn()} />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('shows a singular headline for one stale entry', () => {
    render(<StalePluginsBanner entries={[entry()]} onRemove={vi.fn()} />)
    expect(
      screen.getByText(/1 stale plugin reference in this config/i),
    ).toBeInTheDocument()
  })

  it('shows a plural headline for multiple stale entries', () => {
    render(
      <StalePluginsBanner
        entries={[entry(), entry({ instance_id: 'judge_2' })]}
        onRemove={vi.fn()}
      />,
    )
    expect(
      screen.getByText(/2 stale plugin references in this config/i),
    ).toBeInTheDocument()
  })

  it('renders one Remove button per row + plugin metadata', () => {
    render(
      <StalePluginsBanner
        entries={[
          entry({ plugin_kind: 'evaluation', plugin_name: 'a' }),
          entry({
            plugin_kind: 'reports',
            plugin_name: 'b',
            instance_id: 'b',
            location: 'reports.sections[b]',
          }),
        ]}
        onRemove={vi.fn()}
      />,
    )
    const rows = screen.getAllByTestId('stale-plugin-row')
    expect(rows).toHaveLength(2)
    expect(within(rows[0]!).getByText('a')).toBeInTheDocument()
    expect(within(rows[1]!).getByText('b')).toBeInTheDocument()
    expect(
      screen.getAllByRole('button', { name: /remove from config/i }),
    ).toHaveLength(2)
  })

  it('calls onRemove with the right (kind, instance_id) on click', async () => {
    const user = userEvent.setup()
    const onRemove = vi.fn()
    render(
      <StalePluginsBanner
        entries={[
          entry({ plugin_kind: 'reward', plugin_name: 'r', instance_id: 'r' }),
        ]}
        onRemove={onRemove}
      />,
    )
    await user.click(screen.getByRole('button', { name: /remove from config/i }))
    expect(onRemove).toHaveBeenCalledExactlyOnceWith('reward', 'r')
  })

  it('disables every Remove button while busy=true', () => {
    render(
      <StalePluginsBanner
        entries={[entry(), entry({ instance_id: 'judge_2' })]}
        onRemove={vi.fn()}
        busy
      />,
    )
    const buttons = screen.getAllByRole('button', { name: /remove from config/i })
    expect(buttons).toHaveLength(2)
    buttons.forEach((btn) => expect(btn).toBeDisabled())
  })

  it('emits a per-row data-testid the rest of the suite can target', () => {
    render(
      <StalePluginsBanner
        entries={[
          entry({
            plugin_kind: 'validation',
            plugin_name: 'min_samples_old',
            instance_id: 'main',
          }),
        ]}
        onRemove={vi.fn()}
      />,
    )
    expect(
      screen.getByTestId('stale-plugin-remove-validation-main'),
    ).toBeInTheDocument()
  })
})
