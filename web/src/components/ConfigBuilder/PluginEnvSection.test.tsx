/**
 * Tests for ``PluginEnvSection`` (PR15 / D1).
 *
 * Three behaviours the UI relies on:
 *
 * - secret rows render as ``type="password"`` with a Show/Hide toggle;
 * - rows whose ``managed_by`` is set render as read-only with a deep
 *   link into Settings (no plain-text override of integration/provider
 *   credentials);
 * - empty ``required`` list short-circuits to ``null`` (no headline,
 *   no spurious noise in the modal).
 */

import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { PluginEnvSection } from './PluginEnvSection'
import type { PluginRequiredEnv } from '../../api/types'

// useProjectEnv / useSaveProjectEnv hit the env.json API. Stub them
// with a minimal in-memory mock — we don't care about the network
// surface here (it's covered by integration tests), only how the
// section renders given the data.
const mockEnvData = {
  data: { env: {} as Record<string, string> },
  isLoading: false,
  refetch: () => Promise.resolve(),
}
const mockSaveMut = {
  mutateAsync: vi.fn(),
  isPending: false,
  isSuccess: false,
  error: undefined,
}
vi.mock('../../api/hooks/useProjects', () => ({
  useProjectEnv: () => mockEnvData,
  useSaveProjectEnv: () => mockSaveMut,
}))

function spec(overrides: Partial<PluginRequiredEnv> = {}): PluginRequiredEnv {
  return {
    name: 'EVAL_API_KEY',
    description: 'API key for the upstream judge',
    optional: false,
    secret: true,
    managed_by: '',
    ...overrides,
  }
}

describe('PluginEnvSection', () => {
  it('renders nothing when required is empty', () => {
    const { container } = render(
      <PluginEnvSection projectId="p1" required={[]} />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('renders the headline + per-row name for non-empty required', () => {
    render(
      <PluginEnvSection projectId="p1" required={[spec()]} />,
    )
    expect(
      screen.getByText('Required environment variables'),
    ).toBeInTheDocument()
    expect(screen.getByText('EVAL_API_KEY')).toBeInTheDocument()
  })

  it('uses type=password for secret rows so the value never lands in the DOM as plaintext', () => {
    render(
      <PluginEnvSection projectId="p1" required={[spec()]} />,
    )
    const input = screen.getByPlaceholderText('paste secret')
    expect(input).toHaveAttribute('type', 'password')
  })

  it('Show/Hide toggle swaps the input type', async () => {
    const user = userEvent.setup()
    render(
      <PluginEnvSection projectId="p1" required={[spec()]} />,
    )

    const input = screen.getByPlaceholderText('paste secret')
    expect(input).toHaveAttribute('type', 'password')
    await user.click(screen.getByRole('button', { name: 'Show' }))
    expect(input).toHaveAttribute('type', 'text')
    await user.click(screen.getByRole('button', { name: 'Hide' }))
    expect(input).toHaveAttribute('type', 'password')
  })

  it('renders non-secret rows as plain text inputs', () => {
    render(
      <PluginEnvSection
        projectId="p1"
        required={[spec({ name: 'HELIX_CLI_PATH', secret: false })]}
      />,
    )
    const input = screen.getByPlaceholderText('value')
    expect(input).toHaveAttribute('type', 'text')
    // No Show/Hide toggle for non-secret entries — secret-only widget.
    expect(
      screen.queryByRole('button', { name: /^(Show|Hide)$/ }),
    ).not.toBeInTheDocument()
  })

  it('renders managed-by entries as read-only with a Settings deep link', () => {
    render(
      <PluginEnvSection
        projectId="p1"
        required={[spec({ name: 'EVAL_HF_TOKEN', managed_by: 'integrations' })]}
      />,
    )
    // No editable input for managed entries.
    expect(screen.queryByPlaceholderText('paste secret')).not.toBeInTheDocument()
    expect(
      screen.getByRole('link', { name: /Settings → Integrations/i }),
    ).toBeInTheDocument()
  })

  it('renders the description tooltip text inline', () => {
    render(
      <PluginEnvSection projectId="p1" required={[spec()]} />,
    )
    expect(
      screen.getByText(/API key for the upstream judge/),
    ).toBeInTheDocument()
  })

  it('marks required (non-optional) entries with a star indicator', () => {
    render(
      <PluginEnvSection
        projectId="p1"
        required={[spec({ optional: false })]}
      />,
    )
    expect(screen.getByText('*')).toBeInTheDocument()
  })
})
