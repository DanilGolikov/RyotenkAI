import { useNavigate, useParams } from 'react-router-dom'
import { useProviders } from '../../api/hooks/useProviders'
import { ProviderStatusChip } from './ProviderStatusChip'
import { SelectField } from './SelectField'

interface Props {
  value: unknown
  onChange: (next: unknown) => void
  onFocus?: () => void
  onBlur?: () => void
}

/**
 * Dropdown for ``training.provider``. Uses the backend-computed
 * ``has_training`` flag on every provider summary (PR1) so we don't
 * fetch per-provider configs in a N+1 loop.
 */
export function TrainingProviderField({ value, onChange, onFocus, onBlur }: Props) {
  const providersQuery = useProviders()
  const providers = providersQuery.data ?? []
  const navigate = useNavigate()
  const { id: projectId } = useParams<{ id: string }>()
  const openSettings = () => {
    if (projectId) navigate(`/projects/${encodeURIComponent(projectId)}/settings`)
  }

  const eligible = providers.filter((p) => p.has_training)
  const current = typeof value === 'string' ? value : ''

  if (providersQuery.isLoading && eligible.length === 0) {
    return (
      <div className="h-8 inline-flex items-center px-2.5 rounded border border-line-1 bg-surface-1 text-xs text-ink-3">
        Loading providers…
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2">
      <SelectField
        value={current}
        options={eligible.map((p) => ({
          value: p.id,
          label: `${p.name} (${p.type})`,
        }))}
        onChange={(next) => onChange(next === '' ? undefined : next)}
        allowEmpty
        footer={<AddProviderRow />}
        onFocus={onFocus}
        onBlur={onBlur}
      />
      <ProviderStatusChip onOpenSettings={openSettings} />
    </div>
  )
}

function AddProviderRow() {
  return (
    <a
      href="/settings/providers#new"
      target="_blank"
      rel="noreferrer"
      className="flex items-center gap-2 px-3 py-1.5 text-[13px] font-mono text-brand-alt hover:bg-surface-2 transition-colors cursor-pointer"
    >
      <span className="text-sm">+</span>
      <span>Add new provider in Settings</span>
    </a>
  )
}
