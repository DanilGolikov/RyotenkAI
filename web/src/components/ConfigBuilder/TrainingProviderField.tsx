import { useQueries } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { useParams } from 'react-router-dom'
import { api } from '../../api/client'
import { useProviders } from '../../api/hooks/useProviders'
import { qk } from '../../api/queryKeys'
import type { ConfigResponse } from '../../api/types'
import { ProviderStatusChip } from './ProviderStatusChip'
import { SelectField } from './SelectField'

interface Props {
  value: unknown
  onChange: (next: unknown) => void
  onFocus?: () => void
  onBlur?: () => void
}

/**
 * Dropdown for ``training.provider``. Populated from Settings providers,
 * filtered to those whose saved config has a non-empty ``training``
 * block — so the user only sees providers actually wired up for
 * training. Free-form text is not allowed; to add a new provider the
 * user goes to Settings → Providers.
 */
export function TrainingProviderField({ value, onChange, onFocus, onBlur }: Props) {
  const providersQuery = useProviders()
  const providers = providersQuery.data ?? []
  const navigate = useNavigate()
  const { id: projectId } = useParams<{ id: string }>()
  const openSettings = () => {
    if (projectId) navigate(`/projects/${encodeURIComponent(projectId)}/settings`)
  }

  const configQueries = useQueries({
    queries: providers.map((p) => ({
      queryKey: qk.providerConfig(p.id),
      queryFn: () =>
        api.get<ConfigResponse>(`/providers/${encodeURIComponent(p.id)}/config`),
    })),
  })

  const loading =
    providersQuery.isLoading || configQueries.some((q) => q.isLoading && !q.data)

  const eligible = providers
    .map((p, i) => ({ provider: p, config: configQueries[i]?.data }))
    .filter(({ config }) => {
      const t = config?.parsed_json?.training
      return t && typeof t === 'object' && !Array.isArray(t) && Object.keys(t).length > 0
    })
    .map(({ provider }) => provider)

  const current = typeof value === 'string' ? value : ''

  if (loading && eligible.length === 0) {
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
