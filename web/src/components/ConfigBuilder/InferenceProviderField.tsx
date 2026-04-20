import { useProviders } from '../../api/hooks/useProviders'
import { SelectField } from './SelectField'

interface Props {
  value: unknown
  onChange: (next: unknown) => void
  onFocus?: () => void
  onBlur?: () => void
}

/**
 * Dropdown for ``inference.provider``. Populated from Settings providers
 * whose saved config has a populated ``inference`` block — the backend
 * pre-computes ``has_inference`` on every provider summary, so the UI
 * doesn't need to fetch each config individually.
 *
 * Empty by default (dash placeholder). Providers without an inference
 * runtime are hidden — to add one, the user opens Settings → Providers.
 */
export function InferenceProviderField({ value, onChange, onFocus, onBlur }: Props) {
  const providersQuery = useProviders()
  const providers = providersQuery.data ?? []

  const eligible = providers.filter((p) => p.has_inference)
  const rawValue = typeof value === 'string' ? value : ''
  // Hide legacy/unknown values so the user sees a dash instead of a
  // provider id that doesn't exist in Settings. The actual value is still
  // in state (round-trips through YAML) until the user picks a real one.
  const isEligible = !rawValue || eligible.some((p) => p.id === rawValue)
  const displayValue = isEligible ? rawValue : ''

  if (providersQuery.isLoading && eligible.length === 0) {
    return (
      <div className="h-8 inline-flex items-center px-2.5 rounded border border-line-1 bg-surface-1 text-xs text-ink-3">
        Loading providers…
      </div>
    )
  }

  return (
    <SelectField
      value={displayValue}
      options={eligible.map((p) => ({
        value: p.id,
        label: `${p.name} (${p.type})`,
      }))}
      onChange={(next) => onChange(next === '' ? undefined : next)}
      allowEmpty
      placeholder="—"
      footer={<AddProviderRow />}
      onFocus={onFocus}
      onBlur={onBlur}
    />
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
