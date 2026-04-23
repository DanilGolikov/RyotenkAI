import { useProviders } from '../../api/hooks/useProviders'
import { SelectField } from './SelectField'

interface Props {
  value: unknown
  onChange: (next: unknown) => void
  onFocus?: () => void
  onBlur?: () => void
  /** Root form value — supplied by FieldRenderer when available; used
   *  here to keep ``inference.enabled`` in sync with ``inference.provider``
   *  (PR3: ``enabled`` is hidden from the form and derived from provider
   *  selection, so YAML round-trips stay consistent without a separate
   *  switch). */
  rootValue?: Record<string, unknown>
  onRootChange?: (next: Record<string, unknown>) => void
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

/**
 * Dropdown for ``inference.provider``. Populated from Settings providers
 * whose saved config has a populated ``inference`` block — the backend
 * pre-computes ``has_inference`` on every provider summary, so the UI
 * doesn't need to fetch each config individually.
 *
 * Empty by default (dash placeholder). Providers without an inference
 * runtime are hidden — to add one, the user opens Settings → Providers.
 *
 * When ``rootValue``/``onRootChange`` are supplied, this also writes
 * ``inference.enabled`` atomically — the backend schema still carries
 * ``enabled``, but the FE is the single source of truth and keeps the
 * two fields consistent.
 */
export function InferenceProviderField({
  value,
  onChange,
  onFocus,
  onBlur,
  rootValue,
  onRootChange,
}: Props) {
  const providersQuery = useProviders()
  const providers = providersQuery.data ?? []

  const eligible = providers.filter((p) => p.has_inference)
  const rawValue = typeof value === 'string' ? value : ''
  const isEligible = !rawValue || eligible.some((p) => p.id === rawValue)
  const displayValue = isEligible ? rawValue : ''

  function handleChange(next: string) {
    const providerId = next === '' ? undefined : next
    if (rootValue && onRootChange) {
      const inference = isRecord(rootValue.inference) ? rootValue.inference : {}
      onRootChange({
        ...rootValue,
        inference: {
          ...inference,
          provider: providerId,
          enabled: Boolean(providerId),
        },
      })
    } else {
      onChange(providerId)
    }
  }

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
      onChange={handleChange}
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
