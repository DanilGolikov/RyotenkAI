import { useState } from 'react'
import { useIntegrations, useTestIntegrationConnection } from '../../api/hooks/useIntegrations'
import { SelectField } from './SelectField'

interface Props {
  value: unknown
  onChange: (next: unknown) => void
  onFocus?: () => void
  onBlur?: () => void
}

/** Dropdown for ``experiment_tracking.mlflow.integration``.
 *  Lists integrations of type ``mlflow`` from Settings + a
 *  deep-link to create a new one. Shows an inline "Test connection"
 *  button next to the selected integration to validate tracking-URI
 *  reachability without leaving the ConfigBuilder. */
export function MLflowIntegrationField({ value, onChange, onFocus, onBlur }: Props) {
  const integrationsQuery = useIntegrations('mlflow')
  const integrations = integrationsQuery.data ?? []
  const current = typeof value === 'string' ? value : ''
  const testMut = useTestIntegrationConnection(current)
  const [result, setResult] = useState<{ ok: boolean; detail: string } | null>(null)

  const isEligible = !current || integrations.some((i) => i.id === current)
  const displayValue = isEligible ? current : ''

  async function runTest() {
    setResult(null)
    try {
      const res = await testMut.mutateAsync()
      setResult(res)
    } catch (exc) {
      setResult({ ok: false, detail: (exc as Error).message })
    }
  }

  if (integrationsQuery.isLoading && integrations.length === 0) {
    return (
      <div className="h-8 inline-flex items-center px-2.5 rounded border border-line-1 bg-surface-1 text-xs text-ink-3">
        Loading integrations…
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <SelectField
        value={displayValue}
        options={integrations.map((i) => ({ value: i.id, label: i.name }))}
        onChange={(next) => onChange(next === '' ? undefined : next)}
        allowEmpty
        placeholder="—"
        footer={<AddIntegrationRow type="mlflow" />}
        onFocus={onFocus}
        onBlur={onBlur}
      />
      {displayValue && (
        <button
          type="button"
          onClick={runTest}
          disabled={testMut.isPending}
          className="h-8 px-2.5 text-2xs rounded border border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2 transition disabled:opacity-50"
        >
          {testMut.isPending ? 'Testing…' : 'Test'}
        </button>
      )}
      {result && (
        <span
          className={[
            'text-[0.65rem] px-1.5 py-0.5 rounded border whitespace-nowrap',
            result.ok
              ? 'text-ok border-ok/40 bg-ok/10'
              : 'text-err border-err/40 bg-err/10',
          ].join(' ')}
          title={result.detail}
        >
          {result.ok ? 'OK' : 'FAIL'} — {result.detail.slice(0, 60)}
        </span>
      )}
    </div>
  )
}

function AddIntegrationRow({ type }: { type: string }) {
  return (
    <a
      href={`/settings/integrations#new=${type}`}
      target="_blank"
      rel="noreferrer"
      className="flex items-center gap-2 px-3 py-1.5 text-[13px] font-mono text-brand-alt hover:bg-surface-2 transition-colors cursor-pointer"
    >
      <span className="text-sm">+</span>
      <span>Add new {type} integration</span>
    </a>
  )
}
