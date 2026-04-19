import { useEffect, useMemo, useState } from 'react'
import {
  useProviderConfig,
  useProviderTypes,
  useSaveProviderConfig,
  useValidateProviderConfig,
} from '../../api/hooks/useProviders'
import type { ConfigValidationResult } from '../../api/types'
import { ConfigBuilder } from '../ConfigBuilder/ConfigBuilder'
import type { PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { YamlView } from '../YamlView'
import { dumpYaml, safeYamlParse } from '../../lib/yaml'
import { Spinner } from '../ui'

type ViewMode = 'form' | 'yaml'

export function ProviderConfigTab({
  providerId,
  providerType,
}: {
  providerId: string
  providerType: string
}) {
  const configQuery = useProviderConfig(providerId)
  const typesQuery = useProviderTypes()
  const saveMut = useSaveProviderConfig(providerId)
  const validateMut = useValidateProviderConfig(providerId)

  const [view, setView] = useState<ViewMode>('form')
  const [yamlEditing, setYamlEditing] = useState(false)
  const [yamlText, setYamlText] = useState('')
  const [formValue, setFormValue] = useState<Record<string, unknown>>({})
  const [dirty, setDirty] = useState(false)
  const [yamlParseError, setYamlParseError] = useState<string | null>(null)

  useEffect(() => {
    if (configQuery.data && !dirty) {
      const text = configQuery.data.yaml
      setYamlText(text)
      const parsed = safeYamlParse(text)
      setFormValue(parsed ?? {})
    }
  }, [configQuery.data, dirty])

  const typeInfo = typesQuery.data?.types.find((t) => t.id === providerType)
  const schema = typeInfo?.json_schema as PipelineJsonSchema | undefined

  const validationResult: ConfigValidationResult | undefined = validateMut.data

  const statusLine = useMemo(() => {
    if (saveMut.isPending) return 'Saving…'
    if (validateMut.isPending) return 'Validating…'
    if (yamlParseError && view === 'form') return 'YAML parse error — form hidden'
    if (dirty) return 'Unsaved changes'
    if (saveMut.isSuccess) return 'Saved'
    return ''
  }, [saveMut.isPending, saveMut.isSuccess, validateMut.isPending, dirty, yamlParseError, view])

  function applyFormChange(next: Record<string, unknown>) {
    setFormValue(next)
    setYamlText(dumpYaml(next))
    setDirty(true)
    setYamlParseError(null)
  }

  function applyYamlChange(next: string) {
    setYamlText(next)
    setDirty(true)
    const parsed = safeYamlParse(next)
    if (parsed == null) {
      setYamlParseError('YAML is not a mapping. Form disabled until it parses.')
    } else {
      setFormValue(parsed)
      setYamlParseError(null)
    }
  }

  if (configQuery.isLoading || typesQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading config
      </div>
    )
  }
  if (configQuery.error) {
    return <div className="text-sm text-err">{(configQuery.error as Error).message}</div>
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <div className="inline-flex rounded-md border border-line-1 overflow-hidden text-2xs">
          <button
            type="button"
            onClick={() => setView('form')}
            disabled={!!yamlParseError || !schema}
            className={`px-3 py-1.5 transition ${
              view === 'form' ? 'bg-surface-2 text-ink-1' : 'text-ink-3 hover:text-ink-1 hover:bg-surface-2/60'
            } disabled:opacity-40`}
          >
            Form
          </button>
          <button
            type="button"
            onClick={() => setView('yaml')}
            className={`px-3 py-1.5 transition ${
              view === 'yaml' ? 'bg-surface-2 text-ink-1' : 'text-ink-3 hover:text-ink-1 hover:bg-surface-2/60'
            }`}
          >
            YAML
          </button>
        </div>
        <span className="ml-auto text-2xs text-ink-3">{statusLine}</span>
      </div>

      {view === 'form' && schema ? (
        <ConfigBuilder
          schema={schema}
          value={formValue}
          onChange={applyFormChange}
          hashPrefix={`provider:${providerId}`}
        />
      ) : view === 'form' && !schema ? (
        <div className="text-xs text-warn">
          No schema registered for provider type <span className="font-mono">{providerType}</span>.
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-2xs">
            <button
              type="button"
              onClick={() => setYamlEditing((v) => !v)}
              className="rounded-md border border-line-1 px-2 py-1 text-ink-3 hover:text-ink-1 hover:border-line-2"
            >
              {yamlEditing ? 'Preview' : 'Edit'}
            </button>
          </div>
          {yamlEditing ? (
            <div className="rounded-md border border-line-1 bg-surface-0 overflow-hidden">
              <textarea
                value={yamlText}
                onChange={(e) => applyYamlChange(e.target.value)}
                spellCheck={false}
                rows={24}
                className="w-full bg-surface-0 text-ink-1 font-mono text-xs px-4 py-3 focus:outline-none resize-y"
                placeholder="# provider config"
              />
            </div>
          ) : (
            <YamlView text={yamlText} maxHeight="max-h-[640px]" />
          )}
        </div>
      )}

      <div className="flex items-center gap-2 text-xs">
        <button
          type="button"
          onClick={() => validateMut.mutate(yamlText)}
          className="rounded-md border border-line-1 px-3 py-1.5 text-ink-2 hover:text-ink-1 hover:border-line-2"
          disabled={validateMut.isPending}
        >
          Validate
        </button>
        <button
          type="button"
          onClick={async () => {
            await saveMut.mutateAsync(yamlText)
            setDirty(false)
          }}
          className="btn-primary px-3 py-1.5"
          disabled={saveMut.isPending || !dirty}
        >
          {saveMut.isPending ? 'Saving…' : 'Save'}
        </button>
      </div>

      {validationResult && (
        <div className="rounded-md border border-line-1 bg-surface-1 p-3 space-y-1.5">
          <div
            className={`text-xs font-medium ${validationResult.ok ? 'text-ok' : 'text-err'}`}
          >
            {validationResult.ok ? 'Configuration looks valid' : 'Configuration has issues'}
          </div>
          {validationResult.checks.map((c, idx) => (
            <div key={idx} className="flex items-start gap-2 text-2xs">
              <span
                className={[
                  'w-1.5 h-1.5 mt-1.5 rounded-full shrink-0',
                  c.status === 'ok' ? 'bg-ok' : c.status === 'warn' ? 'bg-warn' : 'bg-err',
                ].join(' ')}
              />
              <div className="min-w-0">
                <div className="text-ink-1">{c.label}</div>
                {c.detail && <div className="text-ink-3 font-mono truncate">{c.detail}</div>}
              </div>
            </div>
          ))}
        </div>
      )}

      {yamlParseError && (
        <div className="rounded-md border border-warn/40 bg-warn/10 text-warn text-xs px-3 py-2">
          {yamlParseError}
        </div>
      )}

      {saveMut.error && (
        <div className="rounded-md border border-err/40 bg-err/10 text-err text-xs px-3 py-2">
          {(saveMut.error as Error).message}
        </div>
      )}
    </div>
  )
}
