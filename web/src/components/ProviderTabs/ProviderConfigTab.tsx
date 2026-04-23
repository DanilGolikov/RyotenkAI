import { useEffect, useMemo, useState } from 'react'
import {
  useDeleteProviderToken,
  useProvider,
  useProviderConfig,
  useProviderTypes,
  useSaveProviderConfig,
  useSetProviderToken,
  useTestProviderConnection,
  useValidateProviderConfig,
} from '../../api/hooks/useProviders'
import type { ConfigValidationResult } from '../../api/types'
import { ConfigBuilder } from '../ConfigBuilder/ConfigBuilder'
import type { PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { HelpTooltip } from '../ConfigBuilder/HelpTooltip'
import { YamlView } from '../YamlView'
import { dumpYaml, safeYamlParse } from '../../lib/yaml'
import { Spinner } from '../ui'

type ViewMode = 'form' | 'yaml'

const INPUT_CLS =
  'h-8 rounded bg-surface-1 border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors'

export function ProviderConfigTab({
  providerId,
  providerType,
}: {
  providerId: string
  providerType: string
}) {
  const detailQuery = useProvider(providerId)
  const configQuery = useProviderConfig(providerId)
  const typesQuery = useProviderTypes()
  const saveConfig = useSaveProviderConfig(providerId)
  const validateMut = useValidateProviderConfig(providerId)
  const setToken = useSetProviderToken(providerId)
  const deleteToken = useDeleteProviderToken(providerId)
  const testMut = useTestProviderConnection(providerId)

  const hasToken =
    (detailQuery.data as { has_token?: boolean } | undefined)?.has_token ?? false

  const [view, setView] = useState<ViewMode>('form')
  const [yamlText, setYamlText] = useState('')
  const [formValue, setFormValue] = useState<Record<string, unknown>>({})
  const [dirty, setDirty] = useState(false)

  const [tokenInput, setTokenInput] = useState('')
  const [tokenDirty, setTokenDirty] = useState(false)
  const [tokenReveal, setTokenReveal] = useState(false)
  const [flashOk, setFlashOk] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [testResult, setTestResult] = useState<
    | { ok: boolean; detail: string; latency_ms: number | null }
    | null
  >(null)

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
    if (saveConfig.isPending) return 'Saving…'
    if (validateMut.isPending) return 'Validating…'
    if (dirty || tokenDirty) return 'Unsaved changes'
    if (saveConfig.isSuccess) return 'Saved'
    return ''
  }, [
    saveConfig.isPending,
    saveConfig.isSuccess,
    validateMut.isPending,
    dirty,
    tokenDirty,
  ])

  function applyFormChange(next: Record<string, unknown>) {
    setFormValue(next)
    setYamlText(dumpYaml(next))
    setDirty(true)
    setErrorMsg(null)
  }

  async function save() {
    setErrorMsg(null)
    try {
      await saveConfig.mutateAsync(yamlText)
      if (tokenDirty) {
        const trimmed = tokenInput.trim()
        if (trimmed) await setToken.mutateAsync(trimmed)
        else if (hasToken) await deleteToken.mutateAsync()
        setTokenInput('')
        setTokenDirty(false)
        setTokenReveal(false)
      }
      setDirty(false)
      setFlashOk(true)
      setTimeout(() => setFlashOk(false), 1500)
    } catch (exc) {
      setErrorMsg((exc as Error).message)
    }
  }

  async function clearTokenAction() {
    if (
      !window.confirm(
        'Remove the stored token? Pipelines using this provider will lose authentication until a new one is set.',
      )
    )
      return
    try {
      await deleteToken.mutateAsync()
    } catch (exc) {
      setErrorMsg((exc as Error).message)
    }
  }

  async function runTest() {
    setTestResult(null)
    try {
      const res = await testMut.mutateAsync()
      setTestResult(res)
    } catch (exc) {
      setTestResult({ ok: false, detail: (exc as Error).message, latency_ms: null })
    }
  }

  if (detailQuery.isLoading || configQuery.isLoading || typesQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading
      </div>
    )
  }
  if (configQuery.error) {
    return <div className="text-sm text-err">{(configQuery.error as Error).message}</div>
  }

  const saving = saveConfig.isPending || setToken.isPending || deleteToken.isPending
  const isDirty = dirty || tokenDirty

  return (
    <div className="space-y-4">
      {/* Token — write-only, stored encrypted; never lands in YAML. */}
      <TokenRow
        hasToken={hasToken}
        value={tokenInput}
        onChange={(v) => {
          setTokenInput(v)
          setTokenDirty(true)
        }}
        reveal={tokenReveal}
        onToggleReveal={() => setTokenReveal((v) => !v)}
        onClear={clearTokenAction}
      />

      <div className="flex items-center gap-2">
        <div className="inline-flex rounded-md border border-line-1 overflow-hidden text-2xs">
          <button
            type="button"
            onClick={() => setView('form')}
            disabled={!schema}
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
        /* YAML view is read-only for now. Editing is blocked at the UI
           layer — the Form view is the authoritative editor, and
           hand-editing YAML can silently produce shapes the schema
           rejects at Save time. */
        <YamlView text={yamlText} maxHeight="max-h-[640px]" />
      )}

      <div className="pt-3 border-t border-line-1 space-y-2">
        <div className="flex items-center gap-3 flex-wrap">
          <button
            type="button"
            onClick={() => validateMut.mutate(yamlText)}
            className="rounded-md border border-line-1 px-3 py-1.5 text-xs text-ink-2 hover:text-ink-1 hover:border-line-2"
            disabled={validateMut.isPending}
          >
            Validate
          </button>
          <button
            type="button"
            onClick={save}
            disabled={!isDirty || saving}
            className="btn-primary px-3 py-1.5 text-xs disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
          {flashOk && <span className="text-2xs text-ok">Saved</span>}
          {errorMsg && <span className="text-2xs text-err">{errorMsg}</span>}
          <button
            type="button"
            onClick={runTest}
            disabled={testMut.isPending || isDirty}
            title={isDirty ? 'Save changes before testing' : 'Probe connection with stored token'}
            className="ml-auto rounded-md border border-line-1 px-3 py-1.5 text-2xs text-ink-2 hover:text-ink-1 hover:border-line-2 transition disabled:opacity-50"
          >
            {testMut.isPending ? 'Testing…' : 'Test connection'}
          </button>
        </div>

        {testResult && (
          <div
            className={[
              'rounded-md px-3 py-2 text-2xs flex items-start gap-2',
              testResult.ok
                ? 'border border-ok/40 bg-ok/10 text-ok'
                : 'border border-err/40 bg-err/10 text-err',
            ].join(' ')}
          >
            <span className="font-medium shrink-0">{testResult.ok ? 'OK' : 'FAILED'}</span>
            <span className="min-w-0 flex-1">{testResult.detail}</span>
            {testResult.latency_ms != null && (
              <span className="text-ink-3 shrink-0 font-mono">
                {testResult.latency_ms} ms
              </span>
            )}
          </div>
        )}

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

      </div>
    </div>
  )
}

// ─────────────────────────── UI primitives ──────────────────────────────

function TokenRow({
  hasToken,
  value,
  onChange,
  reveal,
  onToggleReveal,
  onClear,
}: {
  hasToken: boolean
  value: string
  onChange: (v: string) => void
  reveal: boolean
  onToggleReveal: () => void
  onClear: () => void
}) {
  return (
    <div className="py-1.5 grid grid-cols-1 sm:grid-cols-[220px_minmax(0,1fr)] gap-2 sm:gap-4 items-start sm:items-center">
      <div className="flex items-center gap-2 min-w-0 rounded bg-surface-1 border border-line-1 px-2.5 h-8">
        <span className="flex-1 min-w-0 text-xs text-ink-2 font-medium tracking-tight truncate">
          Token
        </span>
        <HelpTooltip
          text="Bearer credential for this provider (e.g. RUNPOD_API_KEY, SSH passphrase). Stored encrypted on disk; never returned through the API."
          label="Help for Token"
        />
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <input
          type={reveal ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={hasToken ? '••••••••  (leave empty to keep current)' : 'paste token'}
          autoComplete="off"
          className={`${INPUT_CLS} w-[420px] max-w-full`}
        />
        <button
          type="button"
          onClick={onToggleReveal}
          className="h-8 px-2.5 text-2xs rounded border border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2 transition"
        >
          {reveal ? 'Hide' : 'Show'}
        </button>
        {hasToken && (
          <span className="text-[0.65rem] text-ok border border-ok/40 bg-ok/10 rounded px-1.5 py-0.5">
            token set
          </span>
        )}
        {hasToken && (
          <button
            type="button"
            onClick={onClear}
            className="h-8 px-2.5 text-2xs rounded border border-err/50 text-err hover:bg-err/10 hover:border-err transition"
          >
            Clear
          </button>
        )}
      </div>
    </div>
  )
}
