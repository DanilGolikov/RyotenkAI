import { useEffect, useMemo, useState } from 'react'
import {
  useDeleteIntegrationToken,
  useIntegration,
  useIntegrationConfig,
  useSaveIntegrationConfig,
  useSetIntegrationToken,
  useTestIntegrationConnection,
} from '../../api/hooks/useIntegrations'
import { dumpYaml, safeYamlParse } from '../../lib/yaml'
import { HelpTooltip } from '../ConfigBuilder/HelpTooltip'
import { Spinner, Toggle } from '../ui'

const INPUT_CLS =
  'h-8 rounded bg-surface-1 border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors'

const ROW_CLS =
  'py-1.5 grid grid-cols-1 sm:grid-cols-[220px_minmax(0,1fr)] gap-2 sm:gap-4 items-start sm:items-center'

export function IntegrationConfigTab({
  integrationId,
  integrationType,
}: {
  integrationId: string
  integrationType: string
}) {
  if (integrationType === 'mlflow') {
    return <MLflowForm integrationId={integrationId} />
  }
  if (integrationType === 'huggingface') {
    return <HuggingFaceForm integrationId={integrationId} />
  }
  return (
    <div className="text-xs text-warn">
      No editor registered for integration type{' '}
      <span className="font-mono">{integrationType}</span>.
    </div>
  )
}

// ─────────────────────────────── MLflow ──────────────────────────────────

interface MLflowValues {
  tracking_uri: string
  local_tracking_uri: string
  ca_bundle_path: string
  system_metrics_sampling_interval: string
  system_metrics_samples_before_logging: string
  system_metrics_callback_enabled: boolean
  system_metrics_callback_interval: string
}

const MLFLOW_DEFAULTS: MLflowValues = {
  tracking_uri: '',
  local_tracking_uri: '',
  ca_bundle_path: '',
  system_metrics_sampling_interval: '5',
  system_metrics_samples_before_logging: '1',
  system_metrics_callback_enabled: false,
  system_metrics_callback_interval: '10',
}

function MLflowForm({ integrationId }: { integrationId: string }) {
  const detailQuery = useIntegration(integrationId)
  const configQuery = useIntegrationConfig(integrationId)
  const saveConfig = useSaveIntegrationConfig(integrationId)
  const setToken = useSetIntegrationToken(integrationId)
  const deleteToken = useDeleteIntegrationToken(integrationId)
  const testMut = useTestIntegrationConnection(integrationId)

  const hasToken = detailQuery.data?.has_token ?? false

  const [values, setValues] = useState<MLflowValues>(MLFLOW_DEFAULTS)
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

  // Hydrate from server YAML once on first load, or when the server
  // value changes and the form isn't dirty.
  useEffect(() => {
    if (!configQuery.data || dirty) return
    const parsed = safeYamlParse(configQuery.data.yaml) as Record<string, unknown> | null
    setValues(normalizeMLflow(parsed))
  }, [configQuery.data, dirty])

  function update<K extends keyof MLflowValues>(key: K, v: MLflowValues[K]) {
    setValues((prev) => ({ ...prev, [key]: v }))
    setDirty(true)
    setErrorMsg(null)
  }

  async function save() {
    setErrorMsg(null)
    try {
      const yaml = dumpYaml(serializeMLflow(values))
      await saveConfig.mutateAsync(yaml)

      if (tokenDirty) {
        const trimmed = tokenInput.trim()
        if (trimmed) {
          await setToken.mutateAsync(trimmed)
        } else if (hasToken) {
          await deleteToken.mutateAsync()
        }
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

  async function clearToken() {
    if (
      !window.confirm(
        'Remove the stored token? Projects using this integration will lose authentication until a new one is set.',
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

  if (detailQuery.isLoading || configQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading
      </div>
    )
  }
  if (detailQuery.error) {
    return <div className="text-sm text-err">{(detailQuery.error as Error).message}</div>
  }

  const saving =
    saveConfig.isPending || setToken.isPending || deleteToken.isPending
  const isDirty = dirty || tokenDirty

  // Required-field gate. Token is write-only (we only know about it via
  // ``has_token``), so "satisfied" means either the server already has
  // one OR the user is typing one in now.
  const tokenSatisfied = hasToken || tokenInput.trim().length > 0
  const trackingUriSatisfied = values.tracking_uri.trim().length > 0
  const missingRequired: string[] = []
  if (!tokenSatisfied) missingRequired.push('Token')
  if (!trackingUriSatisfied) missingRequired.push('Tracking URI')
  const saveDisabled = !isDirty || saving || missingRequired.length > 0

  return (
    <div className="max-w-3xl">
      {/* Token — stored write-only via /token endpoint, never in YAML. */}
      <FieldRow
        label="Token"
        required
        description="MLFLOW_TRACKING_TOKEN for authenticated trackers. Stored encrypted; never returned through the API."
      >
        <div className="flex items-center gap-2 flex-wrap">
          <input
            type={tokenReveal ? 'text' : 'password'}
            value={tokenInput}
            onChange={(e) => {
              setTokenInput(e.target.value)
              setTokenDirty(true)
            }}
            placeholder={hasToken ? '••••••••  (leave empty to keep current)' : 'paste token'}
            autoComplete="off"
            className={`${INPUT_CLS} w-[420px] max-w-full`}
          />
          <button
            type="button"
            onClick={() => setTokenReveal((v) => !v)}
            className="h-8 px-2.5 text-2xs rounded border border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2 transition"
          >
            {tokenReveal ? 'Hide' : 'Show'}
          </button>
          {hasToken && (
            <span className="text-ok border border-ok/40 bg-ok/10 rounded px-1.5 py-0.5 text-[0.65rem]">
              token set
            </span>
          )}
          {hasToken && (
            <button
              type="button"
              onClick={clearToken}
              className="h-8 px-2.5 text-2xs rounded border border-err/50 text-err hover:bg-err/10 hover:border-err transition"
            >
              Clear
            </button>
          )}
        </div>
      </FieldRow>

      <FieldRow
        label="Tracking URI"
        required
        description="Primary MLflow tracking URI — used by training/runtime and external clients."
      >
        <input
          type="text"
          value={values.tracking_uri}
          onChange={(e) => update('tracking_uri', e.target.value)}
          placeholder="http://localhost:5002"
          className={`${INPUT_CLS} w-full max-w-[640px]`}
        />
      </FieldRow>

      <FieldRow
        label="Local tracking URI"
        description="Optional separate URI for the local control plane / orchestrator."
      >
        <input
          type="text"
          value={values.local_tracking_uri}
          onChange={(e) => update('local_tracking_uri', e.target.value)}
          placeholder="http://localhost:5002"
          className={`${INPUT_CLS} w-full max-w-[640px]`}
        />
      </FieldRow>

      <FieldRow
        label="CA bundle path"
        description="Optional CA bundle for HTTPS verification (self-signed or private CA)."
      >
        <input
          type="text"
          value={values.ca_bundle_path}
          onChange={(e) => update('ca_bundle_path', e.target.value)}
          placeholder="/etc/ssl/mlflow-ca.pem"
          className={`${INPUT_CLS} w-full max-w-[640px]`}
        />
      </FieldRow>

      <div className="pt-4 border-t border-line-1">
        <div className="text-[0.6rem] font-medium uppercase tracking-[0.16em] text-ink-4 mb-3">
          System metrics
        </div>

        <FieldRow
          label="Sampling interval"
          description="GPU/CPU/RAM sampling interval, seconds. Applies when MLflow's built-in system-metrics logger is active."
        >
          <input
            type="number"
            min={1}
            max={60}
            value={values.system_metrics_sampling_interval}
            onChange={(e) => update('system_metrics_sampling_interval', e.target.value)}
            className={`${INPUT_CLS} w-32`}
          />
          <span className="text-2xs text-ink-4 ml-2">1–60 s</span>
        </FieldRow>

        <FieldRow
          label="Samples before logging"
          description="How many samples to collect before flushing a batch to MLflow."
        >
          <input
            type="number"
            min={1}
            max={10}
            value={values.system_metrics_samples_before_logging}
            onChange={(e) => update('system_metrics_samples_before_logging', e.target.value)}
            className={`${INPUT_CLS} w-32`}
          />
          <span className="text-2xs text-ink-4 ml-2">1–10</span>
        </FieldRow>

        <FieldRow
          label="Callback enabled"
          description="Enable SystemMetricsCallback — manual GPU/CPU tracking via pynvml/psutil. May hang on some cloud GPU images; keep off unless MLflow's built-in metrics miss what you need."
        >
          <Toggle
            checked={values.system_metrics_callback_enabled}
            onChange={(next) =>
              update('system_metrics_callback_enabled', next)
            }
            aria-label="Callback enabled"
          />
        </FieldRow>

        <FieldRow
          label="Callback interval"
          description="Log system metrics every N training steps when the callback is enabled."
        >
          <input
            type="number"
            min={1}
            max={100}
            value={values.system_metrics_callback_interval}
            onChange={(e) => update('system_metrics_callback_interval', e.target.value)}
            className={`${INPUT_CLS} w-32`}
          />
          <span className="text-2xs text-ink-4 ml-2">1–100 steps</span>
        </FieldRow>
      </div>

      <div className="pt-3 border-t border-line-1 space-y-2">
        <div className="flex items-center gap-3 flex-wrap">
          <button
            type="button"
            onClick={save}
            disabled={saveDisabled}
            title={
              missingRequired.length > 0
                ? `Required: ${missingRequired.join(', ')}`
                : undefined
            }
            className="btn-primary px-3 py-1.5 text-xs disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
          {flashOk && <span className="text-2xs text-ok">Saved</span>}
          {errorMsg && <span className="text-2xs text-err">{errorMsg}</span>}
          {missingRequired.length > 0 && (
            <span className="text-2xs text-warn">
              Required: {missingRequired.join(', ')}
            </span>
          )}
          <button
            type="button"
            onClick={runTest}
            disabled={testMut.isPending || isDirty}
            title={isDirty ? 'Save changes before testing' : 'Probe tracking URI with stored token'}
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
      </div>
    </div>
  )
}

function normalizeMLflow(parsed: Record<string, unknown> | null): MLflowValues {
  if (!parsed) return MLFLOW_DEFAULTS
  const str = (k: string) => {
    const v = parsed[k]
    return typeof v === 'string' ? v : ''
  }
  const num = (k: string, d: string) => {
    const v = parsed[k]
    return typeof v === 'number' ? String(v) : typeof v === 'string' ? v : d
  }
  const bool = (k: string, d: boolean) => {
    const v = parsed[k]
    return typeof v === 'boolean' ? v : d
  }
  return {
    tracking_uri: str('tracking_uri'),
    local_tracking_uri: str('local_tracking_uri'),
    ca_bundle_path: str('ca_bundle_path'),
    system_metrics_sampling_interval: num('system_metrics_sampling_interval', '5'),
    system_metrics_samples_before_logging: num('system_metrics_samples_before_logging', '1'),
    system_metrics_callback_enabled: bool('system_metrics_callback_enabled', false),
    system_metrics_callback_interval: num('system_metrics_callback_interval', '10'),
  }
}

function serializeMLflow(v: MLflowValues): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  if (v.tracking_uri.trim()) out.tracking_uri = v.tracking_uri.trim()
  if (v.local_tracking_uri.trim()) out.local_tracking_uri = v.local_tracking_uri.trim()
  if (v.ca_bundle_path.trim()) out.ca_bundle_path = v.ca_bundle_path.trim()
  out.system_metrics_sampling_interval = toInt(v.system_metrics_sampling_interval, 5)
  out.system_metrics_samples_before_logging = toInt(
    v.system_metrics_samples_before_logging,
    1,
  )
  out.system_metrics_callback_enabled = v.system_metrics_callback_enabled
  out.system_metrics_callback_interval = toInt(v.system_metrics_callback_interval, 10)
  return out
}

function toInt(s: string, fallback: number): number {
  const parsed = Number.parseInt(s, 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

// ──────────────────────────── HuggingFace ────────────────────────────────

function HuggingFaceForm({ integrationId }: { integrationId: string }) {
  const detailQuery = useIntegration(integrationId)
  const setToken = useSetIntegrationToken(integrationId)
  const deleteToken = useDeleteIntegrationToken(integrationId)
  const testMut = useTestIntegrationConnection(integrationId)

  const hasToken = detailQuery.data?.has_token ?? false

  const [tokenInput, setTokenInput] = useState('')
  const [reveal, setReveal] = useState(false)
  const [flashOk, setFlashOk] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [testResult, setTestResult] = useState<
    | { ok: boolean; detail: string; latency_ms: number | null }
    | null
  >(null)

  async function save() {
    setErrorMsg(null)
    const trimmed = tokenInput.trim()
    if (!trimmed) return
    try {
      await setToken.mutateAsync(trimmed)
      setTokenInput('')
      setReveal(false)
      setFlashOk(true)
      setTimeout(() => setFlashOk(false), 1500)
    } catch (exc) {
      setErrorMsg((exc as Error).message)
    }
  }

  async function clear() {
    if (
      !window.confirm(
        'Remove the stored token? Projects using this integration will lose authentication until a new one is set.',
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

  if (detailQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading
      </div>
    )
  }
  if (detailQuery.error) {
    return <div className="text-sm text-err">{(detailQuery.error as Error).message}</div>
  }

  return (
    <div className="space-y-4 max-w-3xl">
      <FieldRow
        label="Token"
        description="HF_TOKEN with write permission. Stored encrypted; never returned through the API."
      >
        <div className="flex items-center gap-2 flex-wrap">
          <input
            type={reveal ? 'text' : 'password'}
            value={tokenInput}
            onChange={(e) => setTokenInput(e.target.value)}
            placeholder={hasToken ? '••••••••  (enter new value to replace)' : 'paste token'}
            autoComplete="off"
            className={`${INPUT_CLS} w-[420px] max-w-full`}
          />
          <button
            type="button"
            onClick={() => setReveal((v) => !v)}
            className="h-8 px-2.5 text-2xs rounded border border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2 transition"
          >
            {reveal ? 'Hide' : 'Show'}
          </button>
          {hasToken && (
            <span className="text-ok border border-ok/40 bg-ok/10 rounded px-1.5 py-0.5 text-[0.65rem]">
              token set
            </span>
          )}
          {hasToken && (
            <button
              type="button"
              onClick={clear}
              className="h-8 px-2.5 text-2xs rounded border border-err/50 text-err hover:bg-err/10 hover:border-err transition"
            >
              Clear
            </button>
          )}
        </div>
      </FieldRow>

      <div className="pt-3 border-t border-line-1 space-y-2">
        <div className="flex items-center gap-3 flex-wrap">
          <button
            type="button"
            onClick={save}
            disabled={!tokenInput.trim() || setToken.isPending}
            className="btn-primary px-3 py-1.5 text-xs disabled:opacity-50"
          >
            {setToken.isPending ? 'Saving…' : hasToken ? 'Replace token' : 'Save token'}
          </button>
          {flashOk && <span className="text-2xs text-ok">Saved</span>}
          {errorMsg && <span className="text-2xs text-err">{errorMsg}</span>}
          <button
            type="button"
            onClick={runTest}
            disabled={testMut.isPending || !hasToken}
            title={hasToken ? 'Probe HuggingFace Hub with stored token' : 'Save a token first'}
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
      </div>
    </div>
  )
}

// ─────────────────────────── Shared primitives ───────────────────────────

function FieldRow({
  label,
  description,
  required,
  children,
}: {
  label: string
  description?: string
  required?: boolean
  children: React.ReactNode
}) {
  const labelText = useMemo(() => label, [label])
  return (
    <div className={ROW_CLS}>
      <div className="flex items-center gap-2 min-w-0 rounded bg-surface-1 border border-line-1 px-2.5 h-8">
        <span
          aria-hidden={!required}
          className={`inline-flex w-2 shrink-0 text-brand-warm text-xs leading-none ${
            required ? '' : 'invisible'
          }`}
        >
          *
        </span>
        <span className="flex-1 min-w-0 text-xs text-ink-2 font-medium tracking-tight truncate">
          {labelText}
          {required && <span className="sr-only"> (required)</span>}
        </span>
        <HelpTooltip text={description} label={`Help for ${label}`} />
      </div>
      <div className="w-full min-w-0 flex items-center flex-wrap">{children}</div>
    </div>
  )
}
