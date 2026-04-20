import { useEffect, useState } from 'react'
import { useProjectEnv, useSaveProjectEnv } from '../../api/hooks/useProjects'
import { HelpTooltip } from '../ConfigBuilder/HelpTooltip'
import { SelectField } from '../ConfigBuilder/SelectField'
import { Spinner } from '../ui'

/**
 * Environment overrides for a project. Kept visually consistent with the
 * ConfigBuilder row grammar: `[label pill + ?][input]`, same 220px
 * column, same input look. Descriptions live in the ``?`` tooltip so
 * the page stays compact.
 *
 * Values are merged into the process env at run-time; storage is
 * `env.json` inside the project workspace — not committed to Git.
 */
interface EnvSpec {
  key: string
  kind?: 'text' | 'secret' | 'enum'
  options?: string[]
  placeholder?: string
  description?: string
}

const CATALOG: EnvSpec[] = [
  {
    key: 'HF_TOKEN',
    kind: 'secret',
    placeholder: 'token',
    description:
      'Hugging Face access token. Required for private models/datasets and for pushing adapters.',
  },
  {
    key: 'RUNPOD_API_KEY',
    kind: 'secret',
    placeholder: 'key',
    description:
      'RunPod API key. Required when the training provider is runpod; ignored for single_node.',
  },
  {
    key: 'MLFLOW_TRACKING_URI',
    kind: 'text',
    placeholder: 'http://localhost:5000',
    description: 'MLflow tracking server. Leave empty to let the pipeline auto-detect.',
  },
  {
    key: 'LOG_LEVEL',
    kind: 'enum',
    options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    description: 'Python logging verbosity for pipeline runs.',
  },
]

const CATALOG_KEYS = new Set(CATALOG.map((e) => e.key))

export function SettingsTab({ projectId }: { projectId: string }) {
  const envQuery = useProjectEnv(projectId)
  const saveMut = useSaveProjectEnv(projectId)

  const [values, setValues] = useState<Record<string, string>>({})
  const [extras, setExtras] = useState<{ key: string; value: string }[]>([])
  const [dirty, setDirty] = useState(false)

  useEffect(() => {
    if (!envQuery.data) return
    const env = envQuery.data.env ?? {}
    const next: Record<string, string> = {}
    const custom: { key: string; value: string }[] = []
    for (const [k, v] of Object.entries(env)) {
      if (CATALOG_KEYS.has(k)) next[k] = v
      else custom.push({ key: k, value: v })
    }
    setValues(next)
    setExtras(custom)
    setDirty(false)
  }, [envQuery.data])

  function update(key: string, value: string) {
    setValues((prev) => ({ ...prev, [key]: value }))
    setDirty(true)
  }

  function updateExtra(idx: number, patch: Partial<{ key: string; value: string }>) {
    setExtras((prev) => prev.map((e, i) => (i === idx ? { ...e, ...patch } : e)))
    setDirty(true)
  }
  function removeExtra(idx: number) {
    setExtras((prev) => prev.filter((_, i) => i !== idx))
    setDirty(true)
  }
  function addExtra() {
    setExtras((prev) => [...prev, { key: '', value: '' }])
    setDirty(true)
  }

  async function save() {
    const payload: Record<string, string> = {}
    for (const [k, v] of Object.entries(values)) {
      if (v.trim()) payload[k] = v
    }
    for (const { key, value } of extras) {
      const k = key.trim()
      if (!k) continue
      if (value.trim()) payload[k] = value
    }
    try {
      await saveMut.mutateAsync(payload)
      setDirty(false)
    } catch (exc) {
      window.alert((exc as Error).message || 'Failed to save environment.')
    }
  }

  if (envQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-xs text-ink-3">
        <Spinner /> loading settings
      </div>
    )
  }
  if (envQuery.error) {
    return <div className="text-xs text-err">{(envQuery.error as Error).message}</div>
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-ink-3 max-w-2xl">
        Values merged into the process env at run-time. Stored as{' '}
        <code className="font-mono text-ink-2">env.json</code> inside the project
        workspace — not committed to the shared config.
      </p>

      <div className="space-y-0.5">
        {CATALOG.map((spec) => (
          <EnvRow
            key={spec.key}
            spec={spec}
            value={values[spec.key] ?? ''}
            onChange={(v) => update(spec.key, v)}
          />
        ))}

        {extras.map((e, idx) => (
          <CustomEnvRow
            key={idx}
            entry={e}
            onChange={(patch) => updateExtra(idx, patch)}
            onRemove={() => removeExtra(idx)}
          />
        ))}
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={addExtra}
          className="text-2xs text-ink-3 hover:text-ink-1 border border-line-1 hover:border-line-2 rounded px-2 py-1 transition"
        >
          + Add variable
        </button>
      </div>

      <div className="flex items-center gap-3 pt-3 border-t border-line-1">
        <button
          type="button"
          onClick={save}
          disabled={!dirty || saveMut.isPending}
          className="btn-primary px-3 py-1.5 text-xs"
        >
          {saveMut.isPending ? 'Saving…' : 'Save settings'}
        </button>
        {saveMut.isSuccess && !dirty && (
          <span className="text-2xs text-ink-3">Saved</span>
        )}
        {saveMut.error && (
          <span className="text-2xs text-err">
            {(saveMut.error as Error).message}
          </span>
        )}
      </div>
    </div>
  )
}

// ───── Row primitives matching ConfigBuilder LabelledRow visuals ──────────
//
// Same 220px label pill, same input surface, same ? tooltip. No status
// underlines — env doesn't have the validate pipeline that ConfigBuilder
// rows participate in.

function LabelPill({
  label,
  description,
  mono = true,
  focused = false,
}: {
  label: string
  description?: string
  mono?: boolean
  focused?: boolean
}) {
  return (
    <div
      className={`flex items-center gap-2 min-w-0 rounded bg-surface-1 border ${
        focused ? 'border-brand' : 'border-line-1'
      } px-2.5 h-8 transition-colors`}
    >
      <span className="inline-flex w-2 shrink-0" aria-hidden />
      <span
        className={`flex-1 min-w-0 text-xs text-ink-2 font-medium tracking-tight truncate ${
          mono ? 'font-mono' : ''
        }`}
      >
        {label}
      </span>
      <HelpTooltip text={description} />
    </div>
  )
}

const INPUT_CLS =
  'h-8 rounded bg-surface-1 border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors'

function EnvRow({
  spec,
  value,
  onChange,
}: {
  spec: EnvSpec
  value: string
  onChange: (v: string) => void
}) {
  const [reveal, setReveal] = useState(false)
  const [focused, setFocused] = useState(false)
  const focusProps = {
    onFocus: () => setFocused(true),
    onBlur: () => setFocused(false),
  }
  return (
    <div className="py-1.5 grid grid-cols-1 sm:grid-cols-[220px_minmax(0,1fr)] gap-2 sm:gap-4 items-center">
      <LabelPill
        label={spec.key}
        description={spec.description}
        focused={focused}
      />
      <div className="flex items-center gap-2 w-fit max-w-full">
        {spec.kind === 'enum' ? (
          <SelectField
            value={value}
            options={(spec.options ?? []).map((v) => ({ value: v }))}
            onChange={onChange}
            allowEmpty
            triggerClassName="min-w-[160px]"
            {...focusProps}
          />
        ) : spec.kind === 'secret' ? (
          <>
            <input
              type={reveal ? 'text' : 'password'}
              value={value}
              onChange={(e) => onChange(e.target.value)}
              placeholder={spec.placeholder}
              autoComplete="off"
              {...focusProps}
              className={`${INPUT_CLS} w-[360px] max-w-full`}
            />
            <button
              type="button"
              onClick={() => setReveal((v) => !v)}
              className="h-8 px-2.5 text-2xs rounded border border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2 transition shrink-0"
            >
              {reveal ? 'Hide' : 'Show'}
            </button>
          </>
        ) : (
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={spec.placeholder}
            {...focusProps}
            className={`${INPUT_CLS} w-[360px] max-w-full`}
          />
        )}
      </div>
    </div>
  )
}

function CustomEnvRow({
  entry,
  onChange,
  onRemove,
}: {
  entry: { key: string; value: string }
  onChange: (patch: Partial<{ key: string; value: string }>) => void
  onRemove: () => void
}) {
  const [focused, setFocused] = useState(false)
  const focusProps = {
    onFocus: () => setFocused(true),
    onBlur: () => setFocused(false),
  }
  return (
    <div className="py-1.5 grid grid-cols-1 sm:grid-cols-[220px_minmax(0,1fr)] gap-2 sm:gap-4 items-center">
      <input
        value={entry.key}
        onChange={(e) => onChange({ key: e.target.value })}
        placeholder="KEY_NAME"
        {...focusProps}
        className={`h-8 rounded bg-surface-1 border ${
          focused ? 'border-brand' : 'border-line-1'
        } px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none hover:border-line-2 transition-colors w-full`}
      />
      <div className="flex items-center gap-2 w-fit max-w-full">
        <input
          value={entry.value}
          onChange={(e) => onChange({ value: e.target.value })}
          placeholder="value"
          {...focusProps}
          className={`${INPUT_CLS} w-[360px] max-w-full`}
        />
        <button
          type="button"
          onClick={onRemove}
          title="Remove"
          className="w-8 h-8 inline-flex items-center justify-center rounded text-err hover:bg-err/10 transition text-xs shrink-0"
        >
          ✕
        </button>
      </div>
    </div>
  )
}
