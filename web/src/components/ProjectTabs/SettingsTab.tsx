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
    key: 'MLFLOW_TRACKING_URI',
    kind: 'text',
    placeholder: 'http://localhost:5000',
    description:
      'Optional runtime override for MLflow tracking URI — wins over the integration value when set. Useful in CI.',
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
        Values merged into the training subprocess env at launch time, on top
        of the parent process. Stored as{' '}
        <code className="font-mono text-ink-2">env.json</code> inside the
        project workspace — not committed to the shared config.
      </p>
      <div className="text-2xs text-ink-3 border border-line-1 rounded-md px-3 py-2 max-w-2xl bg-surface-1 space-y-1.5">
        <div>
          <span className="font-medium text-ink-2">Precedence at run-time</span>{' '}
          (highest → lowest):
        </div>
        <ol className="list-decimal list-inside space-y-0.5">
          <li>
            <span className="text-ink-2">Encrypted tokens</span> from{' '}
            <a
              href="/settings/integrations"
              className="text-brand-alt hover:underline"
            >
              Settings → Integrations
            </a>{' '}
            (HF) and{' '}
            <a
              href="/settings/providers"
              className="text-brand-alt hover:underline"
            >
              Settings → Providers
            </a>{' '}
            (RunPod) — resolved per‑integration at the point of use
          </li>
          <li>
            <span className="text-ink-2">Project env.json</span> (this tab) —
            per‑project override, wins over repo defaults
          </li>
          <li>
            <span className="text-ink-2">Repo‑root secrets.env</span> — shared
            team defaults; a mismatch warning is logged when a project
            override shadows it
          </li>
          <li>
            <span className="text-ink-2">Ambient shell env</span> — pre‑server
            environment; lowest precedence
          </li>
        </ol>
      </div>

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
          className="text-2xs text-ink-3 hover:text-ink-1 border border-line-1 hover:border-line-2 rounded-md px-2 py-1 transition"
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
// Same 200px label column, same input surface, same ? tooltip. Plain-text
// label (no pill chrome) so env rows read identically to the
// ConfigBuilder form. No status underlines — env doesn't have the
// validate pipeline that ConfigBuilder rows participate in. (`focused`
// is kept in the API for future use, currently a no-op since the input
// itself shows focus via its own focus-ring.)

function LabelText({
  label,
  description,
  mono = true,
}: {
  label: string
  description?: string
  mono?: boolean
}) {
  return (
    <div className="flex items-center gap-1.5 min-w-0 h-8 px-0.5">
      <span
        className={`flex-1 min-w-0 text-xs text-ink-2 tracking-tight truncate ${
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
  'h-8 rounded-md bg-surface-inset border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors'

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
  return (
    <div className="py-1.5 grid grid-cols-1 sm:grid-cols-[200px_minmax(0,1fr)] gap-1 sm:gap-4 items-center">
      <LabelText label={spec.key} description={spec.description} />
      <div className="flex items-center gap-2 w-fit max-w-full">
        {spec.kind === 'enum' ? (
          <SelectField
            value={value}
            options={(spec.options ?? []).map((v) => ({ value: v }))}
            onChange={onChange}
            allowEmpty
            triggerClassName="min-w-[160px]"
          />
        ) : spec.kind === 'secret' ? (
          <>
            <input
              type={reveal ? 'text' : 'password'}
              value={value}
              onChange={(e) => onChange(e.target.value)}
              placeholder={spec.placeholder}
              autoComplete="off"
              className={`${INPUT_CLS} w-[360px] max-w-full`}
            />
            <button
              type="button"
              onClick={() => setReveal((v) => !v)}
              className="h-8 px-2.5 text-2xs rounded-md border border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2 transition shrink-0"
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
  // Custom rows don't have a fixed label — the user types the env-var
  // name into the LEFT cell using INPUT_CLS so it sits on the same
  // `surface-inset` as every other input.
  return (
    <div className="py-1.5 grid grid-cols-1 sm:grid-cols-[200px_minmax(0,1fr)] gap-1 sm:gap-4 items-center">
      <input
        value={entry.key}
        onChange={(e) => onChange({ key: e.target.value })}
        placeholder="KEY_NAME"
        className={`${INPUT_CLS} w-full`}
      />
      <div className="flex items-center gap-2 w-fit max-w-full">
        <input
          value={entry.value}
          onChange={(e) => onChange({ value: e.target.value })}
          placeholder="value"
          className={`${INPUT_CLS} w-[360px] max-w-full`}
        />
        <button
          type="button"
          onClick={onRemove}
          title="Remove"
          className="w-8 h-8 inline-flex items-center justify-center rounded-md text-err hover:bg-err/10 transition text-xs shrink-0"
        >
          ✕
        </button>
      </div>
    </div>
  )
}
