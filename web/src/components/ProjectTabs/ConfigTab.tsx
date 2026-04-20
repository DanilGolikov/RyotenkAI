import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  useProjectConfig,
  useSaveProjectConfig,
  useValidateProjectConfig,
} from '../../api/hooks/useProjects'
import { useConfigSchema } from '../../api/hooks/useConfigSchema'
import type { ConfigValidationResult } from '../../api/types'
import { useRef } from 'react'
import { ConfigBuilder } from '../ConfigBuilder/ConfigBuilder'
import { DiffBadge } from '../ConfigBuilder/DiffBadge'
import { PresetDropdown } from '../ConfigBuilder/PresetDropdown'
import { ProviderPickerField } from '../ConfigBuilder/ProviderPickerField'
import { ValidationBanner } from '../ConfigBuilder/ValidationBanner'
import { ValidationProvider } from '../ConfigBuilder/ValidationContext'
import {
  deriveGroupValidity,
  SETTINGS_JUMP_TARGET,
} from '../ConfigBuilder/validationMap'
import { YamlEditor } from '../YamlEditor'
import { dumpYaml, safeYamlParse } from '../../lib/yaml'
import { Spinner } from '../ui'

type ViewMode = 'form' | 'yaml'

export function ConfigTab({ projectId }: { projectId: string }) {
  const navigate = useNavigate()
  const configQuery = useProjectConfig(projectId)
  const schemaQuery = useConfigSchema()
  const saveMut = useSaveProjectConfig(projectId)
  const validateMut = useValidateProjectConfig(projectId)

  const [view, setView] = useState<ViewMode>('form')
  const [yamlText, setYamlText] = useState<string>('')
  const [formValue, setFormValue] = useState<Record<string, unknown>>({})
  const [dirty, setDirty] = useState(false)
  const [yamlParseError, setYamlParseError] = useState<string | null>(null)
  const [presetBaseline, setPresetBaseline] = useState<{
    name: string
    value: Record<string, unknown>
  } | null>(null)

  useEffect(() => {
    if (configQuery.data && !dirty) {
      const text = configQuery.data.yaml
      setYamlText(text)
      const parsed = safeYamlParse(text)
      setFormValue(parsed ?? {})
    }
  }, [configQuery.data, dirty])

  const validationResult: ConfigValidationResult | undefined = validateMut.data
  const fieldErrors = validationResult?.field_errors ?? {}

  // Debounced auto-validate on any change (form or yaml).
  useEffect(() => {
    if (!dirty) return
    const handle = window.setTimeout(() => {
      validateMut.mutate(yamlText)
    }, 900)
    return () => window.clearTimeout(handle)
  }, [yamlText, dirty])

  // Blur handler sent down into every field — debounced so a rapid
  // tab-through doesn't spam ``/validate``.
  const validateTimer = useRef<number | null>(null)
  const requestValidate = () => {
    if (validateTimer.current !== null) {
      window.clearTimeout(validateTimer.current)
    }
    validateTimer.current = window.setTimeout(() => {
      validateMut.mutate(yamlText)
      validateTimer.current = null
    }, 250)
  }
  // Accept caller-provided errors for the context (field_errors is
  // authoritative, ConfigTab owns this — no local setter needed).
  const noopSetErrors = () => undefined

  const groupValidity = useMemo(
    () => (validationResult ? deriveGroupValidity(validationResult.checks) : {}),
    [validationResult],
  )

  const statusLine = useMemo(() => {
    if (saveMut.isPending) return 'Saving…'
    if (validateMut.isPending) return 'Validating…'
    if (yamlParseError && view === 'form') return 'YAML has a parse error — form hidden'
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
      setYamlParseError('YAML is not a mapping. Form view is disabled until it parses.')
    } else {
      setFormValue(parsed)
      setYamlParseError(null)
    }
  }

  if (configQuery.isLoading || schemaQuery.isLoading) {
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
      <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="inline-flex rounded-md border border-line-1 overflow-hidden text-2xs">
          <button
            type="button"
            onClick={() => setView('form')}
            disabled={!!yamlParseError}
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
        <div className="ml-auto flex items-center gap-2">
          {presetBaseline && (
            <DiffBadge
              presetName={presetBaseline.name}
              baseline={presetBaseline.value}
              current={formValue}
              onClear={() => setPresetBaseline(null)}
            />
          )}
          <PresetDropdown
            dirty={dirty}
            onLoad={(preset) => {
              const parsed = safeYamlParse(preset.yaml) ?? {}
              setYamlText(preset.yaml)
              setFormValue(parsed)
              setDirty(true)
              setYamlParseError(null)
              setPresetBaseline({ name: preset.name, value: parsed })
            }}
          />
          <span className="text-2xs text-ink-3">{statusLine}</span>
        </div>
      </div>

      <ValidationBanner
        result={validationResult ?? null}
        isValidating={validateMut.isPending}
        hashPrefix="project"
        onJump={(group) => {
          if (group === SETTINGS_JUMP_TARGET) {
            navigate(`/projects/${encodeURIComponent(projectId)}/settings`)
            return
          }
          const nextHash = `#project:${group}`
          if (window.location.hash !== nextHash) {
            history.replaceState(null, '', nextHash)
            window.dispatchEvent(new HashChangeEvent('hashchange'))
          }
        }}
      />
      </div>

      {view === 'form' && schemaQuery.data ? (
        <ValidationProvider
          fieldErrors={fieldErrors}
          setFieldErrors={noopSetErrors}
          onRequestValidate={requestValidate}
        >
          <ConfigBuilder
            schema={schemaQuery.data}
            value={formValue}
            onChange={applyFormChange}
            hashPrefix="project"
            groupRenderers={{ providers: ProviderPickerField }}
            groupValidity={groupValidity}
          />
        </ValidationProvider>
      ) : view === 'form' && schemaQuery.error ? (
        <div className="text-sm text-err">{(schemaQuery.error as Error).message}</div>
      ) : (
        <YamlEditor value={yamlText} onChange={applyYamlChange} />
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

