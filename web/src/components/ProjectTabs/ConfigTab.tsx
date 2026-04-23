import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  useProjectConfig,
  useProjectConfigVersions,
  useReadConfigVersion,
  useSaveProjectConfig,
  useValidateProjectConfig,
} from '../../api/hooks/useProjects'
import { useConfigPresets } from '../../api/hooks/useConfigPresets'
import { useConfigSchema } from '../../api/hooks/useConfigSchema'
import type { ConfigValidationResult } from '../../api/types'
import { ConfigBuilder } from '../ConfigBuilder/ConfigBuilder'
import { DiffBadge } from '../ConfigBuilder/DiffBadge'
import { PresetPickerModal } from '../ConfigBuilder/PresetPickerModal'
import { ProviderPickerField } from '../ConfigBuilder/ProviderPickerField'
import { ValidationBanner } from '../ConfigBuilder/ValidationBanner'
import { ValidationProvider } from '../ConfigBuilder/ValidationContext'
import {
  deriveGroupValidity,
  SETTINGS_JUMP_TARGET,
} from '../ConfigBuilder/validationMap'
import { VersionDiffModal } from '../ConfigBuilder/VersionDiffModal'
import { YamlEditor } from '../YamlEditor'
import { dumpYaml, safeYamlParse } from '../../lib/yaml'
import { useConfigDraft } from '../../hooks/useConfigDraft'
import { Spinner } from '../ui'

// Draft persistence lives in `hooks/useConfigDraft.ts`. See the file's
// header for the rationale — TL;DR: drafts are a device-local notepad
// for half-filled forms, decoupled from the server's validating Save.

type ViewMode = 'form' | 'yaml'

type StatusKindLocal = 'saving' | 'validating' | 'dirty' | 'saved' | 'yamlError' | null

function StatusPill({ status }: { status: { kind: StatusKindLocal; text: string } }) {
  if (!status.kind) return <span className="text-2xs text-ink-3" aria-live="polite" />

  const palette: Record<Exclude<StatusKindLocal, null>, {
    text: string
    dot: string
    pulse: boolean
  }> = {
    saving:     { text: 'text-info',         dot: 'bg-info',  pulse: false },
    validating: { text: 'text-ink-3',        dot: 'bg-ink-3', pulse: false },
    // `dirty` is the one non-CTA spot where we let brand-burgundy through —
    // a pulsing dot + brand-strong text signals "pending action" without a
    // modal or banner.
    dirty:      { text: 'text-brand-strong', dot: 'bg-brand', pulse: true  },
    saved:      { text: 'text-ok',           dot: 'bg-ok',    pulse: false },
    yamlError:  { text: 'text-err',          dot: 'bg-err',   pulse: false },
  }
  const p = palette[status.kind]
  const showSpinner = status.kind === 'saving' || status.kind === 'validating'

  return (
    <span
      aria-live="polite"
      className={`inline-flex items-center gap-1.5 text-2xs transition-colors ${p.text}`}
    >
      {showSpinner ? (
        <Spinner />
      ) : (
        <span className="relative inline-flex w-1.5 h-1.5">
          {p.pulse && (
            <span
              aria-hidden
              className={`absolute inset-0 rounded-full ${p.dot} opacity-75 animate-ping`}
            />
          )}
          <span className={`relative inline-block w-1.5 h-1.5 rounded-full ${p.dot}`} />
        </span>
      )}
      <span className="whitespace-nowrap">{status.text}</span>
    </span>
  )
}

export function ConfigTab({ projectId }: { projectId: string }) {
  const navigate = useNavigate()
  const configQuery = useProjectConfig(projectId)
  const schemaQuery = useConfigSchema()
  const saveMut = useSaveProjectConfig(projectId)
  const validateMut = useValidateProjectConfig(projectId)

  const [view, setView] = useState<ViewMode>('form')
  // Monotonic signal we bump on view toggle so the open preset-picker
  // modal force-closes itself — otherwise the modal lingers over the
  // new view while the user is trying to interact with it.
  const [presetCloseToken, setPresetCloseToken] = useState(0)
  const [yamlText, setYamlText] = useState<string>('')
  const [formValue, setFormValue] = useState<Record<string, unknown>>({})
  const [dirty, setDirty] = useState(false)
  const [yamlParseError, setYamlParseError] = useState<string | null>(null)
  // Local "Save as draft" — see `useConfigDraft` for the rationale
  // (device-local stash so half-filled forms aren't lost to a
  // validating server Save).
  const draft = useConfigDraft(projectId)

  const [compareOpen, setCompareOpen] = useState(false)
  const versionsQuery = useProjectConfigVersions(projectId)
  const latestVersionFilename = useMemo(() => {
    const vs = versionsQuery.data?.versions ?? []
    if (vs.length === 0) return null
    // API returns newest first (see ConfigVersionsResponse). Keep
    // defensive sort in case that ever changes.
    return [...vs].sort((a, b) =>
      b.created_at.localeCompare(a.created_at),
    )[0]?.filename ?? null
  }, [versionsQuery.data])
  const compareVersionQuery = useReadConfigVersion(
    projectId,
    compareOpen ? latestVersionFilename : null,
  )
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

      // Surface the draft-restore banner only if the stashed draft
      // actually differs from what's now on the server.
      draft.probe(text)
    }
    // `draft.probe` is referentially stable (useCallback), but eslint
    // can't see through the hook object, so list the projectId
    // dependency that actually changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configQuery.data, dirty, projectId])

  // Auto-seed new projects with the first preset so a fresh project
  // starts from a sensible ≤ 1B baseline instead of an empty editor.
  // Runs at most once per project session, and only while:
  //   - the project's current config is empty (never saved, or cleared)
  //   - the user hasn't started editing (not dirty)
  //   - presets have loaded from the backend
  // Applied silently (no preview modal) — the whole point is "new
  // project already comes populated". If the user hates the defaults
  // they can either edit or Load preset another one.
  const presetsQuery = useConfigPresets()
  const seededRef = useRef<string | null>(null)
  useEffect(() => {
    if (seededRef.current === projectId) return
    if (!configQuery.data) return
    if (dirty) return
    const serverText = (configQuery.data.yaml ?? '').trim()
    if (serverText) {
      // Project already has a saved config — nothing to seed.
      seededRef.current = projectId
      return
    }
    const first = presetsQuery.data?.presets?.[0]
    if (!first) return
    setYamlText(first.yaml)
    const parsed = safeYamlParse(first.yaml)
    setFormValue(parsed ?? {})
    setDirty(true)
    seededRef.current = projectId
  }, [configQuery.data, presetsQuery.data, dirty, projectId])

  const validationResult: ConfigValidationResult | undefined = validateMut.data
  const fieldErrors = validationResult?.field_errors ?? {}

  // Explicit-only validation. Auto-validate (debounced on edit +
  // on-blur per field) was removed because: it spammed /validate on
  // every keystroke, made the banner jitter between states mid-edit,
  // and could pull focus on repeated 0→N error transitions. Validation
  // now runs ONLY when the user clicks Validate or Save.

  // Warn on tab-close / reload when there are unsaved form changes.
  // Only installs the listener when dirty so we don't pay the
  // beforeunload tax on every page.
  useEffect(() => {
    if (!dirty) return
    function onBeforeUnload(e: BeforeUnloadEvent) {
      e.preventDefault()
      // Older browsers required returnValue to be set; modern browsers
      // show their own generic copy regardless. Both branches here so
      // the guard works on Safari / Firefox consistently.
      e.returnValue = ''
      return ''
    }
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [dirty])

  // No-op: ValidationContext still wires `onRequestValidate` down to
  // every field's onBlur, but we intentionally don't hit the server.
  // Client-side schema checks in ValidationContext stay active and
  // paint bad fields red in real-time without any network traffic.
  const requestValidate = () => undefined
  // Accept caller-provided errors for the context (field_errors is
  // authoritative, ConfigTab owns this — no local setter needed).
  const noopSetErrors = () => undefined

  const groupValidity = useMemo(
    () => (validationResult ? deriveGroupValidity(validationResult.checks) : {}),
    [validationResult],
  )

  // Status descriptor used by the header status pill. Brand-burgundy
  // reserved for `dirty` (needs attention), ok-green for `saved`,
  // info-sky for in-flight work. See FRONTEND_GUIDELINES.md "Brand-usage
  // policy" — dirty state is one of the few sites where brand is allowed
  // outside CTAs.
  type StatusKind = 'saving' | 'validating' | 'dirty' | 'saved' | 'yamlError' | null
  const status: { kind: StatusKind; text: string } = useMemo(() => {
    if (saveMut.isPending) return { kind: 'saving', text: 'Saving…' }
    if (validateMut.isPending) return { kind: 'validating', text: 'Validating…' }
    if (yamlParseError && view === 'form')
      return { kind: 'yamlError', text: 'YAML parse error — form hidden' }
    if (dirty) return { kind: 'dirty', text: 'Unsaved changes' }
    if (saveMut.isSuccess) return { kind: 'saved', text: 'Saved' }
    return { kind: null, text: '' }
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
            onClick={() => { setView('form'); setPresetCloseToken((t) => t + 1) }}
            disabled={!!yamlParseError}
            className={`px-3 py-1.5 transition ${
              view === 'form' ? 'bg-surface-2 text-ink-1' : 'text-ink-3 hover:text-ink-1 hover:bg-surface-2/60'
            } disabled:opacity-40`}
          >
            Form
          </button>
          <button
            type="button"
            onClick={() => { setView('yaml'); setPresetCloseToken((t) => t + 1) }}
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
          <PresetPickerModal
            dirty={dirty}
            closeToken={presetCloseToken}
            current={formValue}
            onLoad={(preset) => {
              const parsed = safeYamlParse(preset.yaml) ?? {}
              setYamlText(preset.yaml)
              setFormValue(parsed)
              setDirty(true)
              setYamlParseError(null)
              setPresetBaseline({ name: preset.name, value: parsed })
            }}
          />
          <StatusPill status={status} />
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
        onJumpToField={(path) => {
          // Field jump is only meaningful in Form view — FieldAnchor
          // elements don't exist in YAML mode. Force-switch first,
          // then set hash so ConfigBuilder's hashchange handler
          // picks up the dotted trailer and scrolls to the field.
          if (view !== 'form') {
            setView('form')
            setPresetCloseToken((t) => t + 1)
          }
          const nextHash = `#project:${path}`
          if (window.location.hash !== nextHash) {
            history.replaceState(null, '', nextHash)
          }
          // Dispatch unconditionally: if we just switched views, the
          // FieldAnchor isn't mounted yet, so the event needs to fire
          // *after* React paints the new form tree. The 120ms delay
          // matches the 100ms scroll timeout inside ConfigBuilder.
          window.setTimeout(() => {
            window.dispatchEvent(new HashChangeEvent('hashchange'))
          }, 120)
        }}
      />

      {draft.draftPrompt && (
        <div className="relative rounded-lg border border-info/30 bg-info/[0.05] overflow-hidden">
          <span
            aria-hidden
            className="absolute inset-y-0 left-0 w-[3px] bg-info"
          />
          <div className="flex items-center gap-3 pl-4 pr-3 py-2.5">
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold text-ink-1">
                Local draft available
              </div>
              <div className="text-[0.7rem] text-ink-3">
                Saved {new Date(draft.draftPrompt.savedAt).toLocaleString()} —
                restore the in-progress form?
              </div>
            </div>
            <button
              type="button"
              onClick={() => {
                if (!draft.draftPrompt) return
                const parsed = safeYamlParse(draft.draftPrompt.yaml)
                setYamlText(draft.draftPrompt.yaml)
                setFormValue(parsed ?? {})
                setDirty(true)
                draft.dismissPrompt()
              }}
              className="rounded-md border border-info/40 bg-info/10 text-info hover:bg-info/15 px-3 py-1 text-xs transition"
            >
              Restore
            </button>
            <button
              type="button"
              onClick={draft.clear}
              className="rounded-md border border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2 px-3 py-1 text-xs transition"
            >
              Discard
            </button>
          </div>
        </div>
      )}
      </div>

      {view === 'form' && schemaQuery.data ? (
        <ValidationProvider
          fieldErrors={fieldErrors}
          setFieldErrors={noopSetErrors}
          validationResult={validationResult ?? null}
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

      {(() => {
        // Explain *why* Save is disabled — previously users saw a
        // greyed-out button with no hint. We prefer the inline caption
        // to a tooltip alone because hover discovery fails for
        // keyboard / touchpad-only users on a desktop dev tool.
        // Save is disabled only while saving or when there's nothing
        // new to commit. Validation state is surfaced by the banner
        // above; we don't second-guess the user's intent to save — if
        // the server can't accept the YAML it will reject the PUT and
        // `saveMut.error` renders as a red block below.
        const saveDisabledReason: string | null = saveMut.isPending
          ? 'Saving…'
          : !dirty
          ? 'No changes to save'
          : null
        const saveDisabled = saveDisabledReason !== null
        return (
          <div className="space-y-1">
            {/* Button order: Save · Save as Draft · Validate — the
                "commit" actions first (primary → secondary), the
                "checking" action last.

                Validation is now EXPLICIT — it runs only when the
                user clicks Save (which fires validate + save in
                parallel) or Validate (check-without-committing).
                Typing no longer spams /validate.

                Save is enabled whenever the form has changes. We
                deliberately don't gate on validation failures — the
                server rejects bad YAML on PUT and surfaces the real
                error below. Client-side schema checks still paint
                bad fields red in real time without hitting the
                network. */}
            <div className="flex items-center gap-2 text-xs">
              <button
                type="button"
                onClick={async () => {
                  // Kick off validate alongside save so the banner
                  // reflects the state of *this* yamlText after the
                  // round-trip (we don't block on it — save still
                  // owns success/failure; validate just paints the
                  // checks list).
                  validateMut.mutate(yamlText)
                  await saveMut.mutateAsync(yamlText)
                  setDirty(false)
                  draft.clear()
                }}
                className="btn-primary px-3 py-1.5"
                disabled={saveDisabled}
                title={saveDisabledReason ?? undefined}
                aria-describedby={saveDisabled ? 'save-disabled-reason' : undefined}
              >
                {saveMut.isPending ? 'Saving…' : 'Save'}
              </button>
              <button
                type="button"
                onClick={() => draft.save(yamlText)}
                title={
                  // Long-form explanation lives in useConfigDraft.ts;
                  // tooltip keeps the in-hand summary close to the
                  // button so users don't need to read the source.
                  'Stash the current form locally (browser-only) so ' +
                  "you don't lose it while you chase down a missing " +
                  'value. Skips server validation — use Save to commit.'
                }
                className={[
                  'rounded-md border px-3 py-1.5 text-xs transition',
                  draft.savedFlash
                    ? 'border-info bg-info/10 text-info'
                    : 'border-line-1 text-ink-2 hover:text-ink-1 hover:border-line-2',
                ].join(' ')}
              >
                {draft.savedFlash ? 'Draft saved ✓' : 'Save as draft'}
              </button>
              <button
                type="button"
                onClick={() => validateMut.mutate(yamlText)}
                className="rounded-md border border-line-1 px-3 py-1.5 text-ink-2 hover:text-ink-1 hover:border-line-2"
                disabled={validateMut.isPending}
              >
                Validate
              </button>
              {/* Compare moved into the main button group: it's a peer
                  inspection action — Save / Save as Draft / Validate /
                  Compare. Previously parked at `ml-auto` (right edge),
                  the eye had to skip the whole row to spot it. */}
              <button
                type="button"
                onClick={() => setCompareOpen(true)}
                className="rounded-md border border-line-1 px-3 py-1.5 text-ink-2 hover:text-ink-1 hover:border-line-2 disabled:opacity-40 disabled:cursor-not-allowed"
                disabled={!latestVersionFilename}
                title={
                  latestVersionFilename
                    ? 'Compare current form with the most recent saved version'
                    : 'No saved versions yet'
                }
              >
                Compare with last run
              </button>
              {saveDisabled && saveDisabledReason && (
                <span
                  id="save-disabled-reason"
                  className="ml-auto text-2xs text-ink-3"
                >
                  {saveDisabledReason}
                </span>
              )}
            </div>
          </div>
        )
      })()}

      {compareOpen && latestVersionFilename && (
        compareVersionQuery.data ? (
          <VersionDiffModal
            baselineYaml={compareVersionQuery.data.yaml}
            baselineLabel={latestVersionFilename}
            current={formValue}
            onClose={() => setCompareOpen(false)}
          />
        ) : compareVersionQuery.error ? (
          <div className="rounded-md border border-err/40 bg-err/10 text-err text-xs px-3 py-2">
            Failed to load last version: {(compareVersionQuery.error as Error).message}
          </div>
        ) : null
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

