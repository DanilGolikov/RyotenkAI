/**
 * Required-env block inside the plugin Configure modal.
 *
 * Plugins declare their env contract in `manifest.toml` (mirrored by
 * the `REQUIRED_ENV` ClassVar in code). The values themselves live in
 * the project's shared `env.json` — same place general project envs
 * are stored — so a single secret used by multiple plugin instances
 * doesn't have to be retyped.
 *
 * Vars whose `managed_by` is set to `integrations` / `providers`
 * render as read-only with a "Configure in Settings →" link. Plain-
 * text override of managed credentials is intentionally blocked from
 * this surface — see Settings tab note for rationale.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { useProjectEnv, useSaveProjectEnv } from '../../api/hooks/useProjects'
import type { PluginRequiredEnv } from '../../api/types'
import { Spinner } from '../ui'

const MANAGED_REDIRECT: Record<NonNullable<PluginRequiredEnv['managed_by']>, { label: string; href: string }> = {
  '': { label: '', href: '' },
  integrations: { label: 'Settings → Integrations', href: '/settings/integrations' },
  providers: { label: 'Settings → Providers', href: '/settings/providers' },
}

interface Props {
  projectId: string
  required: PluginRequiredEnv[]
}

export function PluginEnvSection({ projectId, required }: Props) {
  const envQuery = useProjectEnv(projectId)
  const saveMut = useSaveProjectEnv(projectId)

  const serverEnv = useMemo(() => envQuery.data?.env ?? {}, [envQuery.data])
  const [draft, setDraft] = useState<Record<string, string>>(serverEnv)
  const [reveal, setReveal] = useState<Record<string, boolean>>({})
  const userTouchedRef = useRef<Set<string>>(new Set())

  // Sync from server unless the user has touched a key locally —
  // mirrors the DescriptionEditor pattern.
  useEffect(() => {
    setDraft((prev) => {
      const next = { ...prev }
      for (const [k, v] of Object.entries(serverEnv)) {
        if (userTouchedRef.current.has(k)) continue
        next[k] = v
      }
      return next
    })
  }, [serverEnv])

  const dirty = useMemo(() => {
    for (const spec of required) {
      if ((draft[spec.name] ?? '') !== (serverEnv[spec.name] ?? '')) return true
    }
    return false
  }, [draft, serverEnv, required])

  if (required.length === 0) return null

  const update = (name: string, value: string) => {
    userTouchedRef.current.add(name)
    setDraft((prev) => ({ ...prev, [name]: value }))
  }

  const save = async () => {
    // Merge unchanged server entries with our local edits so we don't
    // accidentally drop env values the modal isn't aware of.
    const payload: Record<string, string> = { ...serverEnv }
    for (const spec of required) {
      const value = (draft[spec.name] ?? '').trim()
      if (value) payload[spec.name] = value
      else delete payload[spec.name]
    }
    await saveMut.mutateAsync(payload)
    userTouchedRef.current.clear()
  }

  return (
    <section className="space-y-2">
      <header className="flex items-baseline gap-2 border-b border-line-1 pb-1">
        <div className="text-xs font-semibold text-ink-1">Required environment variables</div>
        <div className="text-[0.65rem] text-ink-4">·</div>
        <div className="text-[0.65rem] text-ink-3">{required.length}</div>
      </header>
      <p className="text-2xs text-ink-3">
        Stored in the project's <code className="font-mono text-ink-2">env.json</code> next to other project envs.
        Leave a value empty to clear it.
      </p>
      <div className="space-y-2">
        {required.map((spec) => {
          const managed = spec.managed_by ? MANAGED_REDIRECT[spec.managed_by] : { label: '', href: '' }
          const isManaged = !!managed.label
          const isSecret = spec.secret !== false
          const value = draft[spec.name] ?? ''
          const hasServerValue = (serverEnv[spec.name] ?? '').length > 0
          const showValue = reveal[spec.name] ?? false
          return (
            <div key={spec.name} className="grid grid-cols-1 sm:grid-cols-[200px_minmax(0,1fr)] gap-1 sm:gap-3 items-center">
              <div className="flex items-center gap-1.5 min-w-0 px-0.5">
                <span className="flex-1 min-w-0 text-xs text-ink-2 font-mono truncate">
                  {spec.name}
                  {!spec.optional && <span className="ml-0.5 text-brand-warm" aria-hidden>*</span>}
                </span>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {isManaged ? (
                  <div className="flex items-center gap-2 text-2xs text-ink-3">
                    {hasServerValue ? (
                      <span className="pill pill-ok">set</span>
                    ) : (
                      <span className="pill pill-idle">unset</span>
                    )}
                    <span>
                      managed by{' '}
                      <a href={managed.href} className="text-brand-alt underline">{managed.label}</a>
                    </span>
                  </div>
                ) : (
                  <>
                    <input
                      type={isSecret && !showValue ? 'password' : 'text'}
                      value={value}
                      onChange={(e) => update(spec.name, e.target.value)}
                      placeholder={hasServerValue ? '••••••••' : isSecret ? 'paste secret' : 'value'}
                      autoComplete="off"
                      className="h-8 rounded-md bg-surface-inset border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors w-[320px] max-w-full"
                    />
                    {isSecret && (
                      <button
                        type="button"
                        onClick={() => setReveal((prev) => ({ ...prev, [spec.name]: !showValue }))}
                        className="h-8 px-2.5 text-2xs rounded-md border border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2 transition shrink-0"
                      >
                        {showValue ? 'Hide' : 'Show'}
                      </button>
                    )}
                  </>
                )}
                {spec.description && (
                  <span className="text-2xs text-ink-4 max-w-[300px] truncate" title={spec.description}>
                    — {spec.description}
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={save}
          disabled={!dirty || saveMut.isPending || envQuery.isLoading}
          className="rounded-md border border-line-1 px-3 py-1 text-2xs text-ink-2 hover:text-ink-1 hover:border-line-2 transition disabled:opacity-40"
        >
          {saveMut.isPending ? 'Saving env…' : 'Save env values'}
        </button>
        {envQuery.isLoading && <Spinner />}
        {saveMut.isSuccess && !dirty && <span className="text-2xs text-ok">Saved</span>}
        {saveMut.error && <span className="text-2xs text-err">{(saveMut.error as Error).message}</span>}
      </div>
    </section>
  )
}
