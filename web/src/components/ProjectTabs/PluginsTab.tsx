import { useMemo, useState } from 'react'
import yaml from 'js-yaml'
import {
  useProjectConfig,
  useSaveProjectConfig,
} from '../../api/hooks/useProjects'
import { usePlugins } from '../../api/hooks/usePlugins'
import type { PluginKind, PluginManifest } from '../../api/types'
import { dumpYaml } from '../../lib/yaml'
import { Spinner } from '../ui'

const TOGGLEABLE_KINDS: { id: PluginKind; label: string; help: string }[] = [
  { id: 'validation', label: 'Validation', help: 'Dataset pre-flight checks.' },
  { id: 'evaluation', label: 'Evaluation', help: 'Post-training model evaluators.' },
  { id: 'reward', label: 'Reward', help: 'Reward function for GRPO / SAPO strategies. Only one can be active at a time — it applies to every GRPO/SAPO strategy in this project.' },
]

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

type AttachedEntry = { id: string; plugin: string }

/**
 * Read the current plugin list for a given kind from the parsed config.
 * - validation plugins live under every datasets.<name>.validations.plugins[].
 *   For the on/off toggle we treat "attached" as "present on *any* dataset".
 * - evaluation plugins live under evaluation.evaluators.plugins[].
 */
function grpoStrategyIndices(parsed: Record<string, unknown>): number[] {
  const training = isRecord(parsed.training) ? parsed.training : {}
  const strategies = Array.isArray((training as Record<string, unknown>).strategies)
    ? ((training as Record<string, unknown>).strategies as unknown[])
    : []
  const out: number[] = []
  strategies.forEach((s, idx) => {
    if (!isRecord(s)) return
    const t = typeof s.strategy_type === 'string' ? s.strategy_type.toLowerCase() : ''
    if (t === 'grpo' || t === 'sapo') out.push(idx)
  })
  return out
}

function attachedPluginsByKind(
  kind: PluginKind,
  parsed: Record<string, unknown>,
): Record<string, AttachedEntry> {
  const out: Record<string, AttachedEntry> = {}
  if (kind === 'reward') {
    const training = isRecord(parsed.training) ? parsed.training : {}
    const strategies = Array.isArray((training as Record<string, unknown>).strategies)
      ? ((training as Record<string, unknown>).strategies as unknown[])
      : []
    for (const s of strategies) {
      if (!isRecord(s)) continue
      const params = isRecord(s.params) ? s.params : {}
      const name = typeof params.reward_plugin === 'string' ? params.reward_plugin : ''
      if (name) out[name] = { id: name, plugin: name }
    }
    return out
  }
  if (kind === 'evaluation') {
    const evalCfg = isRecord(parsed.evaluation) ? parsed.evaluation : {}
    const evaluators = isRecord((evalCfg as Record<string, unknown>).evaluators)
      ? ((evalCfg as Record<string, unknown>).evaluators as Record<string, unknown>)
      : {}
    const plugins = Array.isArray(evaluators.plugins) ? (evaluators.plugins as unknown[]) : []
    for (const p of plugins) {
      if (!isRecord(p)) continue
      const name = typeof p.plugin === 'string' ? p.plugin : ''
      const id = typeof p.id === 'string' ? p.id : name
      if (name) out[name] = { id, plugin: name }
    }
    return out
  }

  // validation
  const datasets = isRecord(parsed.datasets) ? parsed.datasets : {}
  for (const dsKey of Object.keys(datasets)) {
    const ds = (datasets as Record<string, unknown>)[dsKey]
    if (!isRecord(ds)) continue
    const validations = isRecord(ds.validations) ? ds.validations : null
    const plugins = validations && Array.isArray(validations.plugins)
      ? (validations.plugins as unknown[])
      : []
    for (const p of plugins) {
      if (!isRecord(p)) continue
      const name = typeof p.plugin === 'string' ? p.plugin : ''
      const id = typeof p.id === 'string' ? p.id : name
      if (name && !out[name]) out[name] = { id, plugin: name }
    }
  }
  return out
}

function withAttachmentToggled(
  kind: PluginKind,
  parsed: Record<string, unknown>,
  plugin: PluginManifest,
  enable: boolean,
): Record<string, unknown> {
  const next = structuredClone(parsed) as Record<string, unknown>
  const uniqueId = plugin.id  // keep it simple — one entry per plugin kind

  if (kind === 'reward') {
    const training = (isRecord(next.training) ? next.training : {}) as Record<string, unknown>
    if (!isRecord(next.training)) next.training = training
    const strategies = Array.isArray(training.strategies)
      ? (training.strategies as unknown[])
      : []
    training.strategies = strategies.map((s) => {
      if (!isRecord(s)) return s
      const t = typeof s.strategy_type === 'string' ? s.strategy_type.toLowerCase() : ''
      if (t !== 'grpo' && t !== 'sapo') return s
      const params = isRecord(s.params) ? { ...s.params } : {}
      if (enable) {
        params.reward_plugin = plugin.id
        Object.assign(params, plugin.suggested_params ?? {})
      } else if (params.reward_plugin === plugin.id) {
        delete params.reward_plugin
      }
      return { ...s, params }
    })
    return next
  }

  if (kind === 'evaluation') {
    const evalCfg = (isRecord(next.evaluation) ? next.evaluation : {}) as Record<string, unknown>
    if (!isRecord(next.evaluation)) next.evaluation = evalCfg
    if (enable && evalCfg.enabled === undefined) evalCfg.enabled = true

    const evaluators = (isRecord(evalCfg.evaluators) ? evalCfg.evaluators : {}) as Record<string, unknown>
    if (!isRecord(evalCfg.evaluators)) evalCfg.evaluators = evaluators
    const list = Array.isArray(evaluators.plugins) ? (evaluators.plugins as unknown[]) : []
    const current = list.filter((e) => !(isRecord(e) && e.plugin === plugin.id))
    if (enable) {
      current.push({
        id: uniqueId,
        plugin: plugin.id,
        enabled: true,
        save_report: true,
        params: plugin.suggested_params ?? {},
        thresholds: plugin.suggested_thresholds ?? {},
      })
    }
    evaluators.plugins = current
    return next
  }

  // validation — attach on every dataset (or create a `default` one if none).
  const datasets = (isRecord(next.datasets) ? next.datasets : {}) as Record<string, unknown>
  if (!isRecord(next.datasets)) next.datasets = datasets
  const keys = Object.keys(datasets)
  if (keys.length === 0) {
    datasets.default = { source_type: 'local', source_local: { local_paths: { train: '' } } }
    keys.push('default')
  }
  for (const key of keys) {
    const ds = (isRecord(datasets[key]) ? datasets[key] : {}) as Record<string, unknown>
    if (!isRecord(datasets[key])) datasets[key] = ds
    const validations = (isRecord(ds.validations) ? ds.validations : {}) as Record<string, unknown>
    if (!isRecord(ds.validations)) ds.validations = validations
    const list = Array.isArray(validations.plugins) ? (validations.plugins as unknown[]) : []
    const filtered = list.filter((e) => !(isRecord(e) && e.plugin === plugin.id))
    if (enable) {
      filtered.push({
        id: uniqueId,
        plugin: plugin.id,
        params: plugin.suggested_params ?? {},
        thresholds: plugin.suggested_thresholds ?? {},
      })
    }
    validations.plugins = filtered
  }
  return next
}

interface Props {
  projectId: string
}

export function PluginsTab({ projectId }: Props) {
  const [activeKind, setActiveKind] = useState<PluginKind>('evaluation')
  const configQuery = useProjectConfig(projectId)
  const pluginsQuery = usePlugins(activeKind)
  const saveMut = useSaveProjectConfig(projectId)

  const parsed = configQuery.data?.parsed_json ?? {}

  const attached = useMemo(
    () => attachedPluginsByKind(activeKind, parsed),
    [activeKind, parsed],
  )

  const grpoCount = useMemo(() => grpoStrategyIndices(parsed).length, [parsed])

  async function togglePlugin(plugin: PluginManifest, enable: boolean) {
    let patched = parsed
    // Reward is radio-style: turning a plugin on first clears the others.
    if (activeKind === 'reward' && enable) {
      for (const otherName of Object.keys(attached)) {
        if (otherName === plugin.id) continue
        const stub = { ...plugin, id: otherName }
        patched = withAttachmentToggled('reward', patched, stub, false)
      }
    }
    const next = withAttachmentToggled(activeKind, patched, plugin, enable)
    const yamlText = dumpYaml(next)
    await saveMut.mutateAsync(yamlText)
  }

  if (configQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading config
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {TOGGLEABLE_KINDS.map((k) => (
          <button
            key={k.id}
            type="button"
            onClick={() => setActiveKind(k.id)}
            className={[
              'text-2xs rounded-md px-3 py-1.5 border transition',
              activeKind === k.id
                ? 'border-brand text-ink-1 bg-surface-2'
                : 'border-line-1 text-ink-3 hover:border-line-2 hover:text-ink-1',
            ].join(' ')}
          >
            {k.label}
          </button>
        ))}
        <span className="ml-auto text-2xs text-ink-3">
          {saveMut.isPending ? 'Saving…' : saveMut.isSuccess ? 'Saved' : ''}
        </span>
      </div>
      <div className="text-2xs text-ink-3">
        {TOGGLEABLE_KINDS.find((k) => k.id === activeKind)?.help}{' '}
        <span className="text-ink-4">Toggle adds the plugin to your project config with its suggested params.</span>
      </div>
      {activeKind === 'reward' && grpoCount === 0 && (
        <div className="rounded-md border border-warn/40 bg-warn/10 text-warn text-2xs px-3 py-2">
          No GRPO or SAPO strategy found in <span className="font-mono">training.strategies[]</span>. Enabling a reward plugin writes nothing until you add one.
        </div>
      )}
      {activeKind === 'reward' && grpoCount > 0 && (
        <div className="text-2xs text-ink-3">
          Will apply to {grpoCount} GRPO/SAPO strateg{grpoCount === 1 ? 'y' : 'ies'} in this project.
        </div>
      )}

      {pluginsQuery.isLoading ? (
        <div className="flex items-center gap-2 text-sm text-ink-3">
          <Spinner /> loading plugins
        </div>
      ) : pluginsQuery.error ? (
        <div className="text-sm text-err">{(pluginsQuery.error as Error).message}</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3">
          {(pluginsQuery.data?.plugins ?? []).map((p) => (
            <PluginToggleCard
              key={p.id}
              plugin={p}
              enabled={!!attached[p.id]}
              disabled={saveMut.isPending}
              onToggle={(next) => togglePlugin(p, next)}
            />
          ))}
        </div>
      )}

      {saveMut.error ? (
        <div className="rounded-md border border-err/40 bg-err/10 text-err text-xs px-3 py-2">
          {(saveMut.error as Error).message}
        </div>
      ) : null}
    </div>
  )
}

function PluginToggleCard({
  plugin,
  enabled,
  disabled,
  onToggle,
}: {
  plugin: PluginManifest
  enabled: boolean
  disabled: boolean
  onToggle: (next: boolean) => void
}) {
  const params = yaml.dump(plugin.suggested_params, { indent: 2 }).trim()
  const thresholds = yaml.dump(plugin.suggested_thresholds, { indent: 2 }).trim()
  return (
    <div
      className={[
        'rounded-lg border bg-surface-1 p-4 space-y-2 transition',
        enabled
          ? 'border-ok/70 ring-1 ring-ok/40 shadow-[0_0_0_1px_rgba(74,222,128,0.15)]'
          : 'border-line-1 hover:border-line-2',
      ].join(' ')}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-sm font-medium text-ink-1 truncate">{plugin.name}</div>
          <div className="text-2xs text-ink-3 font-mono">v{plugin.version}</div>
        </div>
        <label className="inline-flex items-center gap-2 text-2xs cursor-pointer">
          <input
            type="checkbox"
            checked={enabled}
            disabled={disabled}
            onChange={(e) => onToggle(e.target.checked)}
            className="accent-ok"
          />
          <span className={enabled ? 'text-ok' : 'text-ink-3'}>
            {enabled ? 'enabled' : 'off'}
          </span>
        </label>
      </div>
      {plugin.description && (
        <div className="text-xs text-ink-2 leading-snug">{plugin.description}</div>
      )}
      {params && params !== '{}' && (
        <pre className="text-[0.6rem] font-mono text-ink-3 bg-surface-0 border border-line-1 rounded px-2 py-1 overflow-auto">
{`params:\n${params}`}
        </pre>
      )}
      {thresholds && thresholds !== '{}' && (
        <pre className="text-[0.6rem] font-mono text-ink-3 bg-surface-0 border border-line-1 rounded px-2 py-1 overflow-auto">
{`thresholds:\n${thresholds}`}
        </pre>
      )}
    </div>
  )
}
