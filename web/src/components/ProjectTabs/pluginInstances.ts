/**
 * Pure helpers for reading / writing plugin instances inside a parsed
 * pipeline config. Keeping them pure (no React, no YAML stringify) so
 * the DnD handlers in ``PluginsTab`` can call them synchronously against
 * local form state before the async ``saveConfig`` round-trip.
 *
 * Kind contracts:
 *   - validation: ``datasets.<default|first>.validations.plugins[]``
 *     is the single source we edit. We don't fan-out changes across
 *     every dataset any more — multi-instance only makes sense on a
 *     single dataset scope. If the user has multiple datasets they
 *     edit each through its own YAML today; the UI lives on ``default``.
 *   - evaluation: ``evaluation.evaluators.plugins[]`` (multi-instance).
 *   - reward: radio-style — ``training.strategies[*].params.reward_plugin``
 *     for every strategy that is grpo / sapo. Only one distinct id at a
 *     time across the project; replaces the previous reward silently.
 *   - reports: ``reports.sections: list[str]`` (single-instance per id,
 *     order matters and is what the user drags).
 */

import type { PluginKind, PluginManifest } from '../../api/types'

export interface PluginInstance {
  /** Unique instance identifier inside this kind's list. For reports this
   *  is the same as ``pluginId`` (reports are single-instance). */
  instanceId: string
  /** Reference to the catalog plugin by its manifest id. */
  pluginId: string
}

export function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

const DEFAULT_DATASET_KEY = 'default'

function firstDatasetKey(parsed: Record<string, unknown>): string {
  const datasets = isRecord(parsed.datasets) ? parsed.datasets : {}
  const keys = Object.keys(datasets)
  return keys.find((k) => k === DEFAULT_DATASET_KEY) ?? keys[0] ?? DEFAULT_DATASET_KEY
}

function ensurePath(
  parent: Record<string, unknown>,
  key: string,
  make: () => Record<string, unknown>,
): Record<string, unknown> {
  if (!isRecord(parent[key])) parent[key] = make()
  return parent[key] as Record<string, unknown>
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

export function readInstances(
  kind: PluginKind,
  parsed: Record<string, unknown>,
): PluginInstance[] {
  if (kind === 'reports') {
    const reports = isRecord(parsed.reports) ? parsed.reports : {}
    const sections = Array.isArray(reports.sections) ? reports.sections : []
    return sections
      .filter((s): s is string => typeof s === 'string')
      .map((id) => ({ instanceId: id, pluginId: id }))
  }

  if (kind === 'reward') {
    const training = isRecord(parsed.training) ? parsed.training : {}
    const strategies = Array.isArray(training.strategies) ? (training.strategies as unknown[]) : []
    // Collect distinct reward ids in order of first occurrence.
    const seen = new Set<string>()
    const out: PluginInstance[] = []
    for (const s of strategies) {
      if (!isRecord(s)) continue
      const t = typeof s.strategy_type === 'string' ? s.strategy_type.toLowerCase() : ''
      if (t !== 'grpo' && t !== 'sapo' && t !== 'dpo' && t !== 'orpo') continue
      const params = isRecord(s.params) ? s.params : {}
      const id = typeof params.reward_plugin === 'string' ? params.reward_plugin : ''
      if (id && !seen.has(id)) {
        seen.add(id)
        out.push({ instanceId: id, pluginId: id })
      }
    }
    return out
  }

  if (kind === 'evaluation') {
    const evalCfg = isRecord(parsed.evaluation) ? parsed.evaluation : {}
    const evaluators = isRecord(evalCfg.evaluators) ? evalCfg.evaluators : {}
    const plugins = Array.isArray(evaluators.plugins) ? (evaluators.plugins as unknown[]) : []
    return plugins
      .filter(isRecord)
      .map((p) => ({
        instanceId: typeof p.id === 'string' ? p.id : String(p.plugin ?? ''),
        pluginId: typeof p.plugin === 'string' ? p.plugin : '',
      }))
      .filter((x) => x.pluginId)
  }

  // validation — default dataset only
  const datasets = isRecord(parsed.datasets) ? parsed.datasets : {}
  const key = firstDatasetKey(parsed)
  const ds = isRecord(datasets[key]) ? (datasets[key] as Record<string, unknown>) : {}
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const plugins = Array.isArray(validations.plugins) ? (validations.plugins as unknown[]) : []
  return plugins
    .filter(isRecord)
    .map((p) => ({
      instanceId: typeof p.id === 'string' ? p.id : String(p.plugin ?? ''),
      pluginId: typeof p.plugin === 'string' ? p.plugin : '',
    }))
    .filter((x) => x.pluginId)
}

// ---------------------------------------------------------------------------
// Mutations (return new top-level parsed object, do not mutate input)
// ---------------------------------------------------------------------------

function cloneParsed(parsed: Record<string, unknown>): Record<string, unknown> {
  return structuredClone(parsed) as Record<string, unknown>
}

/** Add a fresh instance from the palette. Returns the new parsed config
 *  AND the generated instanceId so the caller can focus / highlight it. */
export function addInstance(
  kind: PluginKind,
  parsed: Record<string, unknown>,
  manifest: PluginManifest,
): { next: Record<string, unknown>; instanceId: string } {
  const existing = readInstances(kind, parsed)
  const instanceId = generateInstanceId(manifest.id, new Set(existing.map((e) => e.instanceId)))
  const next = cloneParsed(parsed)

  if (kind === 'reports') {
    // Reports are single-instance — palette filters duplicates upstream,
    // but guard here too.
    const reports = ensurePath(next, 'reports', () => ({}))
    const sections = Array.isArray(reports.sections) ? [...(reports.sections as string[])] : []
    if (!sections.includes(manifest.id)) sections.push(manifest.id)
    reports.sections = sections
    return { next, instanceId: manifest.id }
  }

  if (kind === 'reward') {
    return { next: setRewardPlugin(next, manifest), instanceId: manifest.id }
  }

  if (kind === 'evaluation') {
    const evalCfg = ensurePath(next, 'evaluation', () => ({}))
    if (evalCfg.enabled === undefined) evalCfg.enabled = true
    const evaluators = ensurePath(evalCfg, 'evaluators', () => ({}))
    const list = Array.isArray(evaluators.plugins) ? [...(evaluators.plugins as unknown[])] : []
    list.push({
      id: instanceId,
      plugin: manifest.id,
      enabled: true,
      save_report: true,
      params: manifest.suggested_params ?? {},
      thresholds: manifest.suggested_thresholds ?? {},
    })
    evaluators.plugins = list
    return { next, instanceId }
  }

  // validation
  const datasets = ensurePath(next, 'datasets', () => ({}))
  const key = firstDatasetKey(next)
  if (!isRecord(datasets[key])) {
    datasets[key] = { source_type: 'local', source_local: { local_paths: { train: '' } } }
  }
  const ds = datasets[key] as Record<string, unknown>
  const validations = ensurePath(ds, 'validations', () => ({}))
  const list = Array.isArray(validations.plugins) ? [...(validations.plugins as unknown[])] : []
  list.push({
    id: instanceId,
    plugin: manifest.id,
    params: manifest.suggested_params ?? {},
    thresholds: manifest.suggested_thresholds ?? {},
  })
  validations.plugins = list
  return { next, instanceId }
}

/** Remove an instance by ``instanceId``. */
export function removeInstance(
  kind: PluginKind,
  parsed: Record<string, unknown>,
  instanceId: string,
): Record<string, unknown> {
  const next = cloneParsed(parsed)

  if (kind === 'reports') {
    const reports = isRecord(next.reports) ? next.reports : {}
    const sections = Array.isArray(reports.sections) ? (reports.sections as string[]) : []
    reports.sections = sections.filter((s) => s !== instanceId)
    next.reports = reports
    return next
  }

  if (kind === 'reward') {
    const training = isRecord(next.training) ? next.training : {}
    const strategies = Array.isArray(training.strategies) ? (training.strategies as unknown[]) : []
    training.strategies = strategies.map((s) => {
      if (!isRecord(s)) return s
      if (!isRecord(s.params)) return s
      if (s.params.reward_plugin !== instanceId) return s
      const nextParams = { ...s.params }
      delete nextParams.reward_plugin
      return { ...s, params: nextParams }
    })
    next.training = training
    return next
  }

  if (kind === 'evaluation') {
    const evalCfg = isRecord(next.evaluation) ? next.evaluation : {}
    const evaluators = isRecord(evalCfg.evaluators) ? evalCfg.evaluators : {}
    const list = Array.isArray(evaluators.plugins) ? (evaluators.plugins as unknown[]) : []
    evaluators.plugins = list.filter((e) => !(isRecord(e) && e.id === instanceId))
    evalCfg.evaluators = evaluators
    next.evaluation = evalCfg
    return next
  }

  // validation
  const datasets = isRecord(next.datasets) ? next.datasets : {}
  const key = firstDatasetKey(next)
  const ds = isRecord(datasets[key]) ? (datasets[key] as Record<string, unknown>) : {}
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const list = Array.isArray(validations.plugins) ? (validations.plugins as unknown[]) : []
  validations.plugins = list.filter((e) => !(isRecord(e) && e.id === instanceId))
  return next
}

/** Reorder instances inside one kind. ``orderedIds`` is the new full
 *  order by ``instanceId``. Missing ids are dropped (should not happen
 *  unless the UI got out of sync). */
export function reorderInstances(
  kind: PluginKind,
  parsed: Record<string, unknown>,
  orderedIds: string[],
): Record<string, unknown> {
  const next = cloneParsed(parsed)

  if (kind === 'reports') {
    const reports = ensurePath(next, 'reports', () => ({}))
    // orderedIds are the plugin ids themselves (reports single-instance).
    reports.sections = orderedIds
    return next
  }

  if (kind === 'reward') {
    // Reward has no meaningful reorder — ignore. Caller should avoid
    // wiring a sortable context for reward.
    return next
  }

  if (kind === 'evaluation') {
    const evalCfg = isRecord(next.evaluation) ? next.evaluation : {}
    const evaluators = isRecord(evalCfg.evaluators) ? evalCfg.evaluators : {}
    const list = Array.isArray(evaluators.plugins) ? (evaluators.plugins as unknown[]) : []
    evaluators.plugins = sortByIds(list, orderedIds)
    evalCfg.evaluators = evaluators
    next.evaluation = evalCfg
    return next
  }

  // validation
  const datasets = ensurePath(next, 'datasets', () => ({}))
  const key = firstDatasetKey(next)
  if (!isRecord(datasets[key])) return next
  const ds = datasets[key] as Record<string, unknown>
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const list = Array.isArray(validations.plugins) ? (validations.plugins as unknown[]) : []
  validations.plugins = sortByIds(list, orderedIds)
  ds.validations = validations
  return next
}

function sortByIds(list: unknown[], orderedIds: string[]): unknown[] {
  const byId = new Map<string, unknown>()
  for (const e of list) {
    if (isRecord(e) && typeof e.id === 'string') byId.set(e.id, e)
  }
  const out: unknown[] = []
  for (const id of orderedIds) {
    const e = byId.get(id)
    if (e !== undefined) {
      out.push(e)
      byId.delete(id)
    }
  }
  // Append any items not in orderedIds (defensive — shouldn't happen).
  for (const e of byId.values()) out.push(e)
  return out
}

function setRewardPlugin(
  parsed: Record<string, unknown>,
  manifest: PluginManifest,
): Record<string, unknown> {
  const training = ensurePath(parsed, 'training', () => ({}))
  const strategies = Array.isArray(training.strategies) ? [...(training.strategies as unknown[])] : []
  training.strategies = strategies.map((s) => {
    if (!isRecord(s)) return s
    const t = typeof s.strategy_type === 'string' ? s.strategy_type.toLowerCase() : ''
    if (!['grpo', 'sapo', 'dpo', 'orpo'].includes(t)) return s
    // Only write if the plugin's own supported_strategies matches this
    // strategy type. Backend cross-validator would reject a mismatch
    // later anyway; matching here keeps the UI from silently writing a
    // broken assignment.
    const ok = !manifest.supported_strategies
      || manifest.supported_strategies.length === 0
      || manifest.supported_strategies.map((x) => x.toLowerCase()).includes(t)
    if (!ok) return s
    const params = isRecord(s.params) ? { ...s.params } : {}
    params.reward_plugin = manifest.id
    for (const [k, v] of Object.entries(manifest.suggested_params ?? {})) {
      if (params[k] === undefined) params[k] = v
    }
    return { ...s, params }
  })
  return parsed
}

/** Generate an instance id that doesn't collide with ``taken``.
 *  First tries the bare plugin id; on collision appends ``_2``, ``_3``, etc. */
export function generateInstanceId(pluginId: string, taken: Set<string>): string {
  if (!taken.has(pluginId)) return pluginId
  let n = 2
  while (taken.has(`${pluginId}_${n}`)) n += 1
  return `${pluginId}_${n}`
}

// ---------------------------------------------------------------------------
// Per-instance params / thresholds / instance-level settings
// ---------------------------------------------------------------------------

/** Flattened view of the per-instance settings the Configure modal
 *  needs. Which of the optional fields are meaningful depends on ``kind``. */
export interface PluginInstanceDetails {
  /** Unique id within its kind's list (same as reports plugin id). */
  instanceId: string
  /** Catalog plugin id this instance references. */
  pluginId: string
  /** Free-form params dict — edited through ``FieldRenderer`` against
   *  the plugin's ``params_schema``. */
  params: Record<string, unknown>
  /** Free-form thresholds dict — same mechanism with ``thresholds_schema``. */
  thresholds: Record<string, unknown>
  /** kind=validation: which dataset phases to apply to (train / eval). */
  apply_to?: string[]
  /** kind=validation|evaluation: whether this instance runs on launch. */
  enabled?: boolean
  /** kind=validation: whether a failing validation aborts the run. */
  fail_on_error?: boolean
  /** kind=evaluation: whether to persist the plugin's per-sample report. */
  save_report?: boolean
}

function findEvaluationEntry(
  parsed: Record<string, unknown>,
  instanceId: string,
): { list: unknown[]; idx: number } | null {
  const evalCfg = isRecord(parsed.evaluation) ? parsed.evaluation : {}
  const evaluators = isRecord(evalCfg.evaluators) ? evalCfg.evaluators : {}
  const list = Array.isArray(evaluators.plugins) ? [...(evaluators.plugins as unknown[])] : []
  const idx = list.findIndex((e) => isRecord(e) && e.id === instanceId)
  return idx >= 0 ? { list, idx } : null
}

function findValidationEntry(
  parsed: Record<string, unknown>,
  instanceId: string,
): { datasetKey: string; list: unknown[]; idx: number } | null {
  const datasets = isRecord(parsed.datasets) ? parsed.datasets : {}
  const key = firstDatasetKey(parsed)
  const ds = isRecord(datasets[key]) ? (datasets[key] as Record<string, unknown>) : {}
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const list = Array.isArray(validations.plugins) ? [...(validations.plugins as unknown[])] : []
  const idx = list.findIndex((e) => isRecord(e) && e.id === instanceId)
  return idx >= 0 ? { datasetKey: key, list, idx } : null
}

/** Read the full per-instance details for the Configure modal. Returns
 *  ``null`` when the instance can't be located (e.g. racing with a
 *  concurrent edit). */
export function readInstanceDetails(
  kind: PluginKind,
  parsed: Record<string, unknown>,
  instanceId: string,
): PluginInstanceDetails | null {
  if (kind === 'reports') {
    const instances = readInstances('reports', parsed)
    const hit = instances.find((i) => i.instanceId === instanceId)
    return hit ? { ...hit, params: {}, thresholds: {} } : null
  }

  if (kind === 'reward') {
    // Reward params live on every matching strategy; we pick the first
    // phase that references this plugin and use its params (minus the
    // reward_plugin marker itself). The modal writes back to all.
    const training = isRecord(parsed.training) ? parsed.training : {}
    const strategies = Array.isArray(training.strategies) ? (training.strategies as unknown[]) : []
    for (const s of strategies) {
      if (!isRecord(s)) continue
      const p = isRecord(s.params) ? s.params : {}
      if (p.reward_plugin !== instanceId) continue
      const params: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(p)) {
        if (k !== 'reward_plugin') params[k] = v
      }
      return { instanceId, pluginId: instanceId, params, thresholds: {} }
    }
    return null
  }

  if (kind === 'evaluation') {
    const hit = findEvaluationEntry(parsed, instanceId)
    if (!hit) return null
    const entry = hit.list[hit.idx] as Record<string, unknown>
    return {
      instanceId,
      pluginId: typeof entry.plugin === 'string' ? entry.plugin : '',
      params: isRecord(entry.params) ? { ...entry.params } : {},
      thresholds: isRecord(entry.thresholds) ? { ...entry.thresholds } : {},
      enabled: typeof entry.enabled === 'boolean' ? entry.enabled : true,
      save_report: typeof entry.save_report === 'boolean' ? entry.save_report : true,
    }
  }

  // validation
  const hit = findValidationEntry(parsed, instanceId)
  if (!hit) return null
  const entry = hit.list[hit.idx] as Record<string, unknown>
  return {
    instanceId,
    pluginId: typeof entry.plugin === 'string' ? entry.plugin : '',
    params: isRecord(entry.params) ? { ...entry.params } : {},
    thresholds: isRecord(entry.thresholds) ? { ...entry.thresholds } : {},
    apply_to: Array.isArray(entry.apply_to)
      ? (entry.apply_to.filter((x) => typeof x === 'string') as string[])
      : undefined,
    enabled: typeof entry.enabled === 'boolean' ? entry.enabled : true,
    fail_on_error: typeof entry.fail_on_error === 'boolean' ? entry.fail_on_error : undefined,
  }
}

/** Write instance details back. Instance-level fields are only written
 *  when the provided value differs from ``undefined`` — callers that
 *  don't edit (say) ``save_report`` can omit it.
 *
 *  For reward, params are mirrored to every phase that already
 *  references this reward plugin (we don't create new strategy phases;
 *  if the user wants a phase added they do it in the strategies
 *  section of the config builder). */
export function writeInstanceDetails(
  kind: PluginKind,
  parsed: Record<string, unknown>,
  details: PluginInstanceDetails,
): Record<string, unknown> {
  const next = cloneParsed(parsed)

  if (kind === 'reports') return next // reports carry no per-instance state

  if (kind === 'reward') {
    const training = ensurePath(next, 'training', () => ({}))
    const strategies = Array.isArray(training.strategies)
      ? [...(training.strategies as unknown[])]
      : []
    training.strategies = strategies.map((s) => {
      if (!isRecord(s)) return s
      const p = isRecord(s.params) ? s.params : {}
      if (p.reward_plugin !== details.instanceId) return s
      const mergedParams: Record<string, unknown> = {
        ...details.params,
        reward_plugin: details.instanceId,
      }
      return { ...s, params: mergedParams }
    })
    return next
  }

  if (kind === 'evaluation') {
    const evalCfg = ensurePath(next, 'evaluation', () => ({}))
    const evaluators = ensurePath(evalCfg, 'evaluators', () => ({}))
    const list = Array.isArray(evaluators.plugins) ? [...(evaluators.plugins as unknown[])] : []
    const idx = list.findIndex((e) => isRecord(e) && e.id === details.instanceId)
    if (idx < 0) return next
    const prev = list[idx] as Record<string, unknown>
    list[idx] = {
      ...prev,
      id: details.instanceId,
      plugin: details.pluginId,
      params: details.params,
      thresholds: details.thresholds,
      ...(details.enabled !== undefined && { enabled: details.enabled }),
      ...(details.save_report !== undefined && { save_report: details.save_report }),
    }
    evaluators.plugins = list
    return next
  }

  // validation
  const datasets = ensurePath(next, 'datasets', () => ({}))
  const key = firstDatasetKey(next)
  if (!isRecord(datasets[key])) return next
  const ds = datasets[key] as Record<string, unknown>
  const validations = ensurePath(ds, 'validations', () => ({}))
  const list = Array.isArray(validations.plugins) ? [...(validations.plugins as unknown[])] : []
  const idx = list.findIndex((e) => isRecord(e) && e.id === details.instanceId)
  if (idx < 0) return next
  const prev = list[idx] as Record<string, unknown>
  list[idx] = {
    ...prev,
    id: details.instanceId,
    plugin: details.pluginId,
    params: details.params,
    thresholds: details.thresholds,
    ...(details.apply_to !== undefined && { apply_to: details.apply_to }),
    ...(details.enabled !== undefined && { enabled: details.enabled }),
    ...(details.fail_on_error !== undefined && { fail_on_error: details.fail_on_error }),
  }
  validations.plugins = list
  return next
}

/** Rename an instance (change its ``id`` while keeping the reference
 *  to the catalog plugin). No-op for reports (instanceId === pluginId).
 *  Returns ``null`` if the target id is already taken in the same list. */
export function renameInstance(
  kind: PluginKind,
  parsed: Record<string, unknown>,
  oldId: string,
  newId: string,
): Record<string, unknown> | null {
  if (oldId === newId) return parsed
  if (kind === 'reports') return parsed
  const existing = readInstances(kind, parsed).map((x) => x.instanceId)
  if (existing.includes(newId)) return null

  const next = cloneParsed(parsed)

  if (kind === 'reward') {
    const training = ensurePath(next, 'training', () => ({}))
    const strategies = Array.isArray(training.strategies)
      ? [...(training.strategies as unknown[])]
      : []
    training.strategies = strategies.map((s) => {
      if (!isRecord(s)) return s
      const p = isRecord(s.params) ? { ...s.params } : {}
      if (p.reward_plugin === oldId) p.reward_plugin = newId
      return { ...s, params: p }
    })
    return next
  }

  if (kind === 'evaluation') {
    const evalCfg = ensurePath(next, 'evaluation', () => ({}))
    const evaluators = ensurePath(evalCfg, 'evaluators', () => ({}))
    const list = Array.isArray(evaluators.plugins) ? [...(evaluators.plugins as unknown[])] : []
    const idx = list.findIndex((e) => isRecord(e) && e.id === oldId)
    if (idx < 0) return null
    const prev = list[idx] as Record<string, unknown>
    list[idx] = { ...prev, id: newId }
    evaluators.plugins = list
    return next
  }

  // validation
  const datasets = ensurePath(next, 'datasets', () => ({}))
  const key = firstDatasetKey(next)
  if (!isRecord(datasets[key])) return null
  const ds = datasets[key] as Record<string, unknown>
  const validations = ensurePath(ds, 'validations', () => ({}))
  const list = Array.isArray(validations.plugins) ? [...(validations.plugins as unknown[])] : []
  const idx = list.findIndex((e) => isRecord(e) && e.id === oldId)
  if (idx < 0) return null
  const prev = list[idx] as Record<string, unknown>
  list[idx] = { ...prev, id: newId }
  validations.plugins = list
  return next
}
