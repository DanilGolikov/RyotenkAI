/**
 * Dataset-scoped validation plugin instance helpers.
 *
 * Validation plugins live under `datasets.<key>.validations.plugins[]`
 * — strictly per-dataset. The old `pluginInstances.ts` wraps them
 * behind a `firstDatasetKey()` shortcut (the Plugins tab supported
 * only a single active dataset); the Datasets tab needs the full
 * per-key view, so these helpers take `datasetKey` explicitly.
 *
 * Pure functions, same contract as pluginInstances.ts: take parsed
 * config, return a new parsed config (structuredClone'd).
 */

import type { PluginManifest } from '../../api/types'
import { generateInstanceId, isRecord, type PluginInstanceDetails } from '../ProjectTabs/pluginInstances'

export interface ValidationInstance {
  instanceId: string
  pluginId: string
  enabled?: boolean
  /** Apply-to split list ("train"/"eval"). Undefined means default
   *  (both) per backend semantics. */
  applyTo?: string[]
  failOnError?: boolean
}

function getDatasetBlock(
  parsed: Record<string, unknown>,
  datasetKey: string,
): Record<string, unknown> | null {
  const datasets = isRecord(parsed.datasets) ? parsed.datasets : {}
  const ds = datasets[datasetKey]
  return isRecord(ds) ? ds : null
}

function ensureDatasetBlock(
  parsed: Record<string, unknown>,
  datasetKey: string,
): Record<string, unknown> {
  const datasets = isRecord(parsed.datasets) ? parsed.datasets : {}
  if (!isRecord(datasets[datasetKey])) {
    datasets[datasetKey] = { source_type: 'local', source_local: { local_paths: { train: '' } } }
  }
  parsed.datasets = datasets
  return datasets[datasetKey] as Record<string, unknown>
}

export function readValidationInstancesFor(
  parsed: Record<string, unknown>,
  datasetKey: string,
): ValidationInstance[] {
  const ds = getDatasetBlock(parsed, datasetKey)
  if (!ds) return []
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const plugins = Array.isArray(validations.plugins) ? (validations.plugins as unknown[]) : []
  return plugins
    .filter(isRecord)
    .map((p) => ({
      instanceId: typeof p.id === 'string' ? p.id : String(p.plugin ?? ''),
      pluginId: typeof p.plugin === 'string' ? p.plugin : '',
      enabled: typeof p.enabled === 'boolean' ? p.enabled : undefined,
      applyTo: Array.isArray(p.apply_to)
        ? (p.apply_to as unknown[]).filter((v): v is string => typeof v === 'string')
        : undefined,
      failOnError: typeof p.fail_on_error === 'boolean' ? p.fail_on_error : undefined,
    }))
    .filter((x) => x.pluginId)
}

export function addValidationInstanceFor(
  parsed: Record<string, unknown>,
  manifest: PluginManifest,
  datasetKey: string,
): { next: Record<string, unknown>; instanceId: string } {
  const existing = readValidationInstancesFor(parsed, datasetKey)
  const instanceId = generateInstanceId(
    manifest.id,
    new Set(existing.map((e) => e.instanceId)),
  )
  const next = structuredClone(parsed) as Record<string, unknown>
  const ds = ensureDatasetBlock(next, datasetKey)
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const list = Array.isArray(validations.plugins) ? [...(validations.plugins as unknown[])] : []
  list.push({
    id: instanceId,
    plugin: manifest.id,
    params: manifest.suggested_params ?? {},
    thresholds: manifest.suggested_thresholds ?? {},
  })
  validations.plugins = list
  ds.validations = validations
  return { next, instanceId }
}

export function removeValidationInstanceFor(
  parsed: Record<string, unknown>,
  instanceId: string,
  datasetKey: string,
): Record<string, unknown> {
  const next = structuredClone(parsed) as Record<string, unknown>
  const ds = getDatasetBlock(next, datasetKey)
  if (!ds) return next
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const list = Array.isArray(validations.plugins) ? (validations.plugins as unknown[]) : []
  validations.plugins = list.filter((e) => !(isRecord(e) && e.id === instanceId))
  ds.validations = validations
  return next
}

export function reorderValidationInstancesFor(
  parsed: Record<string, unknown>,
  orderedIds: string[],
  datasetKey: string,
): Record<string, unknown> {
  const next = structuredClone(parsed) as Record<string, unknown>
  const ds = getDatasetBlock(next, datasetKey)
  if (!ds) return next
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const list = Array.isArray(validations.plugins) ? [...(validations.plugins as unknown[])] : []
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
  // Any leftovers (shouldn't happen in normal DnD flow) are appended
  // preserving their relative order from the original list.
  for (const e of byId.values()) out.push(e)
  validations.plugins = out
  ds.validations = validations
  return next
}

export function readValidationInstanceDetailsFor(
  parsed: Record<string, unknown>,
  instanceId: string,
  datasetKey: string,
): PluginInstanceDetails | null {
  const ds = getDatasetBlock(parsed, datasetKey)
  if (!ds) return null
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const list = Array.isArray(validations.plugins) ? (validations.plugins as unknown[]) : []
  const entry = list.find((e) => isRecord(e) && e.id === instanceId)
  if (!isRecord(entry)) return null
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

export function writeValidationInstanceDetailsFor(
  parsed: Record<string, unknown>,
  details: PluginInstanceDetails,
  datasetKey: string,
): Record<string, unknown> {
  const next = structuredClone(parsed) as Record<string, unknown>
  const ds = ensureDatasetBlock(next, datasetKey)
  const validations = isRecord(ds.validations) ? ds.validations : {}
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
  ds.validations = validations
  return next
}

export function renameValidationInstanceFor(
  parsed: Record<string, unknown>,
  oldId: string,
  newId: string,
  datasetKey: string,
): Record<string, unknown> | null {
  if (oldId === newId) return parsed
  const existing = readValidationInstancesFor(parsed, datasetKey).map((x) => x.instanceId)
  if (existing.includes(newId)) return null
  const next = structuredClone(parsed) as Record<string, unknown>
  const ds = getDatasetBlock(next, datasetKey)
  if (!ds) return null
  const validations = isRecord(ds.validations) ? ds.validations : {}
  const list = Array.isArray(validations.plugins) ? [...(validations.plugins as unknown[])] : []
  const idx = list.findIndex((e) => isRecord(e) && e.id === oldId)
  if (idx < 0) return null
  const prev = list[idx] as Record<string, unknown>
  list[idx] = { ...prev, id: newId }
  validations.plugins = list
  ds.validations = validations
  return next
}
