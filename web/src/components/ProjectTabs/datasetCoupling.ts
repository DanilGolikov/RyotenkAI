/**
 * Strategy → dataset auto-coupling (one-way only).
 *
 * Invoked from ConfigTab after every form change: when a NEW strategy
 * phase is added without an explicit ``dataset`` field, we generate a
 * unique key (``<strategy_type>_dataset[_N]``) and seed an
 * ``auto_created`` ``datasets.<key>`` block so the user is never left
 * staring at a strategy with no data source.
 *
 * **Removal is intentionally NOT cascading.** When the user deletes a
 * strategy, the dataset entry is preserved — datasets are an
 * independent first-class object inside the project, with their own
 * preview / validation / file paths the user may still want to keep.
 * Cleanup is the user's call, made explicitly through the Datasets
 * tab's Delete button.
 *
 * Manual edits (user picks an existing dataset via the selector or
 * renames ``strategy.dataset``) are pass-through — we never clobber a
 * dataset key the user deliberately typed.
 *
 * The helpers are pure — call site (ConfigTab) is responsible for
 * round-tripping through YAML + save.
 */

import { isRecord } from './pluginInstances'

interface PhaseSummary {
  strategy_type: string
  dataset: string | null
}

/** Return a fresh config with coupling applied. `prev` is the last
 *  parsed config the UI saw; `next` is the one about to be persisted. */
export function syncStrategyDatasetCoupling(
  prev: Record<string, unknown>,
  next: Record<string, unknown>,
): Record<string, unknown> {
  const out = structuredClone(next) as Record<string, unknown>

  const prevPhases = extractPhases(prev)
  const nextPhases = extractPhases(out)

  // Pair new phases with auto-generated dataset keys + scaffold.
  // No cascade-delete: removed strategies leave their dataset behind,
  // and the user can clean it up explicitly from the Datasets tab.
  const existingKeys = new Set(datasetKeysOf(out))
  const strategiesArr = ensureStrategiesArray(out)
  for (let idx = 0; idx < nextPhases.length; idx += 1) {
    const phase = nextPhases[idx]
    if (phase.dataset) continue
    if (!phase.strategy_type) continue
    const key = deriveDatasetKey(phase.strategy_type, existingKeys)
    existingKeys.add(key)
    const phaseDict = strategiesArr[idx] as Record<string, unknown>
    phaseDict.dataset = key
    ensureDatasetScaffold(out, key, phase.strategy_type)
    nextPhases[idx] = { strategy_type: phase.strategy_type, dataset: key }
  }

  // `prevPhases` retained for symmetry with the previous diff-based
  // signature and for any future opt-in cleanup; intentionally unused
  // now that removal is no-op.
  void prevPhases

  return out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function extractPhases(parsed: Record<string, unknown>): PhaseSummary[] {
  const training = isRecord(parsed.training) ? parsed.training : {}
  const strategies = Array.isArray(training.strategies) ? (training.strategies as unknown[]) : []
  return strategies.map((raw) => {
    if (!isRecord(raw)) return { strategy_type: '', dataset: null }
    const st = typeof raw.strategy_type === 'string' ? raw.strategy_type : ''
    const ds = typeof raw.dataset === 'string' && raw.dataset ? raw.dataset : null
    return { strategy_type: st, dataset: ds }
  })
}

function ensureStrategiesArray(parsed: Record<string, unknown>): unknown[] {
  const training = (isRecord(parsed.training) ? parsed.training : {}) as Record<string, unknown>
  if (!Array.isArray(training.strategies)) training.strategies = []
  parsed.training = training
  return training.strategies as unknown[]
}

function datasetKeysOf(parsed: Record<string, unknown>): string[] {
  const block = getDatasetsBlock(parsed)
  return Object.keys(block)
}

function getDatasetsBlock(parsed: Record<string, unknown>): Record<string, unknown> {
  if (!isRecord(parsed.datasets)) parsed.datasets = {}
  return parsed.datasets as Record<string, unknown>
}

/** Deterministic key choice — start with ``<strategy_type>_dataset``,
 *  add ``_2`` / ``_3`` on collision. Good enough for the typical case
 *  of 1–5 strategies per project. */
export function deriveDatasetKey(strategyType: string, existing: Set<string>): string {
  const base = `${strategyType}_dataset`
  if (!existing.has(base)) return base
  let n = 2
  while (existing.has(`${base}_${n}`)) n += 1
  return `${base}_${n}`
}

function ensureDatasetScaffold(
  parsed: Record<string, unknown>,
  key: string,
  strategyType: string,
): void {
  const block = getDatasetsBlock(parsed)
  if (isRecord(block[key])) return
  // DatasetConfig is StrictBaseModel (extra='forbid') — any synthetic
  // UI-only key triggers a 422 on save. The strategy type is already
  // derivable from the key (`<strategy_type>_dataset[_N]`), so we
  // don't need to stash it. Default path follows the convention
  // documented in the plan: `<strategy_type>/datasets/<split>.jsonl`
  // anchored at the project root.
  block[key] = {
    source_type: 'local',
    source_local: {
      local_paths: { train: `${strategyType}/datasets/train.jsonl` },
    },
    auto_created: true,
  }
}

