/**
 * Right-pane view for a single selected dataset.
 *
 *   header  — key + source-type chip + auto/manual tag + Delete button
 *   source  — DatasetSourceFields (local vs HF, paths, path-check)
 *   preview — paginated rows (DatasetPreviewPane)
 *   validation — run + render per-plugin results (ValidationPanel)
 *
 * Edits are dirty-tracked locally — the Save button writes the whole
 * parsed config back via `persist()`. No sync with ConfigTab's dirty
 * state (known Phase A limitation — documented in the plan).
 */

import { useMemo, useState } from 'react'
import { useDatasetValidation } from '../../api/hooks/useDatasets'
import type { DatasetEntry } from '../../api/hooks/useDatasets'
import type { DatasetValidateResponse } from '../../api/types'
import { DatasetPreviewPane } from './DatasetPreviewPane'
import type { SourcePatch } from './DatasetSourceFields'
import { DatasetSourceFields } from './DatasetSourceFields'
import { ValidationPanel } from './ValidationPanel'
import { ValidationPluginsSection } from './ValidationPluginsSection'

interface Props {
  projectId: string
  parsed: Record<string, unknown>
  entry: DatasetEntry
  persist: (next: Record<string, unknown>) => Promise<void>
  saving: boolean
  saveError: Error | null
  onDeleted: () => void
}

export function DatasetDetail({ projectId, parsed, entry, persist, saving, saveError, onDeleted }: Props) {
  const [draft, setDraft] = useState<DatasetEntry>(entry)
  const dirty = useMemo(() => !sameEntry(draft, entry), [draft, entry])
  const validation = useDatasetValidation(projectId, entry.key)
  const lastResult: DatasetValidateResponse | null = validation.data ?? null

  // Map globalIdx → list of plugin ids that flagged it.
  const badRows = useMemo(() => {
    const map = new Map<number, string[]>()
    if (!lastResult) return map
    for (const run of lastResult.plugin_results) {
      if (run.passed || run.crashed) continue
      for (const g of run.error_groups ?? []) {
        for (const idx of g.sample_indices) {
          const prev = map.get(idx) ?? []
          prev.push(run.plugin_id)
          map.set(idx, prev)
        }
      }
    }
    return map
  }, [lastResult])

  const applyPatch = (patch: SourcePatch) => {
    setDraft((prev) => ({
      ...prev,
      sourceType: patch.sourceType ?? prev.sourceType,
      trainPath: patch.trainPath !== undefined ? patch.trainPath : prev.trainPath,
      evalPath: patch.evalPath !== undefined ? patch.evalPath : prev.evalPath,
      hasEvalSplit: patch.hasEvalSplit !== undefined ? patch.hasEvalSplit : prev.hasEvalSplit,
    }))
  }

  const save = async () => {
    const next = writeDatasetBackIntoConfig(parsed, draft)
    await persist(next)
  }

  const reset = () => setDraft(entry)

  const deleteDataset = async () => {
    const confirmMsg = draft.autoCreated
      ? `Remove "${entry.key}"?\n\nThe strategy that auto-created it will end up without a dataset; you'll need to pick another one in the Config tab.`
      : `Remove "${entry.key}"?\n\nThe YAML entry will be deleted.`
    if (!window.confirm(confirmMsg)) return
    const next = deleteDatasetFromConfig(parsed, entry.key)
    await persist(next)
    onDeleted()
  }

  const runValidation = (full: boolean) =>
    validation.mutate({ split: 'train', max_samples: full ? null : 1000 })

  return (
    <div className="space-y-6 max-w-4xl pb-12">
      <header className="pb-3 border-b border-line-1 flex items-start gap-3 flex-wrap">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h2 className="text-[1.15rem] font-semibold text-ink-1 tracking-tight leading-tight truncate">
              <span className="font-mono">{entry.key}</span>
            </h2>
            <span className={`pill ${entry.sourceType === 'huggingface' ? 'pill-info' : 'pill-idle'}`}>
              {entry.sourceType === 'huggingface' ? 'HF' : 'local'}
            </span>
            {entry.autoCreated && (
              <span className="pill pill-skip" title="Created together with a strategy">
                auto
              </span>
            )}
          </div>
          <div className="mt-1 text-xs text-ink-3">
            {entry.sourceType === 'huggingface' ? 'HuggingFace Hub dataset' : 'Local JSON Lines file'} —
            used by every strategy whose <span className="font-mono text-ink-2">dataset</span> field equals{' '}
            <span className="font-mono text-ink-2">{entry.key}</span>.
          </div>
        </div>
        <button
          type="button"
          onClick={deleteDataset}
          disabled={saving}
          className="rounded-md border border-err/40 px-3 py-1.5 text-2xs text-err hover:bg-err/10 hover:border-err transition disabled:opacity-50"
        >
          Delete
        </button>
      </header>

      <section className="space-y-2">
        <DatasetSourceFields projectId={projectId} entry={draft} onChange={applyPatch} />
        <div className="flex items-center gap-2 pt-3 border-t border-line-1/50">
          <button
            type="button"
            onClick={save}
            disabled={!dirty || saving}
            className="btn-primary px-3 py-1.5 text-xs disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
          <button
            type="button"
            onClick={reset}
            disabled={!dirty || saving}
            className="rounded-md border border-line-1 px-3 py-1.5 text-2xs text-ink-2 hover:text-ink-1 hover:border-line-2 transition disabled:opacity-50"
          >
            Reset
          </button>
          {saveError && <span className="text-2xs text-err">{saveError.message}</span>}
        </div>
      </section>

      <ValidationPluginsSection
        parsed={parsed}
        datasetKey={entry.key}
        projectId={projectId}
        persist={persist}
        saving={saving}
      />

      <DatasetPreviewPane projectId={projectId} entry={entry} badRowIndices={badRows} />

      {/* Validation triggers + results sit DIRECTLY UNDER the preview
          pane (which is itself self-scrolling, capped at ~55vh). This
          keeps the buttons reachable without forcing the user past
          the whole preview, and the results render inline below so
          flagged-row indices line up with rows visible right above. */}
      <section className="space-y-2 pt-3 border-t border-line-1/50">
        <div className="flex items-center gap-2 flex-wrap">
          <button
            type="button"
            onClick={() => runValidation(false)}
            disabled={validation.isPending}
            className="btn-primary px-3 py-1.5 text-xs disabled:opacity-50"
            title="Validate against a 1000-row sample"
          >
            {validation.isPending ? 'Validating…' : 'Validate'}
          </button>
          <button
            type="button"
            onClick={() => runValidation(true)}
            disabled={validation.isPending}
            className="rounded-md border border-line-1 px-3 py-1.5 text-2xs text-ink-2 hover:text-ink-1 hover:border-line-2 transition disabled:opacity-50"
            title="Validate against the full dataset (slower)"
          >
            Full dataset
          </button>
          <span className="text-2xs text-ink-4 ml-1">
            sample = 1000 rows · full = entire dataset
          </span>
        </div>
        {(lastResult || validation.isPending || validation.error) && (
          <ValidationPanel
            result={lastResult}
            loading={validation.isPending}
            error={(validation.error as Error | null) ?? null}
          />
        )}
      </section>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Config-rewriting helpers — keep here (not shared) to make it obvious
// that DatasetDetail owns the schema layout it writes.
// ---------------------------------------------------------------------------

function sameEntry(a: DatasetEntry, b: DatasetEntry): boolean {
  return (
    a.key === b.key &&
    a.sourceType === b.sourceType &&
    a.trainPath === b.trainPath &&
    a.evalPath === b.evalPath &&
    a.hasEvalSplit === b.hasEvalSplit &&
    a.autoCreated === b.autoCreated
  )
}

function writeDatasetBackIntoConfig(
  parsed: Record<string, unknown>,
  draft: DatasetEntry,
): Record<string, unknown> {
  const next: Record<string, unknown> = structuredClone(parsed)
  const datasets = (next.datasets ?? {}) as Record<string, unknown>
  const current = (datasets[draft.key] ?? {}) as Record<string, unknown>

  const updated: Record<string, unknown> = { ...current }
  updated.source_type = draft.sourceType
  updated.auto_created = draft.autoCreated

  if (draft.sourceType === 'local') {
    const localPaths: Record<string, unknown> = { train: draft.trainPath }
    if (draft.hasEvalSplit && draft.evalPath) localPaths.eval = draft.evalPath
    updated.source_local = { local_paths: localPaths }
    delete updated.source_hf
  } else {
    const hf: Record<string, unknown> = { train_id: draft.trainPath }
    if (draft.hasEvalSplit && draft.evalPath) hf.eval_id = draft.evalPath
    updated.source_hf = hf
    delete updated.source_local
  }

  datasets[draft.key] = updated
  next.datasets = datasets
  return next
}

function deleteDatasetFromConfig(
  parsed: Record<string, unknown>,
  key: string,
): Record<string, unknown> {
  const next: Record<string, unknown> = structuredClone(parsed)
  const datasets = (next.datasets ?? {}) as Record<string, unknown>
  delete datasets[key]
  next.datasets = datasets
  return next
}
