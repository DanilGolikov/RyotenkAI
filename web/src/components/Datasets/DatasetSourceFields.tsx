/**
 * Form for editing a single dataset's source config. Mirrors the
 * Linear/Stripe-style label-left + input-right grammar used by
 * LabelledRow / FieldRow / EnvRow elsewhere in the app — labels are
 * plain text, inputs sit on `surface-inset` so they read as one widget
 * family with the rest of the forms.
 *
 * Two source types: local (jsonl paths) and huggingface (repo + split).
 * The toggle is a SelectField so it picks up the same focus-ring and
 * popover treatment as the rest of the config builder.
 */

import { useDatasetPathCheck } from '../../api/hooks/useDatasets'
import type { DatasetEntry } from '../../api/hooks/useDatasets'
import { SelectField } from '../ConfigBuilder/SelectField'
import { Spinner } from '../ui'

interface Props {
  projectId: string
  entry: DatasetEntry
  onChange: (patch: SourcePatch) => void
}

export interface SourcePatch {
  sourceType?: 'local' | 'huggingface'
  trainPath?: string
  evalPath?: string | null
  /** True → write `eval` field into the YAML; false → strip it. */
  hasEvalSplit?: boolean
}

const INPUT_CLS =
  'h-8 rounded-md bg-surface-inset border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-brand hover:border-line-2 transition-colors'

const ROW_CLS =
  'py-1.5 grid grid-cols-1 sm:grid-cols-[200px_minmax(0,1fr)] gap-1 sm:gap-4 items-start sm:items-center'

export function DatasetSourceFields({ projectId, entry, onChange }: Props) {
  const pathCheck = useDatasetPathCheck(projectId, entry.key)
  const isHF = entry.sourceType === 'huggingface'

  return (
    <div className="space-y-1 max-w-3xl">
      <FieldRow label="Source type">
        <SelectField
          value={entry.sourceType}
          options={[
            { value: 'local', label: 'Local file' },
            { value: 'huggingface', label: 'HuggingFace Hub' },
          ]}
          onChange={(next) =>
            onChange({ sourceType: (next as 'local' | 'huggingface') })
          }
          triggerClassName="min-w-[180px]"
        />
      </FieldRow>

      <FieldRow label={isHF ? 'Train repo' : 'Train path'} required>
        <input
          type="text"
          value={entry.trainPath}
          onChange={(e) => onChange({ trainPath: e.target.value })}
          placeholder={isHF ? 'org/dataset-id' : './data/train.jsonl'}
          className={`${INPUT_CLS} w-full max-w-[640px]`}
        />
      </FieldRow>
      <PathStatus
        loading={pathCheck.isFetching}
        result={pathCheck.data?.train}
        kind={isHF ? 'hf' : 'local'}
      />

      <FieldRow label="Has eval split">
        <input
          type="checkbox"
          checked={entry.hasEvalSplit}
          onChange={(e) =>
            onChange({
              hasEvalSplit: e.target.checked,
              evalPath: e.target.checked ? entry.evalPath ?? '' : null,
            })
          }
          className="h-4 w-4 accent-brand"
        />
      </FieldRow>

      {entry.hasEvalSplit && (
        <>
          <FieldRow label={isHF ? 'Eval repo' : 'Eval path'}>
            <input
              type="text"
              value={entry.evalPath ?? ''}
              onChange={(e) => onChange({ evalPath: e.target.value })}
              placeholder={isHF ? 'org/dataset-id' : './data/eval.jsonl'}
              className={`${INPUT_CLS} w-full max-w-[640px]`}
            />
          </FieldRow>
          <PathStatus
            loading={pathCheck.isFetching}
            result={pathCheck.data?.eval ?? undefined}
            kind={isHF ? 'hf' : 'local'}
          />
        </>
      )}
    </div>
  )
}

function FieldRow({
  label,
  required,
  children,
}: {
  label: string
  required?: boolean
  children: React.ReactNode
}) {
  return (
    <div className={ROW_CLS}>
      <div className="flex items-center gap-1.5 min-w-0 h-8 px-0.5">
        <span className="flex-1 min-w-0 text-xs text-ink-2 tracking-tight truncate">
          {label}
          {required && <span aria-hidden className="ml-0.5 text-brand-warm">*</span>}
        </span>
      </div>
      <div className="w-full min-w-0 flex items-center flex-wrap gap-2">{children}</div>
    </div>
  )
}

function PathStatus({
  loading,
  result,
  kind,
}: {
  loading: boolean
  result: { exists: boolean; line_count?: number | null; size_bytes?: number | null; error?: string | null } | undefined
  kind: 'local' | 'hf'
}) {
  if (loading && !result) {
    return (
      <div className="ml-0 sm:ml-[216px] text-2xs text-ink-3 flex items-center gap-1.5">
        <Spinner /> checking…
      </div>
    )
  }
  if (!result) return null

  if (result.exists && !result.error) {
    if (kind === 'local') {
      return (
        <div className="ml-0 sm:ml-[216px] text-2xs text-ok flex items-center gap-2">
          <span>✓ exists</span>
          {result.line_count != null && (
            <span className="text-ink-3">{result.line_count.toLocaleString()} rows</span>
          )}
          {result.size_bytes != null && (
            <span className="text-ink-4">· {formatBytes(result.size_bytes)}</span>
          )}
        </div>
      )
    }
    return (
      <div className="ml-0 sm:ml-[216px] text-2xs text-ok">✓ repo reachable</div>
    )
  }

  if (result.error === 'auth_required') {
    return (
      <div className="ml-0 sm:ml-[216px] text-2xs text-warn">
        ⚠︎ HF token required — set in Settings → Integrations
      </div>
    )
  }
  return (
    <div className="ml-0 sm:ml-[216px] text-2xs text-err">
      ✗ {result.error ?? 'not reachable'}
    </div>
  )
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}
