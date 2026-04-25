/**
 * Custom renderer for the `training.strategies.*.dataset` field.
 *
 * Instead of a free-form text input (the generic schema renderer would
 * produce), we show a dropdown seeded from the keys of
 * `parsed.datasets`, plus a "Configure →" link that deep-links to the
 * Datasets tab detail for the selected key.
 */

import { useParams } from 'react-router-dom'
import { useValidationCtx } from './ValidationContext'
import { SelectField } from './SelectField'

interface Props {
  value: unknown
  onChange: (next: unknown) => void
  onFocus?: () => void
  onBlur?: () => void
  rootValue?: Record<string, unknown>
}

export function DatasetSelectField({ value, onChange, onFocus, onBlur, rootValue }: Props) {
  const { id: projectId } = useParams<{ id: string }>()
  const ctx = useValidationCtx()
  void ctx // reserved for future field-level validation piping

  const datasetKeys = extractDatasetKeys(rootValue)
  const current = typeof value === 'string' ? value : ''
  const options = datasetKeys.map((k) => ({ value: k, label: k }))
  const needsEmpty = !datasetKeys.includes(current)

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <SelectField
        value={current}
        options={options}
        onChange={(next) => onChange(next === '' ? undefined : next)}
        allowEmpty={needsEmpty}
        placeholder={datasetKeys.length === 0 ? '(no datasets configured)' : '—'}
        onFocus={onFocus}
        onBlur={onBlur}
        triggerClassName="min-w-[220px]"
      />
      {projectId && current && datasetKeys.includes(current) && (
        <a
          href={`/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(current)}`}
          className="text-2xs text-ink-3 hover:text-ink-1 underline decoration-dotted"
          title="Jump to dataset detail"
        >
          Configure →
        </a>
      )}
    </div>
  )
}

function extractDatasetKeys(root: Record<string, unknown> | undefined): string[] {
  if (!root || typeof root !== 'object') return []
  const block = (root as Record<string, unknown>).datasets
  if (!block || typeof block !== 'object' || Array.isArray(block)) return []
  return Object.keys(block as Record<string, unknown>)
}
