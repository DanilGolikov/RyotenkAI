/**
 * Top-level validation panel for a single dataset — the "Validate"
 * button, format-check banner, and per-plugin results. Results include
 * structured `error_groups` that the preview pane consumes to paint
 * bad rows.
 */

import type {
  DatasetErrorGroupPayload,
  DatasetFormatCheckPayload,
  DatasetPluginRunPayload,
  DatasetValidateResponse,
} from '../../api/types'
import { Spinner } from '../ui'

interface Props {
  result: DatasetValidateResponse | null
  loading: boolean
  error: Error | null
}

/**
 * Render panel for the last validation run. Trigger buttons live in
 * the DatasetDetail header (outside this component) so the user can
 * kick validation off without scrolling past the preview.
 */
export function ValidationPanel({ result, loading, error }: Props) {
  return (
    <section className="space-y-3">
      {result && (
        <div className="flex items-center justify-end text-2xs text-ink-4">
          finished in {result.duration_ms}ms
        </div>
      )}

      {error && (
        <div className="rounded-md border border-err/40 bg-err/10 px-3 py-2 text-xs text-err">
          {error.message}
        </div>
      )}

      {loading && !result && (
        <div className="flex items-center gap-2 text-xs text-ink-3">
          <Spinner /> validating
        </div>
      )}

      {result && <FormatCheckBanner result={result} />}
      {result?.plugin_results && result.plugin_results.length > 0 && (
        <PluginResults runs={result.plugin_results} />
      )}
    </section>
  )
}

function FormatCheckBanner({ result }: { result: DatasetValidateResponse }) {
  if (result.format_check_error) {
    return (
      <div className="rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn leading-snug">
        <div className="font-medium">Format check skipped</div>
        <div className="text-ink-3 mt-0.5">{result.format_check_error}</div>
      </div>
    )
  }
  const items = result.format_check ?? []
  if (items.length === 0) return null
  const allOk = items.every((r) => r.ok)
  return (
    <div
      className={[
        'rounded-md px-3 py-2 text-xs leading-snug border',
        allOk
          ? 'border-ok/30 bg-ok/10 text-ok'
          : 'border-err/40 bg-err/10 text-err',
      ].join(' ')}
    >
      <div className="font-medium">
        {allOk ? '✓ Format compatible with all strategies' : '✗ Format mismatch'}
      </div>
      <ul className="mt-1 space-y-0.5">
        {items.map((item: DatasetFormatCheckPayload) => (
          <li key={item.strategy_type} className="flex items-start gap-2">
            <span className={item.ok ? 'text-ok' : 'text-err'}>{item.ok ? '✓' : '✗'}</span>
            <span className="font-mono">{item.strategy_type}</span>
            {item.message && <span className="text-ink-3">— {item.message}</span>}
          </li>
        ))}
      </ul>
    </div>
  )
}

function PluginResults({ runs }: { runs: DatasetPluginRunPayload[] }) {
  return (
    <div className="space-y-2">
      <div className="text-2xs text-ink-3 uppercase tracking-wide">Plugins</div>
      {runs.map((run) => (
        <PluginRunRow key={run.plugin_id} run={run} />
      ))}
    </div>
  )
}

function PluginRunRow({ run }: { run: DatasetPluginRunPayload }) {
  const statusClass = run.crashed
    ? 'pill pill-err'
    : run.passed
    ? 'pill pill-ok'
    : 'pill pill-warn'
  const statusLabel = run.crashed ? 'crash' : run.passed ? 'pass' : 'fail'
  const errors = run.errors ?? []
  const groups = run.error_groups ?? []
  const warnings = run.warnings ?? []
  const recs = run.recommendations ?? []
  const hasDetail = errors.length + groups.length + warnings.length + recs.length > 0
  return (
    <div className="rounded-md border border-line-1 bg-surface-2">
      <div className="flex items-center gap-2 px-3 py-2">
        <span className={statusClass}>{statusLabel}</span>
        <span className="text-xs font-mono text-ink-1">{run.plugin_id}</span>
        <span className="text-2xs text-ink-3">{run.plugin_name}</span>
        <span className="ml-auto text-2xs text-ink-4">{run.duration_ms.toFixed(0)}ms</span>
      </div>
      {hasDetail && (
        <div className="px-3 pb-2.5 pt-0 space-y-1.5 text-2xs leading-snug">
          {groups.length > 0 && (
            <ul className="space-y-0.5">
              {groups.map((g: DatasetErrorGroupPayload) => (
                <li key={g.error_type} className="text-err">
                  <span className="font-mono">{g.error_type}</span>{' '}
                  <span className="text-ink-3">
                    — {g.total_count} row{g.total_count === 1 ? '' : 's'}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {errors.length > 0 && (
            <ul className="text-err list-disc list-inside">
              {errors.map((msg, i) => (
                <li key={i}>{msg}</li>
              ))}
            </ul>
          )}
          {warnings.length > 0 && (
            <ul className="text-warn list-disc list-inside">
              {warnings.map((msg, i) => (
                <li key={i}>{msg}</li>
              ))}
            </ul>
          )}
          {recs.length > 0 && (
            <ul className="text-ink-3 list-disc list-inside italic">
              {recs.map((msg, i) => (
                <li key={i}>{msg}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
