/**
 * Paginated dataset preview with two view modes:
 *   - "raw"        — one line per row, JSON-encoded, monospace
 *   - "structured" — table with one column per schema_hint key
 *
 * IntersectionObserver on the bottom sentinel triggers the next page
 * from useDatasetPreview's useInfiniteQuery — DOM grows but we cap at
 * the built-in staleTime so stale datasets don't hog memory.
 *
 * Validation results propagate row accents: bad_row_indices is a set
 * computed by the parent from the latest validation result's
 * error_groups.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { DatasetEntry } from '../../api/hooks/useDatasets'
import { useDatasetPreview } from '../../api/hooks/useDatasets'
import type { DatasetSplit } from '../../api/types'
import { Spinner } from '../ui'

interface Props {
  projectId: string
  entry: DatasetEntry
  /** Row indices (global, 0-based) flagged by the last validation run.
   *  UI renders these with an amber accent + tooltip. */
  badRowIndices?: Map<number, string[]>
}

type ViewMode = 'raw' | 'structured'

export function DatasetPreviewPane({ projectId, entry, badRowIndices }: Props) {
  const [split, setSplit] = useState<DatasetSplit>('train')
  const [mode, setMode] = useState<ViewMode>('structured')
  const [fullscreen, setFullscreen] = useState(false)
  const [onlyErrors, setOnlyErrors] = useState(false)
  // Selected row index (global). null → detail panel hidden, table
  // fills full width. Master-detail pattern: cells stay one-line in
  // the table; full content (long strings, message arrays, nested
  // objects) lives in the right-side detail panel where the user
  // actually has room to read it.
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)

  // Eval tab hidden when the user hasn't configured an eval split.
  const canShowEval = entry.hasEvalSplit

  const query = useDatasetPreview(projectId, entry.key, split)

  const rows = useMemo(() => {
    const out: { row: Record<string, unknown>; globalIdx: number }[] = []
    let idx = 0
    for (const page of query.data?.pages ?? []) {
      for (const row of page.rows) {
        out.push({ row, globalIdx: idx })
        idx += 1
      }
    }
    return out
  }, [query.data])

  const schemaHint = query.data?.pages[0]?.schema_hint ?? []
  const total = query.data?.pages[0]?.total ?? null

  const hasBadRows = (badRowIndices?.size ?? 0) > 0
  // `onlyErrors` is only meaningful once a validation result exists.
  // If nothing is flagged, we silently clear the toggle so the UI
  // doesn't show an empty list on subsequent renders.
  const effectiveOnlyErrors = onlyErrors && hasBadRows
  const visibleRows = useMemo(() => {
    if (!effectiveOnlyErrors) return rows
    return rows.filter(({ globalIdx }) => badRowIndices?.has(globalIdx))
  }, [rows, effectiveOnlyErrors, badRowIndices])

  const sentinelRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (!sentinelRef.current) return
    if (!query.hasNextPage || query.isFetchingNextPage) return
    // Disable auto-paging while the user is in "only errors" mode —
    // pages are loaded on demand from the ValidationResultsPanel links
    // instead (the sentinel sits below the filtered view, which may
    // never hit the viewport). Keeps memory bounded.
    if (effectiveOnlyErrors) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          void query.fetchNextPage()
        }
      },
      { rootMargin: '400px 0px' },
    )
    observer.observe(sentinelRef.current)
    return () => observer.disconnect()
  }, [query, effectiveOnlyErrors])

  // Escape closes detail panel first; if no detail open → exits
  // fullscreen. Mirrors VS Code / Notion row-detail UX.
  useEffect(() => {
    if (!fullscreen && selectedIdx === null) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (selectedIdx !== null) setSelectedIdx(null)
      else if (fullscreen) setFullscreen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [fullscreen, selectedIdx])

  const selectedRow = selectedIdx === null
    ? null
    : visibleRows.find((r) => r.globalIdx === selectedIdx) ?? null

  const toolbar = (
    <div className="flex items-center gap-2 text-2xs flex-wrap">
      {canShowEval && (
        <div className="inline-flex rounded-md border border-line-1 overflow-hidden">
          <button
            type="button"
            onClick={() => setSplit('train')}
            className={`px-3 py-1 transition ${
              split === 'train' ? 'bg-surface-2 text-ink-1' : 'text-ink-3 hover:text-ink-1 hover:bg-surface-2/60'
            }`}
          >
            train
          </button>
          <button
            type="button"
            onClick={() => setSplit('eval')}
            className={`px-3 py-1 transition ${
              split === 'eval' ? 'bg-surface-2 text-ink-1' : 'text-ink-3 hover:text-ink-1 hover:bg-surface-2/60'
            }`}
          >
            eval
          </button>
        </div>
      )}
      <div className="inline-flex rounded-md border border-line-1 overflow-hidden">
        <button
          type="button"
          onClick={() => setMode('structured')}
          className={`px-3 py-1 transition ${
            mode === 'structured' ? 'bg-surface-2 text-ink-1' : 'text-ink-3 hover:text-ink-1 hover:bg-surface-2/60'
          }`}
        >
          Structured
        </button>
        <button
          type="button"
          onClick={() => setMode('raw')}
          className={`px-3 py-1 transition ${
            mode === 'raw' ? 'bg-surface-2 text-ink-1' : 'text-ink-3 hover:text-ink-1 hover:bg-surface-2/60'
          }`}
        >
          Raw
        </button>
      </div>
      {hasBadRows && (
        <button
          type="button"
          onClick={() => setOnlyErrors((v) => !v)}
          className={`rounded-md border px-2.5 py-1 transition ${
            effectiveOnlyErrors
              ? 'bg-warn/15 text-warn border-warn/50'
              : 'border-line-1 text-ink-3 hover:text-ink-1 hover:border-line-2'
          }`}
          title="Show only rows flagged by the last validation run"
        >
          {effectiveOnlyErrors ? 'Only errors ✓' : 'Only errors'}
        </button>
      )}
      <button
        type="button"
        onClick={() => setFullscreen((v) => !v)}
        className="rounded-md border border-line-1 px-2.5 py-1 text-ink-3 hover:text-ink-1 hover:border-line-2 transition"
        title={fullscreen ? 'Collapse (Esc)' : 'Open fullscreen'}
      >
        {fullscreen ? '✕ close' : '⤡ expand'}
      </button>
      <span className="ml-auto text-ink-3">
        {visibleRows.length.toLocaleString()}
        {effectiveOnlyErrors && ` / ${rows.length.toLocaleString()}`} of{' '}
        {total != null ? total.toLocaleString() : '…'} rows
      </span>
    </div>
  )

  // Body is a split layout when a row is selected:
  //   left  — table / raw, scrollable in both axes
  //   right — detail panel showing the active row's fields, scrollable
  // When nothing is selected the right panel collapses and the table
  // takes the full width — no wasted space on narrow datasets.
  const bodyHeight = fullscreen
    ? 'flex-1 min-h-0'
    // Bounded height so the surrounding DatasetDetail keeps its
    // controls (Validate, Save) on screen. The user can lift the cap
    // with the ⤡ expand button.
    : 'h-[55vh]'

  const tableScroller = (
    <div className="flex-1 min-w-0 rounded-md border border-line-1 bg-surface-inset overflow-auto">
      {query.isLoading && (
        <div className="flex items-center gap-2 text-xs text-ink-3 p-3">
          <Spinner /> loading
        </div>
      )}
      {query.error && (
        <div className="text-xs text-err p-3">
          {(query.error as Error).message}
        </div>
      )}
      {!query.isLoading && !query.error && rows.length === 0 && (
        <div className="text-xs text-ink-3 italic p-3">(no rows)</div>
      )}
      {effectiveOnlyErrors && rows.length > 0 && visibleRows.length === 0 && (
        <div className="text-xs text-ink-3 italic p-3">
          No flagged rows in the pages loaded so far — scroll the full view to load more, then re-filter.
        </div>
      )}
      {visibleRows.length > 0 && mode === 'structured' && (
        <StructuredRows
          rows={visibleRows}
          schemaHint={schemaHint}
          badRowIndices={badRowIndices}
          selectedIdx={selectedIdx}
          onSelect={setSelectedIdx}
        />
      )}
      {visibleRows.length > 0 && mode === 'raw' && (
        <RawRows
          rows={visibleRows}
          badRowIndices={badRowIndices}
          selectedIdx={selectedIdx}
          onSelect={setSelectedIdx}
        />
      )}
      {query.hasNextPage && !effectiveOnlyErrors && (
        <div
          ref={sentinelRef}
          className="flex items-center justify-center gap-2 text-2xs text-ink-3 py-2 border-t border-line-1/50"
        >
          {query.isFetchingNextPage ? (
            <>
              <Spinner /> loading more
            </>
          ) : (
            <span>scroll for more</span>
          )}
        </div>
      )}
      {!query.hasNextPage && rows.length > 0 && (
        <div className="text-center text-2xs text-ink-4 py-1.5 border-t border-line-1/50">
          end of dataset
        </div>
      )}
    </div>
  )

  const detailPanel = selectedRow ? (
    <RowDetailPanel
      row={selectedRow.row}
      globalIdx={selectedRow.globalIdx}
      flags={badRowIndices?.get(selectedRow.globalIdx)}
      onClose={() => setSelectedIdx(null)}
    />
  ) : null

  const body = (
    <div className={`flex gap-2 ${bodyHeight}`}>
      {tableScroller}
      {detailPanel}
    </div>
  )

  if (fullscreen) {
    // Portal into body so the overlay escapes any overflow:hidden
    // parent. Layout mimics the YAML fullscreen editor: fade-in
    // backdrop + centred panel that spans the viewport.
    return createPortal(
      <div className="fs-backdrop-in fixed inset-0 z-50 bg-black/70 backdrop-blur-sm p-6 flex">
        <div className="fs-enter flex flex-col min-h-0 flex-1 gap-2 bg-surface-1 border border-line-1 rounded-lg p-4 shadow-card">
          <div className="flex items-center gap-2">
            <div className="text-xs text-ink-2 font-mono truncate flex-1 min-w-0">{entry.key}</div>
            {toolbar}
          </div>
          {body}
        </div>
      </div>,
      document.body,
    )
  }

  return (
    <div className="space-y-2">
      {toolbar}
      {body}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Row rendering
// ---------------------------------------------------------------------------

function StructuredRows({
  rows,
  schemaHint,
  badRowIndices,
  selectedIdx,
  onSelect,
}: {
  rows: { row: Record<string, unknown>; globalIdx: number }[]
  schemaHint: string[]
  badRowIndices?: Map<number, string[]>
  selectedIdx: number | null
  onSelect: (idx: number | null) => void
}) {
  const columns = useMemo(() => {
    if (schemaHint.length > 0) return schemaHint
    const seen: Record<string, true> = {}
    for (const { row } of rows.slice(0, 50)) {
      for (const k of Object.keys(row)) if (!k.startsWith('__')) seen[k] = true
    }
    return Object.keys(seen)
  }, [rows, schemaHint])

  return (
    // No inner scroller — the parent body already overflows both ways,
    // and a nested `overflow-x-auto` would steal events from it. Sticky
    // header attaches to the parent's scroll, which is what we want.
    <div>
      <table className="w-full text-[12px] font-mono border-collapse table-fixed">
        <colgroup>
          <col className="w-12" />
          {columns.map((col) => (
            <col key={col} className="w-[220px]" />
          ))}
        </colgroup>
        <thead className="sticky top-0 bg-surface-2 text-ink-3 z-10">
          <tr>
            <th className="text-left px-3 py-1.5 font-medium text-2xs uppercase tracking-wide border-r border-line-1">
              #
            </th>
            {columns.map((col, idx) => (
              <th
                key={col}
                className={`text-left px-3 py-1.5 font-medium text-2xs uppercase tracking-wide ${
                  idx < columns.length - 1 ? 'border-r border-line-1' : ''
                }`}
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ row, globalIdx }) => {
            const flags = badRowIndices?.get(globalIdx)
            const isSelected = selectedIdx === globalIdx
            // Style precedence: selected > flagged > default. Selected
            // gets a brand-tinted left bar; flagged gets warn; both use
            // distinct accents so the user can tell them apart even on
            // a row that's both selected and flagged.
            const rowClass = isSelected
              ? 'bg-brand/10 border-l-2 border-l-brand'
              : flags
                ? 'bg-warn/[0.06] border-l-2 border-l-warn/60 hover:bg-warn/[0.10]'
                : 'hover:bg-surface-2/40'
            return (
              <tr
                key={globalIdx}
                onClick={() => onSelect(isSelected ? null : globalIdx)}
                className={`border-t border-line-1/40 cursor-pointer ${rowClass}`}
                title={flags ? `flagged by: ${flags.join(', ')}` : undefined}
              >
                <td className="px-3 py-1.5 text-ink-4 align-middle border-r border-line-1/40">{globalIdx}</td>
                {columns.map((col, idx) => (
                  <td
                    key={col}
                    className={`px-3 py-1.5 align-middle ${
                      idx < columns.length - 1 ? 'border-r border-line-1/40' : ''
                    }`}
                  >
                    <CellValue value={row[col]} />
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function RawRows({
  rows,
  badRowIndices,
  selectedIdx,
  onSelect,
}: {
  rows: { row: Record<string, unknown>; globalIdx: number }[]
  badRowIndices?: Map<number, string[]>
  selectedIdx: number | null
  onSelect: (idx: number | null) => void
}) {
  return (
    <pre className="text-[12px] font-mono leading-5 px-3 py-2 m-0">
      {rows.map(({ row, globalIdx }) => {
        const flags = badRowIndices?.get(globalIdx)
        const isSelected = selectedIdx === globalIdx
        const rowClass = isSelected
          ? 'block px-1 -mx-1 rounded bg-brand/10 border-l-2 border-l-brand cursor-pointer'
          : flags
            ? 'block px-1 -mx-1 rounded bg-warn/[0.06] border-l-2 border-l-warn/60 cursor-pointer'
            : 'block px-1 -mx-1 cursor-pointer hover:bg-surface-2/40'
        return (
          <div
            key={globalIdx}
            onClick={() => onSelect(isSelected ? null : globalIdx)}
            className={rowClass}
            title={flags ? `flagged by: ${flags.join(', ')}` : undefined}
          >
            <span className="text-ink-4 select-none mr-2">{globalIdx}</span>
            <span className="whitespace-pre">{highlightJson(safeStringify(row))}</span>
          </div>
        )
      })}
    </pre>
  )
}

// ---------------------------------------------------------------------------
// Right-side detail panel
// ---------------------------------------------------------------------------

function RowDetailPanel({
  row,
  globalIdx,
  flags,
  onClose,
}: {
  row: Record<string, unknown>
  globalIdx: number
  flags?: string[]
  onClose: () => void
}) {
  const fields = useMemo(
    () => Object.keys(row).filter((k) => !k.startsWith('__')),
    [row],
  )
  return (
    <aside className="w-[420px] shrink-0 rounded-md border border-line-1 bg-surface-2 flex flex-col overflow-hidden">
      <header className="flex items-center gap-2 px-3 py-2 border-b border-line-1">
        <div className="text-2xs text-ink-3">row</div>
        <div className="text-xs font-mono text-ink-1">#{globalIdx}</div>
        {flags && flags.length > 0 && (
          <span
            className="pill pill-warn"
            title={`flagged by: ${flags.join(', ')}`}
          >
            flagged
          </span>
        )}
        <button
          type="button"
          onClick={onClose}
          className="ml-auto text-ink-3 hover:text-ink-1 text-xs"
          title="Close (Esc)"
        >
          ✕
        </button>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3 text-[12px] font-mono">
        {fields.length === 0 && (
          <div className="text-ink-4 italic">(empty row)</div>
        )}
        {fields.map((key) => (
          <DetailField key={key} fieldKey={key} value={row[key]} />
        ))}
      </div>
    </aside>
  )
}

function DetailField({ fieldKey, value }: { fieldKey: string; value: unknown }) {
  return (
    <div className="space-y-0.5">
      <div className="text-2xs text-ink-3 uppercase tracking-wide">{fieldKey}</div>
      <DetailValue value={value} />
    </div>
  )
}

function DetailValue({ value }: { value: unknown }) {
  if (value === undefined || value === null) {
    return <div className="text-ink-4 italic">—</div>
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return <div className="text-ink-1">{String(value)}</div>
  }
  if (typeof value === 'string') {
    return (
      <div className="text-ink-1 whitespace-pre-wrap break-words bg-surface-inset rounded px-2 py-1.5 border border-line-1/60">
        {value}
      </div>
    )
  }
  // Object / array — pretty JSON with syntax highlight.
  return (
    <pre className="m-0 text-[11px] text-ink-2 whitespace-pre-wrap break-words bg-surface-inset rounded px-2 py-1.5 border border-line-1/60">
      {highlightJson(safeStringify(value, true))}
    </pre>
  )
}

/**
 * Lightweight JSON tokenizer for the Raw preview mode. Returns a React
 * fragment with per-token spans so keys / strings / numbers / literals
 * pick up distinct colours.
 *
 * Why not @codemirror/lang-json? 150 kB bundle for a one-line preview
 * felt excessive. If/when we add inline row editing (Phase B), we'll
 * upgrade to a proper editor; for read-only preview this regex pass is
 * plenty.
 */
function highlightJson(src: string): React.ReactNode[] {
  // Tokens: strings (greedy with escapes), literals, numbers.
  // We disambiguate a string-as-key from a string-as-value by peeking
  // ahead for `:` skipping whitespace.
  const out: React.ReactNode[] = []
  const re = /"(?:\\.|[^"\\])*"|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g
  let cursor = 0
  let match: RegExpExecArray | null
  let keyCounter = 0
  while ((match = re.exec(src)) !== null) {
    if (match.index > cursor) {
      out.push(<span key={`p${cursor}`} className="text-ink-3">{src.slice(cursor, match.index)}</span>)
    }
    const token = match[0]
    if (token.startsWith('"')) {
      // Look past trailing whitespace for `:` → this is a key.
      const after = src.slice(match.index + token.length)
      const isKey = /^\s*:/.test(after)
      out.push(
        <span
          key={`t${match.index}-${keyCounter++}`}
          className={isKey ? 'text-info' : 'text-ok'}
        >
          {token}
        </span>,
      )
    } else if (token === 'true' || token === 'false') {
      out.push(<span key={`t${match.index}`} className="text-brand-alt">{token}</span>)
    } else if (token === 'null') {
      out.push(<span key={`t${match.index}`} className="text-ink-4">{token}</span>)
    } else {
      out.push(<span key={`t${match.index}`} className="text-warn">{token}</span>)
    }
    cursor = match.index + token.length
  }
  if (cursor < src.length) {
    out.push(<span key={`p${cursor}`} className="text-ink-3">{src.slice(cursor)}</span>)
  }
  return out
}

/**
 * Single-line cell renderer for the structured table. Long content is
 * truncated with ellipsis at the column width — full content lives in
 * the right-side detail panel. This is the HuggingFace dataset viewer
 * / Airtable / VS Code grid pattern: table for scanning, detail panel
 * for reading.
 *
 * Special-case for arrays / objects: render a typed chip (`[N items]`,
 * `{N keys}`) with a tone hint for "messages"-shaped fields so you can
 * see "this is a 3-message conversation" at a glance.
 */
function CellValue({ value }: { value: unknown }) {
  if (value === undefined || value === null) {
    return <span className="text-ink-4 italic">—</span>
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return <span className="text-ink-1">{String(value)}</span>
  }
  if (typeof value === 'string') {
    // first-line preview with an ellipsis; tooltip carries the full
    // string for power users hovering, but the canonical "read full
    // value" path is the detail panel.
    const firstLine = value.split('\n', 1)[0]
    return (
      <span
        className="block truncate text-ink-1"
        title={value.length > 80 || value.includes('\n') ? value : undefined}
      >
        {firstLine}
      </span>
    )
  }
  if (Array.isArray(value)) {
    return (
      <span className="inline-flex items-center gap-1 text-ink-3">
        <span className="pill pill-info text-[10px]">{`[${value.length}]`}</span>
        <span className="truncate text-2xs">
          {summariseArray(value)}
        </span>
      </span>
    )
  }
  if (typeof value === 'object') {
    const obj = value as Record<string, unknown>
    const keyCount = Object.keys(obj).length
    return (
      <span className="inline-flex items-center gap-1 text-ink-3">
        <span className="pill pill-skip text-[10px]">{`{${keyCount}}`}</span>
        <span className="truncate text-2xs">
          {summariseObject(obj)}
        </span>
      </span>
    )
  }
  return <span className="text-ink-2 truncate">{safeStringify(value)}</span>
}

/** Hint text for an array cell. For chat-style ``messages`` arrays we
 *  show roles ("system → user → assistant"); for plain arrays we show
 *  "N items" with the first scalar peeking through. */
function summariseArray(arr: unknown[]): string {
  if (arr.length === 0) return '(empty)'
  const looksLikeMessages = arr.every(
    (m) => m && typeof m === 'object' && !Array.isArray(m) && 'role' in (m as object),
  )
  if (looksLikeMessages) {
    const roles = arr
      .slice(0, 4)
      .map((m) => String((m as { role?: unknown }).role ?? '?'))
    if (arr.length > 4) roles.push(`+${arr.length - 4}`)
    return roles.join(' → ')
  }
  const first = arr[0]
  if (typeof first === 'string' || typeof first === 'number' || typeof first === 'boolean') {
    return `${arr.length} items · first: ${String(first).slice(0, 40)}`
  }
  return `${arr.length} items`
}

function summariseObject(obj: Record<string, unknown>): string {
  const keys = Object.keys(obj)
  if (keys.length === 0) return '(empty)'
  return keys.slice(0, 4).join(', ') + (keys.length > 4 ? '…' : '')
}

function safeStringify(v: unknown, pretty = false): string {
  try {
    return JSON.stringify(v, null, pretty ? 2 : undefined)
  } catch {
    return '[[unserialisable]]'
  }
}
