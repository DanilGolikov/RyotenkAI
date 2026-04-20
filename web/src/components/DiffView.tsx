import { Fragment, useMemo, useState } from 'react'
import {
  diffLines,
  foldDiff,
  groupHunks,
  revertHunk,
  type DiffHunk,
  type DiffLine,
} from '../lib/lineDiff'
import { tokenizeYamlLine } from '../lib/yamlTokens'
import { Spinner } from './ui'

interface Props {
  oldText: string
  newText: string
  loading?: boolean
  /**
   * Called when the user reverts a single hunk — gets the full new-text
   * with that hunk replaced by its old content. Omit to hide revert UI.
   */
  onRevertHunk?: (nextText: string, hunk: DiffHunk) => void
  /** Header labels. Accept React nodes so callers can pass interactive
   *  version pickers instead of static text. */
  oldLabel?: React.ReactNode
  newLabel?: React.ReactNode
  maxHeight?: string
}

export function DiffView({
  oldText,
  newText,
  loading = false,
  onRevertHunk,
  oldLabel,
  newLabel,
  maxHeight = 'max-h-[560px]',
}: Props) {
  const lines = useMemo(() => diffLines(oldText, newText), [oldText, newText])
  const hunks = useMemo(() => groupHunks(lines), [lines])
  // GitHub-style context: 5 lines above each change, 3 below. Big
  // stretches of unchanged YAML are collapsed into an "Expand" fold.
  const [expandedFolds, setExpandedFolds] = useState<Set<number>>(new Set())
  const chunks = useMemo(
    () => foldDiff(lines, { before: 5, after: 3 }),
    [lines],
  )

  const adds = lines.filter((l) => l.type === 'add').length
  const dels = lines.filter((l) => l.type === 'del').length

  const hunkByLineIdx = new Map<number, DiffHunk>()
  hunks.forEach((h) => {
    for (let k = 0; k < h.length; k++) {
      hunkByLineIdx.set(h.startIdx + k, h)
    }
  })

  return (
    <div className="rounded-md border border-line-1 bg-surface-0 overflow-hidden">
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-line-1 text-[0.65rem] font-mono text-ink-3">
        <span className="text-ok">+{adds}</span>
        <span className="text-err">−{dels}</span>
        {(oldLabel || newLabel) && (
          <span className="ml-3 text-ink-4 truncate flex items-center gap-1.5">
            {oldLabel && (
              <>
                <span>old:</span>
                {oldLabel}
              </>
            )}
            {oldLabel && newLabel && <span className="text-ink-4">·</span>}
            {newLabel && (
              <>
                <span>new:</span>
                {newLabel}
              </>
            )}
          </span>
        )}
        <span className="ml-auto">
          {hunks.length} hunk{hunks.length === 1 ? '' : 's'}
        </span>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 text-xs text-ink-3 px-3 py-3">
          <Spinner /> loading…
        </div>
      ) : lines.length === 0 ? (
        <div className="text-xs text-ink-3 px-3 py-3">Empty diff.</div>
      ) : hunks.length === 0 ? (
        <div className="px-3 py-3 text-xs text-ink-3">
          Identical content on both sides.
        </div>
      ) : (
        <div className={`${maxHeight} overflow-auto font-mono text-xs leading-5`}>
          {chunks.map((chunk, ci) => {
            if (chunk.kind === 'fold') {
              if (expandedFolds.has(chunk.startIdx)) {
                return (
                  <Fragment key={`fold-${ci}`}>
                    {lines
                      .slice(chunk.startIdx, chunk.startIdx + chunk.count)
                      .map((l, k) => {
                        const idx = chunk.startIdx + k
                        const hunk = hunkByLineIdx.get(idx)
                        const isHunkHead = hunk && hunk.startIdx === idx
                        return (
                          <DiffLineRow
                            key={idx}
                            line={l}
                            revertButton={
                              isHunkHead && onRevertHunk ? (
                                <RevertArrow
                                  title="Revert this hunk to the old version"
                                  onClick={() =>
                                    onRevertHunk(revertHunk(lines, hunk!), hunk!)
                                  }
                                />
                              ) : null
                            }
                          />
                        )
                      })}
                  </Fragment>
                )
              }
              return (
                <FoldRow
                  key={`fold-${ci}`}
                  count={chunk.count}
                  onExpand={() =>
                    setExpandedFolds((prev) => {
                      const next = new Set(prev)
                      next.add(chunk.startIdx)
                      return next
                    })
                  }
                />
              )
            }
            return (
              <Fragment key={`lines-${ci}`}>
                {chunk.lines.map((l, k) => {
                  const idx = chunk.startIdx + k
                  const hunk = hunkByLineIdx.get(idx)
                  const isHunkHead = hunk && hunk.startIdx === idx
                  return (
                    <DiffLineRow
                      key={idx}
                      line={l}
                      revertButton={
                        isHunkHead && onRevertHunk ? (
                          <RevertArrow
                            title="Revert this hunk to the old version"
                            onClick={() =>
                              onRevertHunk(revertHunk(lines, hunk!), hunk!)
                            }
                          />
                        ) : null
                      }
                    />
                  )
                })}
              </Fragment>
            )
          })}
        </div>
      )}
    </div>
  )
}

function DiffLineRow({
  line,
  revertButton,
}: {
  line: DiffLine
  revertButton: React.ReactNode
}) {
  const segs = useMemo(() => tokenizeYamlLine(line.text || ''), [line.text])
  const bg =
    line.type === 'add'
      ? 'bg-ok/10'
      : line.type === 'del'
        ? 'bg-err/10'
        : ''
  const marker =
    line.type === 'add' ? (
      <span className="text-ok">+</span>
    ) : line.type === 'del' ? (
      <span className="text-err">−</span>
    ) : (
      ' '
    )
  return (
    <div className={`group flex gap-3 px-3 ${bg}`}>
      <span className="w-10 shrink-0 text-right text-ink-4 select-none">
        {line.oldNo ?? ''}
      </span>
      <span className="w-10 shrink-0 text-right text-ink-4 select-none">
        {line.newNo ?? ''}
      </span>
      <span className="w-4 shrink-0 text-center select-none">{marker}</span>
      <span className="whitespace-pre-wrap break-words flex-1">
        {segs.length === 0 ? (
          <span>&nbsp;</span>
        ) : (
          segs.map((s, i) => (
            <span key={i} className={s.cls}>
              {s.text}
            </span>
          ))
        )}
      </span>
      {revertButton && (
        <span className="shrink-0 self-center">{revertButton}</span>
      )}
    </div>
  )
}

function FoldRow({ count, onExpand }: { count: number; onExpand: () => void }) {
  return (
    <button
      type="button"
      onClick={onExpand}
      className="w-full flex items-center gap-2 px-3 py-1 bg-surface-1/40 hover:bg-surface-2 border-y border-line-1/60 text-ink-4 hover:text-ink-2 transition"
      title={`Expand ${count} unchanged line${count === 1 ? '' : 's'}`}
    >
      <span className="text-[0.65rem]">⋯</span>
      <span className="text-[0.65rem]">
        {count} unchanged line{count === 1 ? '' : 's'}
      </span>
      <span className="ml-auto text-[0.6rem] uppercase tracking-wide">expand</span>
    </button>
  )
}

function RevertArrow({
  onClick,
  title,
}: {
  onClick: () => void
  title: string
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-line-1 text-ink-3 hover:text-ink-1 hover:border-brand hover:bg-brand/10 transition text-[0.6rem]"
    >
      <svg
        className="w-3 h-3"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <path d="M9 14l-4-4 4-4" />
        <path d="M5 10h11a4 4 0 0 1 0 8h-3" />
      </svg>
      <span>revert</span>
    </button>
  )
}
