import { useMemo, useState } from 'react'
import {
  useProjectConfig,
  useProjectConfigVersions,
  useReadConfigVersion,
  useRestoreConfigVersion,
  useSaveProjectConfig,
  useToggleFavoriteVersion,
} from '../../api/hooks/useProjects'
import type { ConfigVersion } from '../../api/types'
import { DiffView } from '../DiffView'
import { SelectField } from '../ConfigBuilder/SelectField'
import { Spinner } from '../ui'
import { YamlView } from '../YamlView'

const CURRENT = 'current'

/**
 * Versions tab: snapshot list on the left, preview-or-diff pane on the
 * right. Clicking a version shows a single-version preview; the header
 * has two clickable pickers (old / new) that switch to a git-style diff
 * whenever both sides are chosen and differ. Each hunk has a ← revert
 * button that saves a new config with only that chunk rolled back.
 */
export function VersionsTab({ projectId }: { projectId: string }) {
  const versionsQuery = useProjectConfigVersions(projectId)
  const configQuery = useProjectConfig(projectId)
  const favMut = useToggleFavoriteVersion(projectId)
  const restoreMut = useRestoreConfigVersion(projectId)
  const saveMut = useSaveProjectConfig(projectId)

  const [leftSel, setLeftSel] = useState<string | null>(null)
  const [rightSel, setRightSel] = useState<string | null>(null)

  const versions = versionsQuery.data?.versions ?? []

  // v1 = oldest. Filenames are ISO timestamps so a lexicographic sort
  // is chronological.
  const labelMap = useMemo(() => {
    const byDate = [...versions].sort((a, b) => a.filename.localeCompare(b.filename))
    const m = new Map<string, string>()
    byDate.forEach((v, i) => m.set(v.filename, `v${i + 1}`))
    return m
  }, [versions])

  const leftFilename = leftSel && leftSel !== CURRENT ? leftSel : null
  const rightFilename = rightSel && rightSel !== CURRENT ? rightSel : null
  const leftVersionQuery = useReadConfigVersion(projectId, leftFilename)
  const rightVersionQuery = useReadConfigVersion(projectId, rightFilename)

  const pickerOptions = useMemo(() => {
    const opts = [{ value: CURRENT, label: 'current · live config' }]
    const byNewest = [...versions].sort((a, b) => b.filename.localeCompare(a.filename))
    for (const v of byNewest) {
      const vn = labelMap.get(v.filename) ?? v.filename
      opts.push({
        value: v.filename,
        label: `${vn} · ${v.created_at}`,
      })
    }
    return opts
  }, [versions, labelMap])

  async function onRevertHunk(nextText: string) {
    try {
      await saveMut.mutateAsync(nextText)
    } catch (exc) {
      window.alert((exc as Error).message || 'Failed to revert hunk.')
    }
  }

  if (versionsQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-xs text-ink-3">
        <Spinner /> loading versions
      </div>
    )
  }
  if (versionsQuery.error) {
    return <div className="text-xs text-err">{(versionsQuery.error as Error).message}</div>
  }
  if (versions.length === 0) {
    return (
      <div className="text-xs text-ink-3">
        No snapshots yet. Each save creates one automatically.
      </div>
    )
  }

  const currentYaml = configQuery.data?.yaml ?? ''

  function yamlFor(
    sel: string | null,
    q: { data?: { yaml: string } | undefined },
  ): string {
    if (sel === null) return ''
    if (sel === CURRENT) return currentYaml
    return q.data?.yaml ?? ''
  }

  const leftYaml = yamlFor(leftSel, leftVersionQuery)
  const rightYaml = yamlFor(rightSel, rightVersionQuery)

  function labelFor(sel: string | null): string {
    if (sel === null) return '—'
    if (sel === CURRENT) return 'current'
    return labelMap.get(sel) ?? sel
  }

  // Decide what to render in the right pane.
  const bothPicked =
    leftSel !== null && rightSel !== null && leftSel !== rightSel
  const singlePick =
    (leftSel !== null && rightSel === null) ||
    (leftSel === null && rightSel !== null) ||
    (leftSel !== null && rightSel !== null && leftSel === rightSel)

  const previewSel = leftSel ?? rightSel
  const previewYaml =
    previewSel === leftSel ? leftYaml : previewSel === rightSel ? rightYaml : ''

  const anyLoading =
    (leftSel !== null && leftSel !== CURRENT && leftVersionQuery.isLoading) ||
    (rightSel !== null && rightSel !== CURRENT && rightVersionQuery.isLoading)

  return (
    // `h-full` pulls height from the Card's scrolling region, which
    // is itself bounded to the viewport (see ProjectDetail.tsx — the
    // page is a viewport-height flex column, the card takes the
    // remainder, its content area is the scroll container). Each
    // column then gets `min-h-0 overflow-y-auto` and scrolls inside
    // its own box — no page-level scrollbar.
    <div className="grid grid-cols-[280px_1fr] gap-4 h-full">
      {/* Snapshot list — internal scroll. `min-h-0` is the flex /
          grid "unstuck overflow" trick: without it, the list would
          prefer its intrinsic height and push the grid row taller
          than the container. */}
      <div className="space-y-1 min-h-0 overflow-y-auto pr-2">
        <div className="text-[0.65rem] uppercase tracking-wider text-ink-4 font-medium px-1 pb-1">
          Snapshots
        </div>
        {versions.map((v: ConfigVersion) => {
          const vn = labelMap.get(v.filename) ?? v.filename
          const active = leftSel === v.filename || rightSel === v.filename
          const fav = !!v.is_favorite
          return (
            <div
              key={v.filename}
              className={[
                // Hover highlights the border in the active-state colour
                // without filling the body — previously the hover applied
                // `hover:border-line-2` which was nearly invisible and the
                // active state added a dark `bg-surface-2` fill that made
                // hover feel stuck halfway through a selection animation.
                // Now hover reads as "outlined target" and active adds
                // the fill as a clear "this one is chosen" signal.
                'rounded-md px-3 py-2 text-xs border transition-colors flex items-start gap-2',
                active
                  ? 'border-brand bg-surface-2 text-ink-1'
                  : fav
                    ? 'border-warn/60 hover:border-warn text-ink-1'
                    : 'border-line-1 hover:border-brand/60 text-ink-2 hover:text-ink-1',
              ].join(' ')}
            >
              <button
                type="button"
                disabled={favMut.isPending}
                onClick={(e) => {
                  e.stopPropagation()
                  favMut.mutate({ filename: v.filename, favorite: !fav })
                }}
                title={fav ? 'Unpin from favorites' : 'Pin as favorite'}
                className={[
                  'shrink-0 text-base leading-none transition',
                  fav ? 'text-warn' : 'text-ink-4 hover:text-warn',
                ].join(' ')}
              >
                {fav ? '★' : '☆'}
              </button>
              <button
                type="button"
                onClick={() => {
                  // Default click: preview-only. Put the version on the
                  // left, clear the right so we don't spuriously diff.
                  setLeftSel(v.filename)
                  setRightSel(null)
                }}
                className="flex-1 min-w-0 text-left"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-ink-1 font-medium font-mono text-xs shrink-0">
                    {vn}
                  </span>
                  <span className="text-ink-4 font-mono text-[0.6rem] truncate">
                    .yaml
                  </span>
                </div>
                <div className="text-ink-3 text-[0.65rem] mt-0.5">
                  {v.created_at} · {v.size_bytes} bytes
                </div>
              </button>
            </div>
          )
        })}
      </div>

      {/* Right pane — fixed-height flex column. Picker row + label row
          take their natural height, YamlView absorbs the rest with
          `flex-1 min-h-0` and scrolls internally. */}
      <div className="min-w-0 min-h-0 flex flex-col space-y-2">
        <div className="flex items-center gap-3 text-[0.65rem] font-mono text-ink-3 flex-wrap">
          <span className="text-ink-4">old:</span>
          <SelectField
            value={leftSel ?? ''}
            options={pickerOptions}
            onChange={(next) => setLeftSel(next === '' ? null : next)}
            placeholder="—"
            allowEmpty
            triggerClassName="w-auto min-w-[180px]"
          />
          <span className="text-ink-4">new:</span>
          <SelectField
            value={rightSel ?? ''}
            options={pickerOptions}
            onChange={(next) => setRightSel(next === '' ? null : next)}
            placeholder="—"
            allowEmpty
            triggerClassName="w-auto min-w-[180px]"
          />
          {(leftSel !== null || rightSel !== null) && (
            <button
              type="button"
              onClick={() => {
                setLeftSel(null)
                setRightSel(null)
              }}
              className="text-ink-4 hover:text-ink-2 transition px-1"
              title="Clear selection"
            >
              clear
            </button>
          )}
        </div>

        {leftSel === null && rightSel === null ? (
          <div className="text-xs text-ink-3 rounded-md border border-line-1 bg-surface-0 px-3 py-6 text-center">
            Click a snapshot to preview, or pick both old &amp; new to compare.
          </div>
        ) : anyLoading ? (
          <div className="flex items-center gap-2 text-xs text-ink-3 px-3 py-3">
            <Spinner /> loading…
          </div>
        ) : bothPicked ? (
          <>
            <DiffView
              oldText={leftYaml}
              newText={rightYaml}
              oldLabel={
                <span className="text-ink-1 font-mono">{labelFor(leftSel)}</span>
              }
              newLabel={
                <span className="text-ink-1 font-mono">{labelFor(rightSel)}</span>
              }
              onRevertHunk={(next) => onRevertHunk(next)}
            />
            <div className="flex items-center justify-between gap-2">
              {saveMut.error ? (
                <span className="text-err text-2xs">
                  {(saveMut.error as Error).message}
                </span>
              ) : saveMut.isPending ? (
                <span className="text-ink-3 text-2xs flex items-center gap-2">
                  <Spinner /> saving…
                </span>
              ) : restoreMut.error ? (
                <span className="text-err text-2xs">
                  {(restoreMut.error as Error).message}
                </span>
              ) : (
                <span className="text-ink-4 text-2xs">
                  Click ← revert on any hunk to save it as a new snapshot.
                </span>
              )}
              {leftSel !== CURRENT && leftSel !== null && (
                <button
                  type="button"
                  disabled={restoreMut.isPending}
                  onClick={() => restoreMut.mutate(leftSel)}
                  className="btn-primary px-3 py-1.5 text-xs"
                >
                  {restoreMut.isPending
                    ? 'Restoring…'
                    : `Restore ${labelFor(leftSel)}`}
                </button>
              )}
            </div>
          </>
        ) : singlePick ? (
          <>
            <div className="flex items-center justify-between">
              <div className="text-[0.65rem] font-mono text-ink-3">
                preview:{' '}
                <span className="text-ink-1">
                  {labelFor(previewSel)}
                </span>
              </div>
              {previewSel !== CURRENT && previewSel !== null && (
                <button
                  type="button"
                  disabled={restoreMut.isPending}
                  onClick={() => restoreMut.mutate(previewSel)}
                  className="btn-primary px-3 py-1.5 text-xs"
                >
                  {restoreMut.isPending
                    ? 'Restoring…'
                    : `Restore ${labelFor(previewSel)}`}
                </button>
              )}
            </div>
            <YamlView
              text={previewYaml || '# (empty)'}
              // `flex-1 min-h-0` absorbs the remaining height inside
              // the sticky wrapper (after the picker + label rows).
              // `maxHeight` becomes the upper bound — internal scroll
              // kicks in when a version's YAML is longer than viewport.
              className="flex-1 min-h-0"
              maxHeight="max-h-full"
              toolbarExtra={
                <span className="text-[0.65rem] font-mono text-ink-3">
                  {labelFor(previewSel)}
                </span>
              }
            />
            {restoreMut.error && (
              <div className="text-err text-2xs">
                {(restoreMut.error as Error).message}
              </div>
            )}
          </>
        ) : null}
      </div>
    </div>
  )
}
