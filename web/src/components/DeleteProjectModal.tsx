import { useEffect, useState } from 'react'
import type { ProjectSummary } from '../api/types'
import { Toggle } from './ui'

interface Props {
  project: ProjectSummary
  onClose: () => void
  onConfirm: (deleteFiles: boolean) => void
  pending?: boolean
}

/**
 * Two-step delete confirmation.
 *
 * Step 1: user chooses whether to wipe the on-disk workspace or just
 *   unregister from the project index.
 * Step 2 (safety): the user types the project id. Mirrors how GitHub /
 *   Vercel / Neon handle "delete repo" — an opinionated extra friction
 *   layer for an irreversible action.
 */
export function DeleteProjectModal({ project, onClose, onConfirm, pending }: Props) {
  const [deleteFiles, setDeleteFiles] = useState(true)
  const [confirmText, setConfirmText] = useState('')

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  const canConfirm = confirmText.trim() === project.id && !pending

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={(e) => {
        e.stopPropagation()
        e.preventDefault()
        if (!pending) onClose()
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-project-title"
        onClick={(e) => {
          e.stopPropagation()
          e.preventDefault()
        }}
        className="w-full max-w-md rounded-lg border border-line-2 bg-surface-1 shadow-card overflow-hidden"
      >
        <div className="px-5 py-4 border-b border-line-1 flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-err/15 text-err inline-flex items-center justify-center shrink-0">
            <svg
              className="w-4 h-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M3 6h18" />
              <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
              <path d="M5 6l1 14a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-14" />
              <path d="M10 11v6" />
              <path d="M14 11v6" />
            </svg>
          </div>
          <div className="min-w-0">
            <h2 id="delete-project-title" className="text-sm font-semibold text-ink-1">
              Delete project
            </h2>
            <div className="text-2xs text-ink-3 font-mono truncate">{project.id}</div>
          </div>
        </div>

        <div className="px-5 py-4 space-y-4">
          <p className="text-xs text-ink-2 leading-relaxed">
            This action is irreversible. All config snapshots and metadata
            for this project will be permanently removed.
          </p>

          <div className="flex items-start gap-3">
            <div className="mt-0.5">
              <Toggle
                checked={deleteFiles}
                onChange={setDeleteFiles}
                variant="danger"
                aria-label="Also delete workspace on disk"
              />
            </div>
            <label
              className="text-xs text-ink-2 cursor-pointer select-none"
              onClick={() => setDeleteFiles((v) => !v)}
            >
              <span className="text-ink-1 font-medium">Also delete workspace on disk</span>
              <span className="block text-2xs text-ink-4 font-mono mt-0.5 break-all">
                {project.path}
              </span>
            </label>
          </div>

          <div>
            <label className="text-2xs text-ink-3 font-medium block mb-1.5">
              Type the project id to confirm
            </label>
            <input
              type="text"
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder={project.id}
              autoFocus
              className="w-full h-8 rounded bg-surface-1 border border-line-1 px-2.5 text-[13px] text-ink-1 font-mono focus:outline-none focus:border-err hover:border-line-2 transition-colors"
            />
          </div>
        </div>

        <div className="px-5 py-3 border-t border-line-1 flex items-center justify-end gap-2 bg-surface-0/50">
          <button
            type="button"
            onClick={onClose}
            disabled={pending}
            className="h-8 px-3 rounded text-xs text-ink-2 border border-line-1 hover:text-ink-1 hover:border-line-2 transition disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onConfirm(deleteFiles)}
            disabled={!canConfirm}
            className="h-8 px-3 rounded text-xs font-medium inline-flex items-center gap-1.5 bg-err/15 text-err border border-err/40 hover:bg-err/25 hover:border-err/70 transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {pending ? 'Deleting…' : 'Delete project'}
          </button>
        </div>
      </div>
    </div>
  )
}
