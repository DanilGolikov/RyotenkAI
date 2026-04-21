import { useCallback, useEffect, useState } from 'react'

/**
 * Local-only "draft" persistence for the project config form.
 *
 * # What "Save as draft" is (and isn't)
 *
 * The `Save` button pushes the YAML to the backend, which validates
 * it against the Pydantic schema and rejects anything that doesn't
 * satisfy required fields / type constraints. That's the right
 * behaviour for committing an actual version — we don't want broken
 * configs on disk. But it blocks the very common case where a user is
 * mid-fill and needs to go find a value: an `HF_TOKEN`, a dataset path
 * on a teammate's machine, a LoRA rank they want to double-check in
 * a notebook. If they close the tab at that point, they lose their
 * progress.
 *
 * `Save as draft` patches that hole by stashing the current YAML text
 * in `localStorage`, scoped per-project. Next time the same user opens
 * the same project in the same browser, they see a restore banner.
 *
 * # Why `localStorage` and not the backend
 *
 * - **No schema pressure.** Drafts can be half-written, syntactically
 *   weird, whatever — they're a personal notepad. A backend draft
 *   endpoint would have to accept anything, which means a separate
 *   bucket on disk and a second sync story.
 * - **Scope matches the need.** The problem is "I close the tab and
 *   lose stuff on this laptop" — device-local storage fits. If you
 *   need cross-device drafts, that's a real backend feature and
 *   should live in the config-versions flow (a "draft" version
 *   category next to "favorite").
 * - **Zero blast radius.** No new backend surface, no auth questions,
 *   no migration.
 *
 * Trade-off we accept: a draft from machine A isn't visible on
 * machine B. If that becomes a pain, promote to a backend endpoint.
 *
 * # Why this lives in a hook
 *
 * Keeps the ConfigTab component focused on state orchestration, hides
 * the `try/catch` wrappers around `localStorage` (which throws in
 * private-browsing + disabled-storage edge cases), and lets other
 * forms (provider config, run template) reuse the same pattern with
 * their own storage key.
 */
const DRAFT_STORAGE_PREFIX = 'ryotenkai:config-draft:'

interface Draft {
  yaml: string
  savedAt: string
}

function keyFor(projectId: string): string {
  return `${DRAFT_STORAGE_PREFIX}${projectId}`
}

function readDraft(projectId: string): Draft | null {
  try {
    const raw = localStorage.getItem(keyFor(projectId))
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<Draft>
    if (typeof parsed.yaml === 'string' && typeof parsed.savedAt === 'string') {
      return { yaml: parsed.yaml, savedAt: parsed.savedAt }
    }
    return null
  } catch {
    return null
  }
}

export interface UseConfigDraft {
  /** Draft detected at mount, if it differs from the server config. */
  draftPrompt: Draft | null
  /** Dismiss the restore banner (keeps the draft on disk). */
  dismissPrompt: () => void
  /** Brief "Draft saved ✓" flash on the button. */
  savedFlash: boolean
  /** Save the current YAML as a draft. */
  save: (yaml: string) => void
  /** Remove the draft (call after a successful Save to server). */
  clear: () => void
  /** Probe once — useful to re-check after the server config has
   *  loaded, so we only prompt when the draft genuinely differs. */
  probe: (serverYaml: string | undefined | null) => void
}

export function useConfigDraft(projectId: string): UseConfigDraft {
  const [draftPrompt, setDraftPrompt] = useState<Draft | null>(null)
  const [savedFlash, setSavedFlash] = useState(false)

  const save = useCallback(
    (yaml: string) => {
      try {
        localStorage.setItem(
          keyFor(projectId),
          JSON.stringify({ yaml, savedAt: new Date().toISOString() } satisfies Draft),
        )
        setSavedFlash(true)
      } catch {
        /* quota / disabled — best-effort */
      }
    },
    [projectId],
  )

  // Flash auto-resets.
  useEffect(() => {
    if (!savedFlash) return
    const t = window.setTimeout(() => setSavedFlash(false), 2000)
    return () => window.clearTimeout(t)
  }, [savedFlash])

  const clear = useCallback(() => {
    try {
      localStorage.removeItem(keyFor(projectId))
    } catch {
      /* ignore */
    }
    setDraftPrompt(null)
  }, [projectId])

  const dismissPrompt = useCallback(() => setDraftPrompt(null), [])

  const probe = useCallback(
    (serverYaml: string | undefined | null) => {
      if (typeof serverYaml !== 'string') return
      const draft = readDraft(projectId)
      if (draft && draft.yaml.trim() !== serverYaml.trim()) {
        setDraftPrompt(draft)
      }
    },
    [projectId],
  )

  return { draftPrompt, dismissPrompt, savedFlash, save, clear, probe }
}
