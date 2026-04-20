import { createContext, useCallback, useContext, useMemo, useState } from 'react'

type FieldErrors = Record<string, string[]>

export interface FieldStatus {
  /** Visual state for the field block. */
  state: 'idle' | 'editing' | 'ok' | 'error'
  /** Joined error message(s) when ``state === 'error'``. */
  message?: string
}

interface ValidationCtx {
  /** Backend field errors from the last ``/validate`` call, keyed by
   *  dotted Pydantic ``loc`` path. */
  fieldErrors: FieldErrors
  setFieldErrors: (errs: FieldErrors) => void
  /** Currently focused field path — drives the yellow "editing" state. */
  focusedPath: string | null
  setFocusedPath: (p: string | null) => void
  /** Paths the user has interacted with this session. Used so first
   *  render doesn't scream "required" at fresh forms — only fields the
   *  user has touched show the red empty-required state. */
  dirtyPaths: Set<string>
  markDirty: (path: string) => void
  /** Caller-provided hook, typically a debounced save+validate. Invoked
   *  on field blur so server validation refreshes without Save click. */
  requestValidate: () => void
}

const ValidationContextImpl = createContext<ValidationCtx | null>(null)

export function ValidationProvider({
  fieldErrors,
  setFieldErrors,
  onRequestValidate,
  children,
}: {
  fieldErrors: FieldErrors
  setFieldErrors: (errs: FieldErrors) => void
  onRequestValidate: () => void
  children: React.ReactNode
}) {
  const [focusedPath, setFocusedPath] = useState<string | null>(null)
  const [dirtyPaths, setDirtyPaths] = useState<Set<string>>(() => new Set())

  const markDirty = useCallback((path: string) => {
    setDirtyPaths((prev) => {
      if (prev.has(path)) return prev
      const next = new Set(prev)
      next.add(path)
      return next
    })
  }, [])

  const value = useMemo<ValidationCtx>(
    () => ({
      fieldErrors,
      setFieldErrors,
      focusedPath,
      setFocusedPath,
      dirtyPaths,
      markDirty,
      requestValidate: onRequestValidate,
    }),
    [fieldErrors, setFieldErrors, focusedPath, dirtyPaths, markDirty, onRequestValidate],
  )

  return (
    <ValidationContextImpl.Provider value={value}>{children}</ValidationContextImpl.Provider>
  )
}

export function useValidationCtx(): ValidationCtx | null {
  return useContext(ValidationContextImpl)
}

/**
 * Status for a single field. Priority:
 * 1. Server error for this path → error (red) with the message
 * 2. Focused → editing (yellow)
 * 3. Required + empty + user has touched it → error (red, no message)
 * 4. Has value + no error → ok (green)
 * 5. Otherwise → idle (no bar)
 */
export function useFieldStatus(
  path: string,
  required: boolean,
  value: unknown,
): FieldStatus {
  const ctx = useContext(ValidationContextImpl)
  if (!ctx) return { state: 'idle' }

  const serverErrs = ctx.fieldErrors[path]
  if (serverErrs && serverErrs.length > 0) {
    return { state: 'error', message: serverErrs.join('; ') }
  }
  if (ctx.focusedPath === path) {
    return { state: 'editing' }
  }
  const isEmpty =
    value === undefined ||
    value === null ||
    (typeof value === 'string' && value.trim() === '')
  const touched = ctx.dirtyPaths.has(path)
  if (required && isEmpty && touched) {
    return { state: 'error', message: 'Required' }
  }
  if (!isEmpty) return { state: 'ok' }
  return { state: 'idle' }
}
