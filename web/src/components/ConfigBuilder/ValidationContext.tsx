import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ConfigValidationResult } from '../../api/types'
import type { JsonSchemaNode } from '../../api/hooks/useConfigSchema'

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
  /** Client-side validation errors (range, enum, pattern) computed
   *  synchronously from the JSON schema — merged into field status
   *  alongside server errors. Server wins on conflicting path. */
  clientErrors: FieldErrors
  reportClientError: (path: string, message: string | null) => void
  /** Full last validation result — lets chips & banners read check
   *  statuses without prop-drilling through FieldRenderer. */
  validationResult: ConfigValidationResult | null
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
  validationResult = null,
  onRequestValidate,
  children,
}: {
  fieldErrors: FieldErrors
  setFieldErrors: (errs: FieldErrors) => void
  validationResult?: ConfigValidationResult | null
  onRequestValidate: () => void
  children: React.ReactNode
}) {
  const [focusedPath, setFocusedPath] = useState<string | null>(null)
  const [dirtyPaths, setDirtyPaths] = useState<Set<string>>(() => new Set())
  // Client-side errors live inside the provider because they're
  // computed per-field inside `useClientFieldValidation` (see hook
  // below) and the provider is the natural ring — useFieldStatus needs
  // to read them during render.
  const [clientErrors, setClientErrors] = useState<FieldErrors>({})

  const markDirty = useCallback((path: string) => {
    setDirtyPaths((prev) => {
      if (prev.has(path)) return prev
      const next = new Set(prev)
      next.add(path)
      return next
    })
  }, [])

  const reportClientError = useCallback(
    (path: string, message: string | null) => {
      setClientErrors((prev) => {
        const existing = prev[path]
        if (!message) {
          if (!existing) return prev
          const { [path]: _drop, ...rest } = prev
          void _drop
          return rest
        }
        if (existing && existing.length === 1 && existing[0] === message) return prev
        return { ...prev, [path]: [message] }
      })
    },
    [],
  )

  const value = useMemo<ValidationCtx>(
    () => ({
      fieldErrors,
      setFieldErrors,
      clientErrors,
      reportClientError,
      validationResult,
      focusedPath,
      setFocusedPath,
      dirtyPaths,
      markDirty,
      requestValidate: onRequestValidate,
    }),
    [
      fieldErrors,
      setFieldErrors,
      clientErrors,
      reportClientError,
      validationResult,
      focusedPath,
      dirtyPaths,
      markDirty,
      onRequestValidate,
    ],
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
  const clientErrs = ctx.clientErrors[path]
  if (clientErrs && clientErrs.length > 0) {
    return { state: 'error', message: clientErrs.join('; ') }
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

/**
 * Validate a field value against its JSON schema constraints
 * synchronously and report the error into ValidationContext. Covers
 * `minimum` / `maximum` (numbers), `minLength` / `maxLength`, `pattern`,
 * and `enum` when the schema lists allowed values. Designed to run
 * cheaply on every render — checks are all O(1) apart from the regex
 * `pattern` test which compiles the regex once per node reference.
 *
 * Returns nothing; side effect lives in the context. Render flows
 * through the same `useFieldStatus` pipe, so client & server errors
 * share the same "error" visual state.
 */
export function useClientFieldValidation(
  path: string,
  node: JsonSchemaNode | undefined,
  value: unknown,
): void {
  const ctx = useContext(ValidationContextImpl)
  const touched = ctx?.dirtyPaths.has(path) ?? false

  const message = useMemo((): string | null => {
    if (!node) return null
    // Never scream before the user has touched the field, otherwise
    // presets or defaults can surface "enum mismatch" errors on mount.
    if (!touched) return null
    if (value === undefined || value === null) return null
    if (typeof value === 'number') {
      const min = typeof node.minimum === 'number' ? node.minimum : null
      const max = typeof node.maximum === 'number' ? node.maximum : null
      if (min !== null && value < min) return `Must be ≥ ${min}`
      if (max !== null && value > max) return `Must be ≤ ${max}`
    }
    if (typeof value === 'string') {
      const minLen = typeof node.minLength === 'number' ? node.minLength : null
      const maxLen = typeof node.maxLength === 'number' ? node.maxLength : null
      if (minLen !== null && value.length < minLen) return `Must be at least ${minLen} chars`
      if (maxLen !== null && value.length > maxLen) return `Must be at most ${maxLen} chars`
      if (typeof node.pattern === 'string' && node.pattern.length > 0) {
        try {
          const re = new RegExp(node.pattern)
          if (!re.test(value)) return 'Value does not match expected format'
        } catch {
          // Bad pattern in schema — skip, backend will catch.
        }
      }
    }
    if (Array.isArray(node.enum) && node.enum.length > 0) {
      if (!node.enum.includes(value as never)) {
        const preview = node.enum.slice(0, 4).map((v) => JSON.stringify(v)).join(', ')
        const more = node.enum.length > 4 ? `, … (${node.enum.length} total)` : ''
        return `Must be one of: ${preview}${more}`
      }
    }
    return null
  }, [node, value, touched])

  useEffect(() => {
    if (!ctx) return
    ctx.reportClientError(path, message)
    // On unmount, clear client error for this path.
    return () => ctx.reportClientError(path, null)
  }, [ctx, path, message])
}
