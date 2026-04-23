import type { PropsWithChildren, ReactNode } from 'react'

export function Card({
  children,
  className = '',
  raised = false,
  hero = false,
  padding = 'p-4',
}: PropsWithChildren<{
  className?: string
  raised?: boolean
  /** The one brand-washed card per screen — burgundy→violet wash at the top.
   *  Use sparingly; guideline says exactly one per screen. */
  hero?: boolean
  padding?: string
}>) {
  const variant = hero ? 'card-hero' : raised ? 'card-raised' : 'card'
  return (
    <div className={`${variant} ${padding} ${className}`}>{children}</div>
  )
}

export function SectionHeader({
  title,
  action,
  subtitle,
}: {
  title: ReactNode
  subtitle?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="flex items-end justify-between gap-4 mb-3">
      <div className="min-w-0 flex-1">
        <h2 className="text-sm font-medium text-ink">{title}</h2>
        {subtitle && <div className="text-2xs text-ink-3 mt-0.5">{subtitle}</div>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  )
}

export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string
  hint?: string
  action?: ReactNode
}) {
  return (
    <div className="px-6 py-12 text-center">
      <div className="text-ink-2 text-sm">{title}</div>
      {hint && <div className="text-ink-3 text-xs mt-1">{hint}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function Spinner() {
  return (
    <span
      aria-hidden
      className="inline-block w-3.5 h-3.5 rounded-full border-2 border-line-2/30 border-t-brand animate-spin"
    />
  )
}

/**
 * Pill-style switch (thumb slides left → right).
 *
 * Replacement for native ``<input type="checkbox">`` across the app so
 * on/off state is visually unambiguous. Semantically a WAI-ARIA switch
 * (``role="switch"`` + ``aria-checked``) — screen readers read it as
 * "on/off" rather than "checked/unchecked", which matches the intent.
 *
 * ``variant="danger"`` uses the error colour when the switch is on —
 * for destructive opt-ins (e.g. "also delete workspace on disk").
 */
export function Toggle({
  checked,
  onChange,
  disabled = false,
  variant = 'brand',
  id,
  'aria-label': ariaLabel,
  'aria-labelledby': ariaLabelledby,
  title,
  onFocus,
  onBlur,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  disabled?: boolean
  variant?: 'brand' | 'danger'
  id?: string
  'aria-label'?: string
  'aria-labelledby'?: string
  title?: string
  onFocus?: () => void
  onBlur?: () => void
}) {
  const onColor = variant === 'danger' ? 'bg-err' : 'bg-brand'
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledby}
      disabled={disabled}
      id={id}
      title={title}
      onFocus={onFocus}
      onBlur={onBlur}
      onClick={(e) => {
        e.stopPropagation()
        onChange(!checked)
      }}
      className={[
        'relative inline-flex h-4 w-7 shrink-0 items-center rounded-full',
        'border transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-brand',
        disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer',
        checked
          ? `${onColor} border-transparent`
          : 'bg-surface-1 border-line-2 hover:border-ink-3',
      ].join(' ')}
    >
      <span
        aria-hidden
        className={[
          'inline-block h-3 w-3 rounded-full bg-white shadow transition-transform',
          checked ? 'translate-x-[14px]' : 'translate-x-[2px]',
        ].join(' ')}
      />
    </button>
  )
}
