import type { PropsWithChildren, ReactNode } from 'react'

export function Card({
  children,
  className = '',
  raised = false,
  padding = 'p-4',
}: PropsWithChildren<{ className?: string; raised?: boolean; padding?: string }>) {
  return (
    <div className={`${raised ? 'card-raised' : 'card'} ${padding} ${className}`}>{children}</div>
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
      <div>
        <h2 className="text-sm font-medium text-ink">{title}</h2>
        {subtitle && <div className="text-2xs text-ink-3 mt-0.5">{subtitle}</div>}
      </div>
      {action}
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
