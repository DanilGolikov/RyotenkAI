import { useState, type ReactNode } from 'react'
import { HelpTooltip } from './HelpTooltip'

export function FormGroup({
  title,
  description,
  required = false,
  defaultOpen = true,
  collapsible = true,
  helpText,
  children,
  badge,
}: {
  title: string
  description?: string
  required?: boolean
  defaultOpen?: boolean
  /** When false, renders a static header + always-open content. */
  collapsible?: boolean
  /** Text for the "?" tooltip next to the title (separate from description
   *  which is shown inline as subtitle when collapsible=true). */
  helpText?: string
  children: ReactNode
  badge?: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  const isOpen = collapsible ? open : true

  const headerInner = (
    <>
      {collapsible && (
        <span
          className={`text-ink-3 text-2xs w-3 ${isOpen ? 'rotate-90' : ''} transition-transform`}
          aria-hidden
        >
          ▶
        </span>
      )}
      <span className="text-sm font-medium text-ink-1">{title}</span>
      <HelpTooltip text={helpText} />
      {required && <span className="text-[0.6rem] text-brand-warm">required</span>}
      {badge}
      {description && (
        <span className="ml-2 text-2xs text-ink-3 truncate flex-1">{description}</span>
      )}
    </>
  )

  return (
    <section className="rounded-md border border-line-1 bg-surface-1 overflow-hidden">
      {collapsible ? (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-2 transition"
        >
          {headerInner}
        </button>
      ) : (
        <div className="w-full flex items-center gap-2 px-3 py-2 bg-surface-2/40">
          {headerInner}
        </div>
      )}
      {isOpen && <div className="border-t border-line-1 p-4 space-y-4">{children}</div>}
    </section>
  )
}
