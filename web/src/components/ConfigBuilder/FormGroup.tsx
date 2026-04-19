import { useState, type ReactNode } from 'react'

export function FormGroup({
  title,
  description,
  required = false,
  defaultOpen = true,
  children,
  badge,
}: {
  title: string
  description?: string
  required?: boolean
  defaultOpen?: boolean
  children: ReactNode
  badge?: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="rounded-md border border-line-1 bg-surface-1 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-2 transition"
      >
        <span className={`text-ink-3 text-2xs w-3 ${open ? 'rotate-90' : ''} transition-transform`}>
          ▶
        </span>
        <span className="text-sm font-medium text-ink-1">{title}</span>
        {required && <span className="text-[0.6rem] text-brand">required</span>}
        {badge}
        {description && (
          <span className="ml-2 text-2xs text-ink-3 truncate flex-1">{description}</span>
        )}
      </button>
      {open && <div className="border-t border-line-1 p-4 space-y-3">{children}</div>}
    </section>
  )
}
