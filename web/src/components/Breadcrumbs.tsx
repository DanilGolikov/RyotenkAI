import { Fragment } from 'react'
import { Link } from 'react-router-dom'
import { ChevronRightIcon } from './icons'

export interface Crumb {
  label: string
  to?: string
}

/**
 * Simple breadcrumb trail. Last crumb is rendered as plain text (not a
 * link) per WAI-ARIA authoring practice: the current page doesn't need
 * to be navigable. Chevrons carry `aria-hidden` so screen readers
 * don't read the separator character.
 */
export function Breadcrumbs({ items }: { items: Crumb[] }) {
  if (items.length === 0) return null
  return (
    <nav aria-label="Breadcrumb" className="text-2xs text-ink-3 select-none">
      <ol className="flex items-center gap-1 flex-wrap">
        {items.map((c, i) => {
          const isLast = i === items.length - 1
          return (
            <Fragment key={i}>
              <li className="inline-flex items-center">
                {c.to && !isLast ? (
                  <Link
                    to={c.to}
                    className="hover:text-ink-1 transition-colors truncate max-w-[220px]"
                  >
                    {c.label}
                  </Link>
                ) : (
                  <span
                    className="text-ink-2 truncate max-w-[320px]"
                    aria-current={isLast ? 'page' : undefined}
                  >
                    {c.label}
                  </span>
                )}
              </li>
              {!isLast && (
                <ChevronRightIcon className="w-3 h-3 text-ink-4 shrink-0" />
              )}
            </Fragment>
          )
        })}
      </ol>
    </nav>
  )
}
