import { useState, type ReactNode } from 'react'
import { CheckIcon, LinkIcon } from '../icons'

interface Props {
  path: string
  hashPrefix?: string
  children: ReactNode
}

/**
 * Wraps a form row with an anchor id (``data-field-path``) so deep-links
 * and the field search omnibox can scroll to it, plus a tiny copy-anchor
 * button that appears on hover.
 */
export function FieldAnchor({ path, hashPrefix = '', children }: Props) {
  const [copied, setCopied] = useState(false)

  const hash = `${hashPrefix ? hashPrefix + ':' : ''}${path}`
  async function copy() {
    const href = `${window.location.origin}${window.location.pathname}#${hash}`
    try {
      await navigator.clipboard.writeText(href)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* ignore clipboard failure */
    }
  }

  return (
    <div data-field-path={path} className="group relative scroll-mt-24">
      {children}
      <button
        type="button"
        onClick={copy}
        aria-label={copied ? `Copied link for ${path}` : `Copy link for ${path}`}
        title="Copy link to this field"
        className="absolute top-0 -right-1 opacity-0 group-hover:opacity-100 transition text-ink-4 hover:text-ink-2"
      >
        {copied ? (
          <CheckIcon className="w-3 h-3 text-ok" />
        ) : (
          <LinkIcon className="w-3 h-3" />
        )}
      </button>
    </div>
  )
}
