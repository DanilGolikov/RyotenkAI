/**
 * Centralised SVG icon set. Per FRONTEND_GUIDELINES.md: no emoji in
 * UI chrome — they render inconsistently across OS fonts, size weirdly
 * with surrounding text, and don't inherit `currentColor`.
 *
 * Every icon here uses `currentColor` and `aria-hidden`, so they
 * inherit whatever text colour the parent sets and stay out of the
 * accessibility tree. Size via Tailwind `w-4 h-4` (etc.) at the call
 * site rather than inside the SVG, so the same icon can be small in a
 * row or larger on a hero header.
 */
import type { SVGProps } from 'react'

const baseProps: SVGProps<SVGSVGElement> = {
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
}

export function LinkIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M6.5 9.5l3-3" />
      <path d="M7 4.5l1-1a2.5 2.5 0 013.5 3.5l-1 1" />
      <path d="M9 11.5l-1 1a2.5 2.5 0 01-3.5-3.5l1-1" />
    </svg>
  )
}

export function CheckIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="3,9 6.5,12 13,5" />
    </svg>
  )
}

export function ChevronRightIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="6,3 11,8 6,13" />
    </svg>
  )
}

export function InfoIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg {...baseProps} {...props}>
      <circle cx="8" cy="8" r="6" />
      <line x1="8" y1="7.5" x2="8" y2="11.5" />
      <circle cx="8" cy="5" r="0.5" fill="currentColor" />
    </svg>
  )
}

export function AlertIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg {...baseProps} {...props}>
      <path d="M8 2l6.5 11h-13z" />
      <line x1="8" y1="6" x2="8" y2="9.5" />
      <circle cx="8" cy="11.5" r="0.5" fill="currentColor" />
    </svg>
  )
}

export function ExpandIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="3,6 3,3 6,3" />
      <polyline points="10,3 13,3 13,6" />
      <polyline points="13,10 13,13 10,13" />
      <polyline points="6,13 3,13 3,10" />
    </svg>
  )
}

export function CollapseIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg {...baseProps} {...props}>
      <polyline points="6,3 6,6 3,6" />
      <polyline points="13,6 10,6 10,3" />
      <polyline points="10,13 10,10 13,10" />
      <polyline points="3,10 6,10 6,13" />
    </svg>
  )
}
