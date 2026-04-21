import { useEffect, useState, type PropsWithChildren } from 'react'

/**
 * Narrow-viewport block for the app. This is a desktop dev-tool — the
 * form grid is single-purpose on wide screens and degrades badly on
 * phones (sidebar can't collapse, label column + input column + nested
 * aside all compete for the same 375 CSS px). Rather than invent a
 * mobile layout we don't plan to support, we render a plain message.
 *
 * Threshold 1024 CSS px matches the conventional `lg` breakpoint.
 */
const MIN_WIDTH_PX = 1024

function useIsWideEnough(minWidth: number): boolean {
  const [wide, setWide] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    return window.matchMedia(`(min-width: ${minWidth}px)`).matches
  })

  useEffect(() => {
    const mql = window.matchMedia(`(min-width: ${minWidth}px)`)
    const onChange = (e: MediaQueryListEvent) => setWide(e.matches)
    // Safari < 14 used addListener; matchMedia.addEventListener is fine
    // for Chrome/Edge/Firefox/Safari 14+, which covers any desktop dev
    // running this tool.
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [minWidth])

  return wide
}

export function DesktopOnlyGate({ children }: PropsWithChildren) {
  const wide = useIsWideEnough(MIN_WIDTH_PX)
  if (wide) return <>{children}</>
  return (
    <div className="min-h-screen flex items-center justify-center p-6 text-center">
      <div className="max-w-md space-y-2">
        <div className="text-ink-1 text-lg font-semibold">Use a wider screen</div>
        <p className="text-ink-3 text-sm leading-relaxed">
          RyotenkAI&apos;s pipeline control plane is a desktop dev tool. The
          config form and run explorer need at least{' '}
          <span className="text-ink-1 font-medium">{MIN_WIDTH_PX}px</span> of
          width to stay usable. Please resize this window or open on a
          larger display.
        </p>
      </div>
    </div>
  )
}
