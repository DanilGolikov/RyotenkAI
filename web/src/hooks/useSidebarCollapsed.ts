import { useEffect, useSyncExternalStore } from 'react'

const STORAGE_KEY = 'ryotenkai:sidebar-collapsed'
const AUTO_COLLAPSE_BREAKPOINT = 1024

function readInitial(): boolean {
  if (typeof window === 'undefined') return false
  const stored = window.localStorage.getItem(STORAGE_KEY)
  if (stored === '1') return true
  if (stored === '0') return false
  // First-load heuristic: auto-collapse on narrow viewports.
  return window.innerWidth < AUTO_COLLAPSE_BREAKPOINT
}

let collapsed = readInitial()
const listeners = new Set<() => void>()

function notify() {
  listeners.forEach((fn) => fn())
}

export function setSidebarCollapsed(next: boolean) {
  if (collapsed === next) return
  collapsed = next
  try {
    window.localStorage.setItem(STORAGE_KEY, next ? '1' : '0')
  } catch {
    /* private browsing / quota — ignore */
  }
  notify()
}

export function toggleSidebar() {
  setSidebarCollapsed(!collapsed)
}

function subscribe(fn: () => void) {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

export function useSidebarCollapsed(): {
  collapsed: boolean
  setCollapsed: (value: boolean) => void
  toggle: () => void
} {
  const value = useSyncExternalStore(
    subscribe,
    () => collapsed,
    () => false,
  )

  // Global Cmd/Ctrl+B hotkey — register once.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const mod = event.metaKey || event.ctrlKey
      if (mod && event.key.toLowerCase() === 'b') {
        // Don't swallow native browser bold inside an input/textarea.
        const target = event.target as HTMLElement | null
        if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
          return
        }
        event.preventDefault()
        toggleSidebar()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return {
    collapsed: value,
    setCollapsed: setSidebarCollapsed,
    toggle: toggleSidebar,
  }
}
