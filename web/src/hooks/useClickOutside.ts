import { useEffect, type RefObject } from 'react'

/**
 * Close a floating element (dropdown, tooltip, popover) when the user
 * interacts outside of it. Listens for `mousedown` (not `click` — click
 * is too late, the element's own click handler has already fired by
 * then) and `Escape`. `enabled` should mirror the element's open state
 * so the document listener is only installed while it's needed.
 *
 * Example:
 *   const ref = useRef<HTMLDivElement>(null)
 *   const [open, setOpen] = useState(false)
 *   useClickOutside(ref, open, () => setOpen(false))
 */
export function useClickOutside<T extends HTMLElement>(
  ref: RefObject<T>,
  enabled: boolean,
  onOutside: () => void,
): void {
  useEffect(() => {
    if (!enabled) return
    function onDocMouseDown(e: MouseEvent) {
      const el = ref.current
      if (!el) return
      if (!el.contains(e.target as Node)) onOutside()
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onOutside()
    }
    document.addEventListener('mousedown', onDocMouseDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [enabled, onOutside, ref])
}
