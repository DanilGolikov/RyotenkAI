import { useSyncExternalStore } from 'react'

interface Store {
  open: boolean
  setOpen: (value: boolean) => void
  toggle: () => void
}

let openState = false
const listeners = new Set<() => void>()

function notify() {
  listeners.forEach((fn) => fn())
}

const store: Store = {
  get open() {
    return openState
  },
  setOpen(value: boolean) {
    if (openState === value) return
    openState = value
    notify()
  },
  toggle() {
    openState = !openState
    notify()
  },
}

function subscribe(fn: () => void) {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}
function getSnapshot() {
  return openState
}

export function useCommandPalette(): Store & { isOpen: boolean } {
  const isOpen = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  return { ...store, open: isOpen, isOpen }
}
