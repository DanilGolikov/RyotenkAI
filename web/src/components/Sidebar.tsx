import { NavLink } from 'react-router-dom'
import type { ReactNode } from 'react'

type Item = { to: string; label: string; icon: ReactNode; end?: boolean }

const ICON_CLS = 'w-4 h-4'

const items: Item[] = [
  {
    to: '/',
    end: true,
    label: 'Overview',
    icon: (
      <svg className={ICON_CLS} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 12l9-9 9 9" /><path d="M5 10v10h14V10" />
      </svg>
    ),
  },
  {
    to: '/runs',
    label: 'Runs',
    icon: (
      <svg className={ICON_CLS} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="5" width="18" height="4" rx="1" />
        <rect x="3" y="11" width="12" height="4" rx="1" />
        <rect x="3" y="17" width="16" height="4" rx="1" />
      </svg>
    ),
  },
  {
    to: '/launch',
    label: 'Launch',
    icon: (
      <svg className={ICON_CLS} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M5 12h14" /><path d="M12 5l7 7-7 7" />
      </svg>
    ),
  },
]

export function Sidebar() {
  return (
    <aside className="w-[240px] shrink-0 bg-gradient-sidebar border-r border-line-1 flex flex-col">
      <div className="px-5 py-5 flex items-center gap-2.5">
        <div className="w-7 h-7 rounded bg-gradient-brand shadow-glow-brand" />
        <div className="leading-tight">
          <div className="text-sm font-semibold">
            <span className="gradient-text">Ryotenk</span>
            <span className="text-ink-1">AI</span>
          </div>
          <div className="text-2xs text-ink-3">pipeline control plane</div>
        </div>
      </div>

      <nav className="flex-1 px-3 space-y-0.5">
        {items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => [isActive ? 'nav-item nav-item-active' : 'nav-item'].join(' ')}
          >
            {({ isActive }) => (
              <>
                <span className={isActive ? 'text-brand' : 'text-ink-3'}>{item.icon}</span>
                <span>{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="px-4 pb-4 text-2xs text-ink-3 space-y-1">
        <div className="flex justify-between">
          <span>docs</span>
          <a href="/docs" target="_blank" rel="noreferrer" className="text-ink-2 hover:text-ink-1">OpenAPI</a>
        </div>
        <div className="flex justify-between">
          <span>hotkey</span>
          <span className="flex gap-0.5"><kbd className="kbd">⌘</kbd><kbd className="kbd">K</kbd></span>
        </div>
      </div>
    </aside>
  )
}
