import { NavLink } from 'react-router-dom'
import type { ReactNode } from 'react'
import { useSidebarCollapsed } from '../hooks/useSidebarCollapsed'

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
  {
    to: '/projects',
    label: 'Projects',
    icon: (
      <svg className={ICON_CLS} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 7l3-3h5l2 2h8v13H3z" />
      </svg>
    ),
  },
  {
    to: '/settings',
    label: 'Settings',
    icon: (
      <svg className={ICON_CLS} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </svg>
    ),
  },
]

const ChevronLeft = (
  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M15 6l-6 6 6 6" />
  </svg>
)

const ChevronRight = (
  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M9 6l6 6-6 6" />
  </svg>
)

export function Sidebar() {
  const { collapsed, toggle } = useSidebarCollapsed()

  return (
    <aside
      data-collapsed={collapsed}
      className={[
        'shrink-0 bg-surface-1 border-r border-line-1 flex flex-col',
        'transition-[width] duration-150 ease-out',
        collapsed ? 'w-[60px]' : 'w-[240px]',
      ].join(' ')}
    >
      {/* Brand — icon always pinned at left (px-4), label block stays in
          the DOM but animates opacity + max-width when collapsed. Keeps
          layout stable during rapid toggles so icons don't jitter from
          the mount/unmount churn that would otherwise reflow the row. */}
      <div className="py-5 px-4 flex items-center gap-2.5">
        <div
          aria-label="RyotenkAI"
          className="w-7 h-7 shrink-0 rounded bg-gradient-brand shadow-glow-brand"
        />
        <div
          aria-hidden={collapsed}
          className={[
            'leading-tight overflow-hidden transition-[max-width,opacity] duration-150 ease-out',
            collapsed ? 'max-w-0 opacity-0' : 'max-w-[180px] opacity-100',
          ].join(' ')}
        >
          <div className="text-sm font-semibold whitespace-nowrap">
            <span className="text-ink-1">Ryotenk</span>
            <span className="gradient-text">AI</span>
          </div>
          <div className="text-2xs text-ink-3 whitespace-nowrap">pipeline control plane</div>
        </div>
      </div>

      {/* Nav — icons stay at a constant x-offset so the collapse animation
          doesn't jitter them. Labels fade/collapse width rather than
          unmounting so the row doesn't reflow mid-transition. */}
      <nav className="flex-1 px-2 space-y-0.5">
        {items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            title={collapsed ? item.label : undefined}
            className={({ isActive }) =>
              isActive ? 'nav-item nav-item-active' : 'nav-item'
            }
          >
            {({ isActive }) => (
              <>
                <span className={isActive ? 'text-brand' : 'text-ink-3'}>{item.icon}</span>
                <span
                  className={[
                    'whitespace-nowrap overflow-hidden transition-[max-width,opacity] duration-150 ease-out',
                    collapsed ? 'max-w-0 opacity-0' : 'max-w-[140px] opacity-100',
                  ].join(' ')}
                >
                  {item.label}
                </span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer: meta fades; collapse toggle keeps icon pinned left. */}
      <div className="pb-4 px-2 space-y-2">
        <div
          aria-hidden={collapsed}
          className={[
            'text-2xs text-ink-3 space-y-1 px-2 overflow-hidden transition-[max-height,opacity] duration-150 ease-out',
            collapsed ? 'max-h-0 opacity-0' : 'max-h-16 opacity-100',
          ].join(' ')}
        >
          <div className="flex justify-between">
            <span>docs</span>
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="text-ink-2 hover:text-ink-1"
            >
              OpenAPI
            </a>
          </div>
          <div className="flex justify-between">
            <span>palette</span>
            <span className="flex gap-0.5"><kbd className="kbd">⌘</kbd><kbd className="kbd">K</kbd></span>
          </div>
        </div>

        <button
          type="button"
          onClick={toggle}
          title={collapsed ? 'Expand sidebar (⌘B)' : 'Collapse sidebar (⌘B)'}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="w-full flex items-center justify-between rounded-md text-2xs text-ink-3 h-8 px-2.5 border border-line-1 bg-surface-1 hover:text-ink-1 hover:border-line-2 transition-colors"
        >
          <span className="flex items-center gap-1.5 min-w-0">
            {collapsed ? ChevronRight : ChevronLeft}
            <span
              className={[
                'whitespace-nowrap overflow-hidden transition-[max-width,opacity] duration-150 ease-out',
                collapsed ? 'max-w-0 opacity-0' : 'max-w-[80px] opacity-100',
              ].join(' ')}
            >
              Collapse
            </span>
          </span>
          <span
            className={[
              'flex gap-0.5 overflow-hidden transition-[max-width,opacity] duration-150 ease-out',
              collapsed ? 'max-w-0 opacity-0' : 'max-w-[60px] opacity-100',
            ].join(' ')}
          >
            <kbd className="kbd">⌘</kbd><kbd className="kbd">B</kbd>
          </span>
        </button>
      </div>
    </aside>
  )
}
