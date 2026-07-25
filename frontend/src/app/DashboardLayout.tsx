import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Rocket, Menu, X, LogOut, Lock, ChevronRight } from 'lucide-react'
import { useMe } from '../lib/hooks/useMe'
import { visibleNav, NAV_GROUPS } from './nav'
import { GradientBackdrop } from '../components/ui/GradientBackdrop'
import { cn } from '../lib/utils'

/** Where the browser goes to log out (Django clears the session cookie). */
const LOGOUT_URL = '/logout/'

/**
 * The authenticated dashboard shell: fixed glass sidebar + header, with the
 * active page rendered through `<Outlet/>`. Guards access via `/me` (redirects
 * to login on 401, handled in the API client) and shows a read-only banner for
 * viewer accounts.
 */
export default function DashboardLayout() {
  const { data: me, isLoading, isError } = useMe()
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  if (isLoading) return <FullscreenState label="Loading your console…" />
  // On 401/403 the API client has already redirected; anything else here means
  // the session couldn't be resolved — send them to log in.
  if (isError || !me) return <FullscreenState label="Redirecting to sign in…" />

  const groups = visibleNav(me)
  const currentTitle = titleForPath(location.pathname)

  return (
    <div className="relative min-h-screen text-fg">
      <GradientBackdrop />

      {/* ---- Sidebar (desktop) ---- */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-[264px] flex-col p-3 lg:flex">
        <SidebarInner me={me} groups={groups} />
      </aside>

      {/* ---- Sidebar (mobile drawer) ---- */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              className="fixed inset-0 z-40 bg-black/30 lg:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileOpen(false)}
            />
            <motion.aside
              className="fixed inset-y-0 left-0 z-50 flex w-[280px] flex-col p-3 lg:hidden"
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: 'spring', stiffness: 400, damping: 40 }}
            >
              <SidebarInner me={me} groups={groups} onNavigate={() => setMobileOpen(false)} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* ---- Main column ---- */}
      <div className="flex min-h-screen flex-col lg:pl-[264px]">
        {/* header */}
        <header className="sticky top-0 z-30 flex h-16 items-center gap-3 px-4 sm:px-6">
          <div className="glass-nav flex h-12 w-full items-center gap-3 rounded-2xl px-3">
            <button
              type="button"
              onClick={() => setMobileOpen(true)}
              className="grid h-9 w-9 place-items-center rounded-xl text-fg hover:bg-black/[0.05] lg:hidden"
              aria-label="Open menu"
            >
              <Menu size={20} />
            </button>
            <h1 className="text-[15px] font-semibold text-fg">{currentTitle}</h1>
            <div className="ml-auto flex items-center gap-2">
              <span className="hidden text-sm text-fg-muted sm:inline">{me.username}</span>
              <span className="grid h-8 w-8 place-items-center rounded-full bg-[#14141a] text-xs font-bold text-white">
                {me.username.slice(0, 2).toUpperCase()}
              </span>
            </div>
          </div>
        </header>

        {/* read-only banner */}
        {me.is_readonly && (
          <div className="mx-4 mt-1 flex items-center gap-2 rounded-xl border border-black/10 bg-black/[0.03] px-4 py-2.5 text-sm text-fg-muted sm:mx-6">
            <Lock size={15} /> Read-only access — you can view data but not make changes.
          </div>
        )}

        {/* page content */}
        <main className="flex-1 px-4 py-5 sm:px-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

function SidebarInner({
  me,
  groups,
  onNavigate,
}: {
  me: { username: string; role: string }
  groups: ReturnType<typeof visibleNav>
  onNavigate?: () => void
}) {
  return (
    <div className="glass-strong flex h-full flex-col rounded-3xl p-4">
      {/* brand */}
      <a href="/dashboard-v2" className="mb-5 flex items-center gap-2.5 px-1">
        <span className="grid h-8 w-8 place-items-center rounded-xl bg-[#14141a]">
          <Rocket size={17} className="text-white" />
        </span>
        <span className="text-[16px] font-bold tracking-tight">
          Rocket<span className="text-fg-muted">Attendance</span>
        </span>
      </a>

      {/* nav */}
      <nav className="flex-1 space-y-5 overflow-y-auto pr-1">
        {groups.map((group) => (
          <div key={group.label}>
            <p className="mb-1.5 px-2 text-[11px] font-semibold uppercase tracking-wider text-fg-faint">
              {group.label}
            </p>
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const Icon = item.icon
                return (
                  <NavLink
                    key={item.key}
                    to={item.to}
                    end={item.to === '/dashboard-v2'}
                    onClick={onNavigate}
                    className={({ isActive }) =>
                      cn(
                        'flex items-center gap-2.5 rounded-xl px-2.5 py-2 text-sm font-medium transition-colors',
                        isActive
                          ? 'bg-[#14141a] text-white'
                          : 'text-fg-muted hover:bg-black/[0.05] hover:text-fg',
                      )
                    }
                  >
                    <Icon size={17} className="shrink-0" />
                    {item.label}
                  </NavLink>
                )
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* user + logout */}
      <div className="mt-4 border-t border-black/10 pt-3">
        <div className="flex items-center gap-2.5 px-1">
          <span className="grid h-9 w-9 place-items-center rounded-full bg-[#14141a] text-xs font-bold text-white">
            {me.username.slice(0, 2).toUpperCase()}
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold text-fg">{me.username}</p>
            <p className="text-xs text-fg-faint">{roleLabel(me.role)}</p>
          </div>
          <a
            href={LOGOUT_URL}
            className="grid h-8 w-8 place-items-center rounded-lg text-fg-muted hover:bg-black/[0.05] hover:text-fg"
            aria-label="Sign out"
            title="Sign out"
          >
            <LogOut size={16} />
          </a>
        </div>
      </div>
    </div>
  )
}

function FullscreenState({ label }: { label: string }) {
  return (
    <div className="relative grid min-h-screen place-items-center text-fg">
      <GradientBackdrop />
      <div className="flex items-center gap-3 text-fg-muted">
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-black/20 border-t-black/60" />
        {label}
      </div>
    </div>
  )
}

function roleLabel(role: string): string {
  if (role === 'superuser') return 'Super Admin'
  if (role === 'viewer') return 'Read-only Viewer'
  if (role === 'site_admin') return 'Site Admin'
  return 'Admin'
}

/** Human title for the current path, from the nav model (fallback: "Dashboard"). */
function titleForPath(pathname: string): string {
  let best: { label: string; len: number } | null = null
  for (const group of NAV_GROUPS) {
    for (const item of group.items) {
      if (pathname === item.to || pathname.startsWith(item.to + '/')) {
        if (!best || item.to.length > best.len) best = { label: item.label, len: item.to.length }
      }
    }
  }
  return best?.label ?? 'Dashboard'
}

// re-exported for convenience in breadcrumbs later
export { ChevronRight }
