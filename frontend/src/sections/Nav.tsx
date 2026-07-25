import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Menu, X, Rocket } from 'lucide-react'
import { GlassButton } from '../components/ui/GlassButton'
import { LOGIN_URL, DEMO_MAILTO, cn } from '../lib/utils'

const LINKS = [
  { label: 'How it works', href: '#how' },
  { label: 'Platform', href: '#capabilities' },
  { label: 'AI Insights', href: '#ai' },
  { label: 'Pricing', href: '#pricing' },
  { label: 'Rollout', href: '#rollout' },
]

export function Nav() {
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header className="fixed inset-x-0 top-0 z-50">
      <div
        className={cn(
          'mx-auto flex h-16 max-w-6xl items-center justify-between px-5 sm:px-8 transition-all duration-300',
          scrolled && 'mt-2 sm:mt-3',
        )}
      >
        <div
          className={cn(
            // Always frosted glass (lighter 16px navbar blur), not just on scroll.
            'flex w-full items-center justify-between rounded-full px-4 py-2 transition-all duration-300 glass-nav',
          )}
        >
          {/* logo */}
          <a href="#top" className="flex items-center gap-2.5 pl-1">
            <span className="relative grid h-8 w-8 place-items-center rounded-xl bg-[#14141a]">
              <Rocket size={17} className="text-white" />
            </span>
            <span className="text-[17px] font-bold tracking-tight text-fg">
              Rocket<span className="text-fg-muted">Attendance</span>
            </span>
          </a>

          {/* desktop links */}
          <nav className="hidden items-center gap-1 lg:flex">
            {LINKS.map((l) => (
              <a
                key={l.href}
                href={l.href}
                className="rounded-full px-4 py-2 text-sm font-medium text-fg-muted transition-colors hover:bg-black/[0.04] hover:text-fg"
              >
                {l.label}
              </a>
            ))}
          </nav>

          {/* desktop actions */}
          <div className="hidden items-center gap-2 md:flex">
            <GlassButton href={LOGIN_URL} variant="glass" size="sm">
              Log in
            </GlassButton>
            <GlassButton href={DEMO_MAILTO} variant="primary" size="sm">
              Book a demo
            </GlassButton>
          </div>

          {/* mobile toggle */}
          <button
            type="button"
            className="grid h-10 w-10 place-items-center rounded-full text-fg md:hidden"
            onClick={() => setOpen((v) => !v)}
            aria-label="Toggle menu"
          >
            {open ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>

      {/* mobile drawer */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="mx-5 mt-2 md:hidden"
          >
            <div className="glass-strong rounded-3xl p-4">
              <nav className="flex flex-col">
                {LINKS.map((l) => (
                  <a
                    key={l.href}
                    href={l.href}
                    onClick={() => setOpen(false)}
                    className="rounded-xl px-4 py-3 text-fg-muted hover:bg-black/[0.04] hover:text-fg"
                  >
                    {l.label}
                  </a>
                ))}
              </nav>
              <div className="mt-3 flex flex-col gap-2">
                <GlassButton href={LOGIN_URL} variant="glass" size="md" className="w-full">
                  Log in
                </GlassButton>
                <GlassButton href={DEMO_MAILTO} variant="primary" size="md" className="w-full">
                  Book a demo
                </GlassButton>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
