import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Users, MapPinned, LayoutGrid, Sparkles, Check, AlertTriangle } from 'lucide-react'
import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal } from '../components/ui/Reveal'
import { cn } from '../lib/utils'

const TABS = [
  { key: 'attendance', label: 'Attendance', icon: Users },
  { key: 'geofence', label: 'Geofence', icon: MapPinned },
  { key: 'distribution', label: 'Distribution', icon: LayoutGrid },
  { key: 'ai', label: 'AI Insights', icon: Sparkles },
] as const

type TabKey = (typeof TABS)[number]['key']

export function ProductShowcase() {
  const [active, setActive] = useState<TabKey>('attendance')

  return (
    <Section id="showcase">
      <SectionHeading
        align="center"
        eyebrow="One console · every worksite"
        title="See the whole workforce without leaving the desk"
        lede="A live operations console built for owners and HR — filtered, exportable and board-ready."
      />

      {/* tab bar */}
      <Reveal className="mt-10 flex justify-center">
        <div className="glass inline-flex flex-wrap justify-center gap-1 rounded-full p-1.5">
          {TABS.map((t) => {
            const Icon = t.icon
            const on = active === t.key
            return (
              <button
                key={t.key}
                type="button"
                onClick={() => setActive(t.key)}
                className={cn(
                  'relative flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors',
                  on ? 'text-white' : 'text-fg-muted hover:text-fg',
                )}
              >
                {on && (
                  <motion.span
                    layoutId="showcase-pill"
                    className="absolute inset-0 rounded-full bg-[#14141a]"
                    transition={{ type: 'spring', stiffness: 400, damping: 34 }}
                  />
                )}
                <Icon size={15} className="relative z-10" />
                <span className="relative z-10">{t.label}</span>
              </button>
            )
          })}
        </div>
      </Reveal>

      {/* panel */}
      <Reveal delay={0.1} className="mt-8">
        <GlassCard strong className="overflow-hidden p-2">
          {/* window chrome */}
          <div className="flex items-center gap-2 px-4 py-3">
            <span className="h-3 w-3 rounded-full bg-alert/70" />
            <span className="h-3 w-3 rounded-full bg-amber/70" />
            <span className="h-3 w-3 rounded-full bg-fence/70" />
            <span className="ml-3 font-mono text-xs text-fg-faint">
              app.rocketattendance.com / {active}
            </span>
          </div>
          <div className="rounded-2xl bg-black/[0.02] p-4 md:p-6">
            <AnimatePresence mode="wait">
              <motion.div
                key={active}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.3 }}
              >
                {active === 'attendance' && <AttendancePanel />}
                {active === 'geofence' && <GeofencePanel />}
                {active === 'distribution' && <DistributionPanel />}
                {active === 'ai' && <AIPanel />}
              </motion.div>
            </AnimatePresence>
          </div>
        </GlassCard>
      </Reveal>
    </Section>
  )
}

function KpiRow() {
  const kpis = [
    { label: 'Present', value: '1,842', tone: 'text-fence' },
    { label: 'Absent', value: '96', tone: 'text-alert' },
    { label: 'Active sites', value: '12', tone: 'text-fg' },
    { label: 'Alerts', value: '3', tone: 'text-amber-glow' },
  ]
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      {kpis.map((k) => (
        <div key={k.label} className="rounded-xl bg-black/[0.04] p-4">
          <p className="text-xs text-fg-faint">{k.label}</p>
          <p className={cn('mt-1 text-2xl font-bold', k.tone)}>{k.value}</p>
        </div>
      ))}
    </div>
  )
}

function AttendancePanel() {
  const rows = [
    { name: 'R. Kumar', trade: 'Carpenter Finishing', site: 'Vida', ok: true },
    { name: 'A. Hossain', trade: 'Steel Fixer', site: 'Opal Garden', ok: true },
    { name: 'M. Farooq', trade: 'Scaffolder', site: 'GRC W/Shop', ok: true },
    { name: 'S. Thapa', trade: 'Electrician', site: 'Site Marina', ok: false },
  ]
  return (
    <div className="flex flex-col gap-4">
      <KpiRow />
      <div className="overflow-hidden rounded-xl border border-black/8">
        {rows.map((r, i) => (
          <div
            key={r.name}
            className={cn(
              'grid grid-cols-[1.2fr_1.4fr_1fr_auto] items-center gap-3 px-4 py-3 text-sm',
              i % 2 && 'bg-black/[0.03]',
            )}
          >
            <span className="font-medium text-fg">{r.name}</span>
            <span className="text-fg-muted">{r.trade}</span>
            <span className="text-fg-faint">{r.site}</span>
            {r.ok ? (
              <span className="flex items-center gap-1 text-fence"><Check size={14} /> in</span>
            ) : (
              <span className="flex items-center gap-1 text-alert"><AlertTriangle size={14} /> off-site</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function GeofencePanel() {
  return (
    <div className="grid gap-4 md:grid-cols-[1.4fr_1fr]">
      <div className="relative h-56 overflow-hidden rounded-xl bg-gradient-to-br from-[#eceef1] to-[#f6f7f9]">
        {/* fake polygon */}
        <svg viewBox="0 0 300 200" className="absolute inset-0 h-full w-full">
          <polygon
            points="60,40 220,30 250,120 150,170 50,130"
            fill="rgba(17,17,26,0.05)"
            stroke="rgba(17,17,26,0.5)"
            strokeWidth="2"
          />
          {[[110, 80], [170, 70], [140, 120], [90, 110]].map(([x, y], i) => (
            <circle key={i} cx={x} cy={y} r="4" fill="#14141a" />
          ))}
          <circle cx="265" cy="150" r="5" fill="#9a9aa2" />
        </svg>
        <span className="absolute bottom-3 left-3 rounded-full bg-white/70 px-3 py-1 font-mono text-[11px] text-fg-muted">
          Site Vida · KML boundary
        </span>
      </div>
      <div className="flex flex-col gap-3">
        <div className="rounded-xl bg-fence/10 p-4">
          <p className="text-sm font-semibold text-fence">312 inside boundary</p>
          <p className="text-xs text-fg-faint">avg 45m from edge</p>
        </div>
        <div className="rounded-xl bg-alert/10 p-4">
          <p className="text-sm font-semibold text-alert">1 out-of-bounds attempt</p>
          <p className="text-xs text-fg-faint">Site Marina · 06:04 · submit blocked</p>
        </div>
        <div className="rounded-xl bg-black/[0.04] p-4">
          <p className="text-sm font-semibold text-fg">Smart-tuning</p>
          <p className="text-xs text-fg-faint">42 false alerts suppressed this week</p>
        </div>
      </div>
    </div>
  )
}

function DistributionPanel() {
  const rows = [
    { trade: 'Carpenter', a: 18, b: 4, c: 0 },
    { trade: 'Steel Fixer', a: 12, b: 9, c: 6 },
    { trade: 'Electrician', a: 8, b: 14, c: 3 },
    { trade: 'Mason', a: 22, b: 7, c: 11 },
  ]
  const max = 22
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-[1fr_auto] px-1 text-xs text-fg-faint">
        <span>Trade × Site</span>
        <span>Vida · Opal · Marina</span>
      </div>
      {rows.map((r) => (
        <div key={r.trade} className="rounded-xl bg-black/[0.04] p-3">
          <div className="mb-2 flex justify-between text-sm">
            <span className="font-medium text-fg">{r.trade}</span>
            <span className="text-fg-faint">{r.a + r.b + r.c} total</span>
          </div>
          <div className="flex gap-2">
            {[
              { v: r.a, c: 'from-brand to-brand-2' },
              { v: r.b, c: 'from-cyan to-brand' },
              { v: r.c, c: 'from-amber to-amber-glow' },
            ].map((seg, i) => (
              <div key={i} className="h-2.5 overflow-hidden rounded-full bg-black/[0.04]" style={{ flex: max }}>
                <motion.div
                  className={cn('h-full rounded-full bg-gradient-to-r', seg.c)}
                  initial={{ width: 0 }}
                  whileInView={{ width: `${(seg.v / max) * 100}%` }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.8, ease: 'easeOut' }}
                />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function AIPanel() {
  const bullets = [
    '4 visas expiring this week — HR queue updated',
    'Vida is short 12 carpenters vs. plan',
    '9% absence spike at Opal Garden (Tue)',
    'Move 12 carpenters GRC W/Shop → Vida',
  ]
  return (
    <div className="grid gap-4 md:grid-cols-[1.2fr_1fr]">
      <div className="rounded-xl bg-gradient-to-br from-brand-2/15 to-cyan/5 p-5">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-brand-2">
          <Sparkles size={16} /> Daily executive summary
        </div>
        <ul className="flex flex-col gap-2.5">
          {bullets.map((b) => (
            <li key={b} className="flex items-start gap-2 text-sm text-fg-muted">
              <Check size={15} className="mt-0.5 shrink-0 text-fence" /> {b}
            </li>
          ))}
        </ul>
      </div>
      <div className="flex flex-col gap-3">
        <div className="rounded-xl bg-black/[0.04] p-4">
          <p className="text-xs text-fg-faint">Attrition risk · top flag</p>
          <p className="mt-1 text-sm font-semibold text-fg">K. Ali — score 82 / 100</p>
          <p className="text-xs text-fg-faint">attendance drop + stale salary</p>
        </div>
        <div className="rounded-xl bg-black/[0.04] p-4">
          <p className="text-xs text-fg-faint">Ask the data</p>
          <p className="mt-1 text-sm text-fg-muted">
            "Electricians on leave at Vida this week?" →{' '}
            <span className="font-semibold text-fg">3</span>
          </p>
        </div>
      </div>
    </div>
  )
}
