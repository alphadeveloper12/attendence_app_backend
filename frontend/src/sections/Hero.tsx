import { motion } from 'framer-motion'
import {
  ScanFace,
  MapPin,
  ShieldCheck,
  TrendingUp,
  Check,
  RefreshCw,
  AlertTriangle,
  FileClock,
} from 'lucide-react'
import { GlassCard } from '../components/ui/GlassCard'
import { GlassButton } from '../components/ui/GlassButton'
import { Badge } from '../components/ui/Badge'
import { Marquee } from '../components/ui/Marquee'
import { DEMO_MAILTO, cn } from '../lib/utils'

const TICKER = [
  { icon: Check, tone: 'ok', text: '06:01 · R. Kumar · Carpenter Finishing · Vida' },
  { icon: Check, tone: 'ok', text: '06:01 · A. Hossain · Steel Fixer · Opal Garden' },
  { icon: Check, tone: 'ok', text: '06:02 · M. Farooq · Scaffolder · GRC W/Shop' },
  { icon: Check, tone: 'ok', text: '06:02 · S. Thapa · Electrician · Vida' },
  { icon: RefreshCw, tone: 'sync', text: '06:03 · 14 offline entries synced · Camp 7' },
  { icon: Check, tone: 'ok', text: '06:03 · J. Mendoza · MEP Foreman · Opal Garden' },
  { icon: AlertTriangle, tone: 'bad', text: '06:04 · out-of-bounds attempt · Site Marina' },
  { icon: Check, tone: 'ok', text: '06:04 · P. Raju · Mason · Precast Yard 2' },
  { icon: FileClock, tone: 'warn', text: '06:05 · 4 visas expiring <14 days · HR queue' },
]

const toneColor: Record<string, string> = {
  ok: 'text-fence',
  sync: 'text-cyan',
  bad: 'text-alert',
  warn: 'text-amber-glow',
}

export function Hero() {
  return (
    <section id="hero" className="relative overflow-hidden pt-32 pb-16 md:pt-40">
      <div className="mx-auto grid max-w-6xl items-center gap-12 px-5 sm:px-8 lg:grid-cols-[1.05fr_0.95fr]">
        {/* copy */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-col items-start gap-6"
        >
          <Badge>05:58 first light · muster begins</Badge>
          <h1 className="text-5xl font-extrabold leading-[1.02] md:text-6xl">
            <span className="text-gradient">From face to dashboard</span>
            <br />
            <span className="text-gradient-brand">in 2 seconds.</span>
          </h1>
          <p className="max-w-xl text-lg leading-relaxed text-fg-muted">
            RocketAttendance turns every supervisor's phone into a tamper-proof
            attendance scanner. Face verified on-device, geofence checked at
            capture, visa expiries flagged daily — for 1,000 to 20,000 workers
            across every site you run.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <GlassButton href={DEMO_MAILTO} variant="primary" size="lg">
              Book a demo
            </GlassButton>
            <GlassButton href="#how" variant="glass" size="lg">
              See how it works
            </GlassButton>
          </div>
          <p className="text-sm text-fg-faint">
            Works offline in remote camps · syncs on signal ·{' '}
            <span className="text-fg-muted">no biometric data leaves the phone</span>
          </p>
        </motion.div>

        {/* glass visual */}
        <HeroVisual />
      </div>

      {/* live ticker */}
      <div className="mx-auto mt-14 max-w-6xl px-5 sm:px-8">
        <GlassCard className="px-4 py-3">
          <Marquee duration={44}>
            {TICKER.map((t, i) => {
              const Icon = t.icon
              return (
                <span
                  key={i}
                  className="flex items-center gap-2 whitespace-nowrap rounded-full bg-black/[0.04] px-4 py-1.5 text-sm text-fg-muted"
                >
                  <Icon size={14} className={toneColor[t.tone]} />
                  {t.text}
                </span>
              )
            })}
          </Marquee>
        </GlassCard>
      </div>
    </section>
  )
}

function HeroVisual() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.94 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.9, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
      className="relative mx-auto w-full max-w-md"
    >
      {/* main dashboard card */}
      <GlassCard strong className="relative z-10 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-fg-muted">
            <span className="h-2 w-2 rounded-full bg-fence" /> Live · Site Vida
          </div>
          <span className="font-mono text-xs text-fg-faint">06:04:12</span>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3">
          <MiniStat label="Present" value="313" tone="text-fence" up />
          <MiniStat label="Off-site" value="2" tone="text-alert" />
        </div>

        {/* face scan row */}
        <div className="surface mt-4 flex items-center gap-3 rounded-2xl p-3">
          <div className="relative grid h-12 w-12 shrink-0 place-items-center rounded-xl bg-[#14141a] text-white">
            <ScanFace size={22} />
            <span className="absolute inset-0 rounded-xl ring-1 ring-black/20 animate-pulse-ring" />
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-fg">Identity confirmed</p>
            <p className="truncate text-xs text-fg-faint">on-device match · 0.4s · inside boundary</p>
          </div>
          <Check size={18} className="ml-auto shrink-0 text-fence" />
        </div>

        {/* mini bar chart */}
        <div className="mt-4 flex items-end gap-1.5">
          {[40, 62, 48, 78, 56, 88, 70, 96, 64, 82].map((h, i) => (
            <motion.span
              key={i}
              initial={{ height: 4 }}
              animate={{ height: h }}
              transition={{ duration: 0.7, delay: 0.4 + i * 0.05, ease: 'easeOut' }}
              className="w-full rounded-t bg-gradient-to-t from-lime/30 to-lime"
              style={{ maxHeight: 96 }}
            />
          ))}
        </div>
      </GlassCard>

      {/* floating chips — positioned on the outer edges so they don't cover card text */}
      <FloatChip
        className="-left-5 bottom-16 sm:-left-9"
        icon={<MapPin size={15} className="text-cyan" />}
        title="Geofence"
        sub="12m from edge"
        delay={0}
      />
      <FloatChip
        className="-right-4 -top-4 sm:-right-8"
        icon={<ShieldCheck size={15} className="text-fence" />}
        title="Compliant"
        sub="0 overstays"
        delay={1.2}
      />
      <FloatChip
        className="-right-5 -bottom-5 sm:-right-9"
        icon={<TrendingUp size={15} className="text-amber-glow" />}
        title="Headcount"
        sub="312 → 313"
        delay={0.6}
      />

      {/* soft shadow lift behind card */}
      <div className="absolute -inset-8 -z-0 rounded-full bg-black/[0.06] blur-3xl" />
    </motion.div>
  )
}

function MiniStat({
  label,
  value,
  tone,
  up,
}: {
  label: string
  value: string
  tone: string
  up?: boolean
}) {
  return (
    <div className="surface rounded-2xl p-3">
      <p className="text-xs text-fg-faint">{label}</p>
      <p className={cn('mt-0.5 text-2xl font-bold', tone)}>
        {value}
        {up && <TrendingUp size={16} className="ml-1 inline text-fence" />}
      </p>
    </div>
  )
}

function FloatChip({
  className,
  icon,
  title,
  sub,
  delay,
}: {
  className?: string
  icon: React.ReactNode
  title: string
  sub: string
  delay: number
}) {
  return (
    <motion.div
      className={cn('absolute z-20 animate-float', className)}
      style={{ animationDelay: `${delay}s` }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.6 + delay * 0.2 }}
    >
      <GlassCard strong className="flex items-center gap-2.5 px-3.5 py-2.5">
        {icon}
        <div className="leading-tight">
          <p className="text-xs font-semibold text-fg">{title}</p>
          <p className="text-[11px] text-fg-faint">{sub}</p>
        </div>
      </GlassCard>
    </motion.div>
  )
}
