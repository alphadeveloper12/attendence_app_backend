import { ScanFace, MapPinned, LayoutDashboard } from 'lucide-react'
import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { IconChip, toneByIndex } from '../components/ui/IconChip'
import { Reveal } from '../components/ui/Reveal'

const STEPS = [
  {
    n: '01 / FACE',
    icon: ScanFace,
    badge: 'Step 1 · on-device',
    title: 'The face is the badge',
    body: 'Multi-angle calibration matched on the phone itself — beard, helmet or mask, no internet required.',
    verdict: '✓ identity confirmed · 0.4s',
  },
  {
    n: '02 / GEOFENCE',
    icon: MapPinned,
    badge: 'Step 2 · GPS + polygon',
    title: 'Inside the boundary, or no entry',
    body: 'Position tested against your KML site polygon at capture time. Off-site? The submit button never lights up.',
    verdict: '✓ inside Site Vida · 12m from edge',
  },
  {
    n: '03 / DASHBOARD',
    icon: LayoutDashboard,
    badge: 'Step 3 · live sync',
    title: 'The dashboard already knows',
    body: 'Present counts, trade distribution and alerts update live — or queue offline and sync the moment signal returns.',
    verdict: '✓ Vida headcount 312 → 313',
  },
]

export function HowItWorks() {
  return (
    <Section id="how">
      <SectionHeading
        eyebrow="06:00 what happens at every check-in"
        title="Three checks in one tap"
        lede="Each scan runs three verifications before a single attendance record is written. Watch the pipeline."
      />
      <div className="mt-12 grid gap-5 md:grid-cols-3">
        {STEPS.map((s, i) => {
          const Icon = s.icon
          return (
            <Reveal key={s.n} delay={i * 0.1}>
              <GlassCard hover className="flex h-full flex-col p-6">
                <div className="mb-5 flex items-center justify-between">
                  <IconChip tone={toneByIndex(i)} className="h-12 w-12">
                    <Icon size={22} />
                  </IconChip>
                  <span className="surface rounded-full px-3 py-1 font-mono text-[11px] text-fg-faint">
                    {s.badge}
                  </span>
                </div>
                <span className="font-mono text-xs tracking-widest text-lime">{s.n}</span>
                <h3 className="mt-2 text-xl font-bold text-fg">{s.title}</h3>
                <p className="mt-2 flex-1 text-sm leading-relaxed text-fg-muted">{s.body}</p>
                <p className="mt-5 rounded-xl bg-fence/10 px-3 py-2 font-mono text-xs text-fence">
                  {s.verdict}
                </p>
              </GlassCard>
            </Reveal>
          )
        })}
      </div>
    </Section>
  )
}
