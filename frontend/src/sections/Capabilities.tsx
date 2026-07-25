import {
  Smartphone,
  UserCog,
  MonitorSmartphone,
  Map,
  LayoutGrid,
  FileCheck2,
  Check,
} from 'lucide-react'
import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal } from '../components/ui/Reveal'

const CAPS = [
  {
    glyph: 'Mobile · Worker',
    icon: Smartphone,
    title: 'Tap-to-attend, anywhere',
    body: 'Face recognition with multi-angle calibration — works after a beard, helmet or mask.',
    points: ['Offline queue, syncs on signal', 'One-tap site reassignment', 'English · Arabic · Hindi · Urdu · Bengali'],
  },
  {
    glyph: 'Mobile · Supervisor',
    icon: UserCog,
    title: 'An office in admin mode',
    body: 'Register new workers and review attendance from the phone — no laptop on site.',
    points: ['2-minute onboarding per worker', 'Live map of last-known positions', 'OTA app updates, no store cycle'],
  },
  {
    glyph: 'Web · Dashboard',
    icon: MonitorSmartphone,
    title: 'The whole site at a glance',
    body: 'Present, absent, active sites and every geofence alert — refreshed live, filtered six ways.',
    points: ['Daily 5-bullet executive summary', 'One-click status filters', 'Site-scoped admin access'],
  },
  {
    glyph: 'Sites · Geofencing',
    icon: Map,
    title: 'KML in, boundaries live',
    body: 'Upload Google Earth polygons or set a radius. Per-site shifts, weekday offs, multi-site workers.',
    points: ['Out-of-bounds alerts in real time', 'Smart-tuning kills false alerts', 'Geofenced reports for client billing'],
  },
  {
    glyph: 'Workforce · Distribution',
    icon: LayoutGrid,
    title: 'Every trade, every site',
    body: 'The manpower matrix owners already trust — live, with 354 pre-seeded construction trades.',
    points: ['Department × site, trade × site', 'Exports in your corporate format', 'Board-ready, zero reformatting'],
  },
  {
    glyph: 'Compliance · Audit',
    icon: FileCheck2,
    title: 'Proof on every change',
    body: 'Status moves, transfers and salary increments logged with who, when and why.',
    points: ['Salary history with effective dates', 'Sick leave + medical certificates', 'Sponsor-level data segregation'],
  },
]

export function Capabilities() {
  return (
    <Section id="capabilities">
      <SectionHeading
        eyebrow="09:00 operations running"
        title="Everything between the face scan and the board report"
        lede="Workers carry the scanner in their pocket. Managers see the whole workforce without leaving the desk."
      />
      <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {CAPS.map((c, i) => {
          const Icon = c.icon
          return (
            <Reveal key={c.title} delay={(i % 3) * 0.08}>
              <GlassCard hover className="flex h-full flex-col p-6">
                <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-brand-2">
                  <Icon size={16} /> {c.glyph}
                </div>
                <h3 className="mt-3 text-lg font-bold text-fg">{c.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-fg-muted">{c.body}</p>
                <ul className="mt-4 flex flex-col gap-2 border-t border-black/8 pt-4">
                  {c.points.map((p) => (
                    <li key={p} className="flex items-start gap-2 text-sm text-fg-muted">
                      <Check size={15} className="mt-0.5 shrink-0 text-fence" />
                      {p}
                    </li>
                  ))}
                </ul>
              </GlassCard>
            </Reveal>
          )
        })}
      </div>
    </Section>
  )
}
