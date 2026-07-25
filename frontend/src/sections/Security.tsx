import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal } from '../components/ui/Reveal'
import { Lock, Server, Fingerprint, Users, FileClock, ScrollText } from 'lucide-react'

const ITEMS = [
  { icon: Fingerprint, title: 'On-device biometrics', body: 'Face matching runs on the phone. No biometric template ever leaves the device or hits your network.' },
  { icon: Server, title: 'UAE-region hosting', body: 'Data residency in-region with HTTPS-only transport and encryption at rest across every environment.' },
  { icon: Users, title: 'Sponsor-level segregation', body: 'Multi-sponsor, multi-employer data is walled off. Site-scoped admins only ever see their own workforce.' },
  { icon: ScrollText, title: 'Immutable audit trail', body: 'Every status move, transfer and salary change logged with who, when and why — export-ready for auditors.' },
  { icon: FileClock, title: 'Compliance automation', body: 'Daily visa & document expiry scans bucket every worker into 90 / 30 / 14-day countdowns automatically.' },
  { icon: Lock, title: 'Role-based access', body: 'Granular, superuser-gated navigation and read-only modes keep the right data in the right hands.' },
]

export function Security() {
  return (
    <Section id="security">
      <SectionHeading
        eyebrow="Enterprise-grade by default"
        title="Security your compliance team will sign off on"
        lede="Built for regulated GCC labour operations — data residency, segregation and audit trails are not add-ons."
      />
      <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {ITEMS.map((it, i) => {
          const Icon = it.icon
          return (
            <Reveal key={it.title} delay={(i % 3) * 0.08}>
              <GlassCard hover className="flex h-full flex-col p-6">
                <span className="grid h-11 w-11 place-items-center rounded-2xl bg-black/[0.04] text-brand-2">
                  <Icon size={20} />
                </span>
                <h3 className="mt-4 text-base font-bold text-fg">{it.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-fg-muted">{it.body}</p>
              </GlassCard>
            </Reveal>
          )
        })}
      </div>
    </Section>
  )
}
