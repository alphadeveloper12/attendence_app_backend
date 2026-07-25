import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal } from '../components/ui/Reveal'

const PROBLEMS = [
  {
    was: 'Manual supervisor checks',
    title: 'Buddy punching ends',
    body: 'On-device face verification means each scan is the worker. No silicon thumbs, no marking in for a friend.',
  },
  {
    was: 'Reactive PM complaints',
    title: 'Off-site sign-ins caught live',
    body: "Mandatory geofence check at capture — a worker can't even submit attendance from outside the boundary.",
  },
  {
    was: 'Excel reminder lists',
    title: 'Visa overstays prevented',
    body: 'Daily auto-scan buckets every document into 90 / 30 / 14-day countdowns before the AED 250/day fine starts.',
  },
  {
    was: 'Phone calls between supervisors',
    title: 'Idle crews redeployed',
    body: "AI spots Site A's surplus against Site B's shortage and suggests the exact move, with quantities.",
  },
  {
    was: '"He just stopped showing up"',
    title: 'Resignations seen coming',
    body: 'Attrition-risk scoring flags the worker — and the reasons — before the resignation letter lands.',
  },
  {
    was: 'Paper forms → data entry',
    title: '100 workers onboarded in a morning',
    body: 'Bulk Excel import with auto column mapping and case-insensitive matching. No duplicates, no cleanup sprint.',
  },
]

export function Problems() {
  return (
    <Section id="problems">
      <SectionHeading
        eyebrow="07:00 the muster sheet retires"
        title="What stops going wrong"
        lede="Built for the way construction actually runs: multi-site, multi-sponsor, multi-employer, multi-shift — and frequently offline."
      />
      <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {PROBLEMS.map((p, i) => (
          <Reveal key={p.title} delay={(i % 3) * 0.08}>
            <GlassCard hover className="h-full p-6">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-alert/10 px-3 py-1 text-xs font-medium text-alert line-through decoration-alert/50">
                {p.was}
              </span>
              <h3 className="mt-4 flex items-center gap-2 text-lg font-bold text-fg">
                {p.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-fg-muted">{p.body}</p>
            </GlassCard>
          </Reveal>
        ))}
      </div>
    </Section>
  )
}
