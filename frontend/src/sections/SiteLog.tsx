import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal } from '../components/ui/Reveal'
import { ArrowRight } from 'lucide-react'

const LOG = [
  {
    when: 'MON 07:40',
    who: 'Project Director · Site Vida',
    q: '"Where are my carpenters today?"',
    body: 'Distribution List → Carpenter Finishing: 18 at Site A, 4 at Site B, 0 at Site C. Manpower Recs has already pre-computed the redeployment.',
    result: 'crews rebalanced before the 9 AM pour',
  },
  {
    when: 'TUE 09:02',
    who: 'Operations · Opal Garden',
    q: '"The supervisor says no one\'s at work"',
    body: 'Geofence Alerts shows 47 workers out of bounds — the map clusters them at an unmarked site. Crews were being redirected without paperwork.',
    result: 'caught the same morning, not month-end',
  },
  {
    when: 'WED 11:15',
    who: 'HR Manager · Head office',
    q: '"We almost lost a labour card"',
    body: 'Expiry Watchtower flags 4 visas critical (<14 days) and 12 urgent (<30 days). Renewals processed before a single overstay day.',
    result: 'zero AED 250/day penalties this quarter',
  },
  {
    when: 'THU 14:30',
    who: 'Finance · Audit visit',
    q: '"Show me proof of the salary change"',
    body: 'Worker profile → Salary tab: AED 1,800 → 2,400, effective 2025-09-01, approved by the HR Manager. Timeline on screen, argument over.',
    result: 'audit closed in one click',
  },
]

export function SiteLog() {
  return (
    <Section>
      <SectionHeading
        eyebrow="Field log · a week with RocketAttendance"
        title="How it plays out on site"
      />
      <div className="relative mt-12">
        {/* vertical line */}
        <div className="absolute left-[7px] top-2 bottom-2 w-px bg-gradient-to-b from-brand/60 via-black/15 to-transparent md:left-1/2" />
        <div className="flex flex-col gap-6">
          {LOG.map((e, i) => (
            <Reveal key={e.when} delay={(i % 2) * 0.05}>
              <div
                className={`relative flex md:w-1/2 ${
                  i % 2 === 0 ? 'md:pr-10' : 'md:ml-auto md:pl-10'
                } pl-8`}
              >
                {/* node */}
                <span
                  className={`absolute top-6 h-3.5 w-3.5 rounded-full bg-brand-2 ring-4 ring-brand-2/20 ${
                    i % 2 === 0 ? 'left-0 md:-right-[7px] md:left-auto' : 'left-0 md:-left-[7px]'
                  }`}
                />
                <GlassCard hover className="w-full p-6">
                  <div className="flex items-center gap-3 font-mono text-xs text-fg-faint">
                    <span className="rounded-full bg-black/[0.04] px-2.5 py-1 text-fg-muted">{e.when}</span>
                    {e.who}
                  </div>
                  <h3 className="mt-3 text-lg font-bold text-fg">{e.q}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-fg-muted">{e.body}</p>
                  <p className="mt-4 flex items-center gap-2 text-sm font-medium text-fence">
                    <ArrowRight size={15} /> {e.result}
                  </p>
                </GlassCard>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </Section>
  )
}
