import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal } from '../components/ui/Reveal'
import { Quote } from 'lucide-react'

const QUOTES = [
  {
    quote:
      'Month-end reporting went from three days of spreadsheets to a 30-minute review. The board asks for a number and it is already on screen.',
    name: 'HR Director',
    role: 'GCC contracting group · 6,000 workers',
    initials: 'HR',
  },
  {
    quote:
      'The geofence caught crews being redirected to an unmarked site the same morning. We used to find that out at month-end, if at all.',
    name: 'Operations Manager',
    role: 'Infrastructure · multi-site',
    initials: 'OM',
  },
  {
    quote:
      'Zero overstay penalties since go-live. The visa watchtower simply does not let a labour card slip through anymore.',
    name: 'PRO / Compliance Lead',
    role: 'Labour supply · 12 sponsors',
    initials: 'PC',
  },
]

export function Testimonials() {
  return (
    <Section id="testimonials">
      <SectionHeading
        align="center"
        eyebrow="From the site office"
        title="Owners run the worksite from one screen"
      />
      <div className="mt-12 grid gap-5 lg:grid-cols-3">
        {QUOTES.map((q, i) => (
          <Reveal key={q.name} delay={i * 0.1}>
            <GlassCard hover className="flex h-full flex-col p-7">
              <Quote size={28} className="text-brand-2/60" />
              <p className="mt-4 flex-1 text-[15px] leading-relaxed text-fg">"{q.quote}"</p>
              <div className="mt-6 flex items-center gap-3 border-t border-black/8 pt-5">
                <span className="surface grid h-11 w-11 place-items-center rounded-full text-sm font-bold text-fg">
                  {q.initials}
                </span>
                <div>
                  <p className="text-sm font-semibold text-fg">{q.name}</p>
                  <p className="text-xs text-fg-faint">{q.role}</p>
                </div>
              </div>
            </GlassCard>
          </Reveal>
        ))}
      </div>
    </Section>
  )
}
