import { Section } from '../components/ui/Section'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal } from '../components/ui/Reveal'
import { CountUp } from '../components/ui/CountUp'

const METRICS = [
  { value: 20000, suffix: '+', label: 'workers tracked per deployment' },
  { value: 354, suffix: '', label: 'construction trades pre-seeded' },
  { value: 5, suffix: '', label: 'worker languages supported' },
  { value: 7, suffix: '', label: 'GCC + India regions hosted' },
]

export function Metrics() {
  return (
    <Section className="py-14">
      <Reveal>
        <GlassCard strong className="relative overflow-hidden px-6 py-10 md:px-12">
          <div className="absolute -right-10 -top-10 h-48 w-48 rounded-full bg-brand-2/20 blur-3xl" />
          <div className="relative grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
            {METRICS.map((m) => (
              <div key={m.label} className="text-center">
                <p className="font-display text-4xl font-extrabold text-gradient-brand md:text-5xl">
                  <CountUp value={m.value} suffix={m.suffix} />
                </p>
                <p className="mt-2 text-sm text-fg-muted">{m.label}</p>
              </div>
            ))}
          </div>
        </GlassCard>
      </Reveal>
    </Section>
  )
}
