import { Section } from '../components/ui/Section'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal, RevealGroup } from '../components/ui/Reveal'
import { motion } from 'framer-motion'

const STATS = [
  { num: '2–4%', label: 'payroll recovered from buddy-punching' },
  { num: 'AED 150K', label: 'visa overstay fines avoided / year' },
  { num: '30 min', label: 'monthly reporting — down from 3 days' },
  { num: '3 weeks', label: 'to go live, incl. 1,000+ face calibrations' },
]

const item = {
  hidden: { opacity: 0, y: 20 },
  shown: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } },
}

export function Stats() {
  return (
    <Section className="pt-4 md:pt-6">
      <RevealGroup className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STATS.map((s) => (
          <motion.div key={s.label} variants={item}>
            <GlassCard hover className="h-full p-6">
              <p className="font-display text-3xl font-extrabold text-gradient-brand md:text-4xl">
                {s.num}
              </p>
              <p className="mt-2 text-sm leading-relaxed text-fg-muted">{s.label}</p>
            </GlassCard>
          </motion.div>
        ))}
      </RevealGroup>
    </Section>
  )
}
