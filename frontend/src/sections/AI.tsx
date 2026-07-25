import { Sparkles, Gauge, Users, MessageSquareText, ShieldCheck } from 'lucide-react'
import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { IconChip, toneByIndex } from '../components/ui/IconChip'
import { Reveal } from '../components/ui/Reveal'

const CARDS = [
  {
    icon: Sparkles,
    title: 'Daily executive summary',
    body: '"4 visas expiring this week · Vida is short 12 carpenters · 9% absence spike at Opal Garden." The whole picture in 30 seconds.',
  },
  {
    icon: Gauge,
    title: 'Attrition risk scoring',
    body: 'A 0–100 score per worker with the specific reasons — attendance drop, sick-leave spike, stale salary — so HR intervenes before the resignation.',
  },
  {
    icon: Users,
    title: 'Manpower recommendations',
    body: '"Move 12 carpenters from GRC W/Shop → Vida." Specific workers, specific quantities. Idle crews redeployed in hours, not weeks.',
  },
  {
    icon: MessageSquareText,
    title: 'Ask the data',
    body: '"How many electricians are on leave at Vida this week?" Plain-English answers for supervisors — no IT ticket required.',
  },
]

export function AI() {
  return (
    <Section id="ai">
      <SectionHeading
        eyebrow="12:00 decisions before lunch"
        title="Your HR team's superpower"
        lede="Insights that used to take a pivot-table afternoon, generated before the morning toolbox talk."
      />
      <div className="mt-12 grid gap-5 sm:grid-cols-2">
        {CARDS.map((c, i) => {
          const Icon = c.icon
          return (
            <Reveal key={c.title} delay={(i % 2) * 0.1}>
              <GlassCard hover className="flex h-full items-start gap-4 p-6">
                <IconChip tone={toneByIndex(i)} className="h-12 w-12">
                  <Icon size={22} />
                </IconChip>
                <div>
                  <h3 className="text-lg font-bold text-fg">{c.title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-fg-muted">{c.body}</p>
                </div>
              </GlassCard>
            </Reveal>
          )
        })}
      </div>
      <Reveal delay={0.15}>
        <GlassCard className="mt-6 flex items-center gap-3 p-5 text-sm text-fg-muted">
          <ShieldCheck size={18} className="shrink-0 text-fence" />
          Every AI feature has a deterministic fallback — remove the AI key and the same
          insights run on built-in rules. Zero dependency, zero hidden cost.
        </GlassCard>
      </Reveal>
    </Section>
  )
}
