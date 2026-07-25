import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal } from '../components/ui/Reveal'

const STEPS = [
  { wk: 'Week 1', title: 'Discovery', body: 'Site survey, your Excel templates analysed, KML boundaries collected.' },
  { wk: 'Week 2', title: 'Import', body: 'Employee list bulk-imported — case-insensitive, zero duplicates.' },
  { wk: 'Week 2–3', title: 'Mobile rollout', body: 'OTA app push to supervisor phones; all active workers calibrated.' },
  { wk: 'Week 3', title: 'Go-live', body: 'Daily attendance via mobile. Legacy systems retired.', live: true },
  { wk: 'Month 2', title: 'Optimisation', body: 'Geofence smart-tuning ends false alerts; AI insights switched on.' },
]

export function Rollout() {
  return (
    <Section id="rollout">
      <SectionHeading
        eyebrow="Implementation · no hardware to ship"
        title="Live in three weeks"
        lede="Including face calibration of 1,000+ workers. No scanners to install, no IT department needed."
      />
      <div className="mt-12 grid gap-4 md:grid-cols-5">
        {STEPS.map((s, i) => (
          <Reveal key={s.title} delay={i * 0.08}>
            <GlassCard
              hover
              strong={s.live}
              className={`flex h-full flex-col p-5 ${s.live ? 'glow-brand' : ''}`}
            >
              <div className="flex items-center gap-2">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    s.live ? 'bg-fence animate-pulse-ring' : 'bg-brand-2'
                  }`}
                />
                <span className="font-mono text-xs text-fg-faint">{s.wk}</span>
              </div>
              <h3 className="mt-3 text-base font-bold text-fg">{s.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-fg-muted">{s.body}</p>
              {s.live && (
                <span className="mt-3 inline-block w-fit rounded-full bg-fence/15 px-2.5 py-0.5 text-[11px] font-semibold text-fence">
                  You go live here
                </span>
              )}
            </GlassCard>
          </Reveal>
        ))}
      </div>
    </Section>
  )
}
