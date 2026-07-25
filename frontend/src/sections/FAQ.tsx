import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { Accordion, type QA } from '../components/ui/Accordion'
import { Reveal } from '../components/ui/Reveal'

const FAQS: QA[] = [
  {
    q: 'Does it work offline in remote camps?',
    a: 'Yes. Face matching and geofence checks run entirely on the phone. Attendance queues locally and syncs automatically the moment signal returns — no connection needed at capture time.',
  },
  {
    q: 'Where is our data stored, and is biometric data safe?',
    a: 'Data is hosted in-region (UAE) over HTTPS and encrypted at rest. Biometric face templates never leave the device — only a verified attendance record is sent, so no biometric data touches your network or our servers.',
  },
  {
    q: 'How long does it take to go live?',
    a: 'Typically three weeks — including bulk-importing your employee list and calibrating 1,000+ worker faces. There is no hardware to ship or install; supervisors use the phones they already carry.',
  },
  {
    q: 'Can it handle multiple sites, sponsors and employers?',
    a: 'That is exactly what it is built for. Multi-site, multi-sponsor and multi-employer data is segregated, and admins can be scoped to only the sites they manage.',
  },
  {
    q: 'What happens if we do not enable the AI features?',
    a: 'Every AI feature has a deterministic fallback. Remove the AI key and the same executive summaries, attrition scores and manpower recommendations run on built-in rules — zero dependency, zero hidden cost.',
  },
  {
    q: 'How do we get our existing worker list in?',
    a: 'Bulk Excel import with automatic column mapping and case-insensitive matching. It de-duplicates as it goes, so 100 workers can be onboarded in a morning with no cleanup sprint.',
  },
]

export function FAQ() {
  return (
    <Section id="faq">
      <SectionHeading align="center" eyebrow="Questions, answered" title="Frequently asked" />
      <Reveal className="mx-auto mt-10 max-w-3xl">
        <Accordion items={FAQS} />
      </Reveal>
    </Section>
  )
}
