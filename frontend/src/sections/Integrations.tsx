import { Section } from '../components/ui/Section'
import { GlassCard } from '../components/ui/GlassCard'
import { Reveal } from '../components/ui/Reveal'
import { SectionHeading } from '../components/ui/SectionHeading'
import { IconChip, toneByIndex } from '../components/ui/IconChip'
import {
  FileSpreadsheet, Map, Globe, Smartphone, Code2, FileText, Bell, Database,
} from 'lucide-react'

const ITEMS = [
  { icon: FileSpreadsheet, label: 'Excel import / export' },
  { icon: Map, label: 'Google Earth KML' },
  { icon: Globe, label: 'Google Maps geofencing' },
  { icon: Smartphone, label: 'OTA mobile updates' },
  { icon: Code2, label: 'REST API' },
  { icon: FileText, label: 'PDF board reports' },
  { icon: Bell, label: 'Expiry alerts' },
  { icon: Database, label: 'Corporate-format exports' },
]

export function Integrations() {
  return (
    <Section className="py-16">
      <SectionHeading
        align="center"
        eyebrow="Fits how you already work"
        title="Plugs into your existing worksite stack"
        lede="No rip-and-replace. Import your spreadsheets, upload your boundaries, export in your format."
      />
      <Reveal className="mt-10">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {ITEMS.map((it, i) => {
            const Icon = it.icon
            return (
              <GlassCard
                key={it.label}
                hover
                className="flex flex-col items-center gap-3 p-6 text-center"
              >
                <IconChip tone={toneByIndex(i)} className="h-12 w-12">
                  <Icon size={22} />
                </IconChip>
                <span className="text-sm font-medium text-fg-muted">{it.label}</span>
              </GlassCard>
            )
          })}
        </div>
      </Reveal>
    </Section>
  )
}
