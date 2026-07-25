import { Section } from '../components/ui/Section'
import { SectionHeading } from '../components/ui/SectionHeading'
import { GlassCard } from '../components/ui/GlassCard'
import { GlassButton } from '../components/ui/GlassButton'
import { Reveal } from '../components/ui/Reveal'
import { Check } from 'lucide-react'
import { DEMO_MAILTO, PRICING_MAILTO, cn } from '../lib/utils'

const TIERS = [
  {
    name: 'Starter',
    tagline: 'Single site, one supervisor',
    priceNote: 'For pilots & small crews',
    features: [
      'Face attendance + geofencing',
      'Up to 250 workers',
      'Live web dashboard',
      'Excel import / export',
      'Email support',
    ],
    cta: 'Request pricing',
    href: PRICING_MAILTO,
    highlight: false,
  },
  {
    name: 'Growth',
    tagline: 'Multi-site operations',
    priceNote: 'Most popular for contractors',
    features: [
      'Everything in Starter',
      'Unlimited sites & sponsors',
      'AI insights + attrition risk',
      'Distribution matrix (354 trades)',
      'Visa & document watchtower',
      'Site-scoped admin access',
    ],
    cta: 'Book a demo',
    href: DEMO_MAILTO,
    highlight: true,
  },
  {
    name: 'Enterprise',
    tagline: '10,000+ workforce',
    priceNote: 'For groups & labour supply',
    features: [
      'Everything in Growth',
      'Dedicated onboarding team',
      'Custom corporate-format exports',
      'REST API access',
      'Priority SLA + success manager',
    ],
    cta: 'Talk to sales',
    href: DEMO_MAILTO,
    highlight: false,
  },
]

export function Pricing() {
  return (
    <Section id="pricing">
      <SectionHeading
        align="center"
        eyebrow="Simple, per-workforce pricing"
        title="Plans that scale from one site to twenty"
        lede="No hardware to buy, no per-scan fees. Pick a tier and go live in three weeks."
      />
      <div className="mt-12 grid items-stretch gap-5 lg:grid-cols-3">
        {TIERS.map((t, i) => (
          <Reveal key={t.name} delay={i * 0.08} className="h-full">
            <GlassCard
              strong={t.highlight}
              hover
              className={cn(
                'relative flex h-full flex-col p-7',
                t.highlight && 'glow-brand ring-1 ring-brand-2/40',
              )}
            >
              {t.highlight && (
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full bg-[#14141a] px-3 py-1 text-[11px] font-semibold text-white">
                  Most popular
                </span>
              )}
              <h3 className="text-lg font-bold text-fg">{t.name}</h3>
              <p className="mt-1 text-sm text-fg-muted">{t.tagline}</p>
              <p className="mt-4 font-display text-2xl font-extrabold text-gradient-brand">
                Custom
              </p>
              <p className="text-xs text-fg-faint">{t.priceNote}</p>

              <ul className="mt-6 flex flex-1 flex-col gap-2.5 border-t border-black/8 pt-6">
                {t.features.map((f) => (
                  <li key={f} className="flex items-start gap-2 text-sm text-fg-muted">
                    <Check size={15} className="mt-0.5 shrink-0 text-fence" /> {f}
                  </li>
                ))}
              </ul>

              <GlassButton
                href={t.href}
                variant={t.highlight ? 'primary' : 'glass'}
                size="md"
                className="mt-6 w-full"
              >
                {t.cta}
              </GlassButton>
            </GlassCard>
          </Reveal>
        ))}
      </div>
    </Section>
  )
}
