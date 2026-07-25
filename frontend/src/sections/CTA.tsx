import { GlassCard } from '../components/ui/GlassCard'
import { GlassButton } from '../components/ui/GlassButton'
import { Reveal } from '../components/ui/Reveal'
import { Badge } from '../components/ui/Badge'
import { DEMO_MAILTO, PRICING_MAILTO } from '../lib/utils'

export function CTA() {
  return (
    <section id="cta" className="relative py-24">
      <div className="mx-auto max-w-5xl px-5 sm:px-8">
        <Reveal>
          <GlassCard strong className="relative overflow-hidden px-6 py-16 text-center md:px-16">
            {/* soft neutral sheen */}
            <div className="absolute -top-24 left-1/2 h-72 w-72 -translate-x-1/2 rounded-full bg-black/[0.04] blur-3xl" />
            <div className="relative">
              <Badge className="mx-auto">17:00 reports done · built in the UAE for GCC contractors</Badge>
              <h2 className="mx-auto mt-6 max-w-2xl text-4xl font-extrabold text-gradient md:text-5xl">
                Catch the visa expiry before the fine.
              </h2>
              <p className="mx-auto mt-5 max-w-xl text-lg leading-relaxed text-fg-muted">
                Stop running your worksite on paper musters and Excel. See your own sites,
                trades and boundaries inside RocketAttendance in a 30-minute walkthrough.
              </p>
              <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                <GlassButton href={DEMO_MAILTO} variant="primary" size="lg">
                  Book a demo
                </GlassButton>
                <GlassButton href={PRICING_MAILTO} variant="glass" size="lg">
                  Request pricing
                </GlassButton>
              </div>
            </div>
          </GlassCard>
        </Reveal>
      </div>
    </section>
  )
}
