import { GradientBackdrop } from '../components/ui/GradientBackdrop'
import { Nav } from '../sections/Nav'
import { Hero } from '../sections/Hero'
import { TrustBar } from '../sections/TrustBar'
import { Stats } from '../sections/Stats'
import { HowItWorks } from '../sections/HowItWorks'
import { Metrics } from '../sections/Metrics'
import { Problems } from '../sections/Problems'
import { Capabilities } from '../sections/Capabilities'
import { ProductShowcase } from '../sections/ProductShowcase'
import { AI } from '../sections/AI'
import { Security } from '../sections/Security'
import { Integrations } from '../sections/Integrations'
import { SiteLog } from '../sections/SiteLog'
import { Testimonials } from '../sections/Testimonials'
import { Rollout } from '../sections/Rollout'
import { Pricing } from '../sections/Pricing'
import { FAQ } from '../sections/FAQ'
import { CTA } from '../sections/CTA'
import { Footer } from '../sections/Footer'

export default function HomePage() {
  return (
    <div id="top" className="relative min-h-screen text-fg">
      <GradientBackdrop />
      <Nav />
      <main>
        <Hero />
        <TrustBar />
        <Stats />
        <HowItWorks />
        <Metrics />
        <Problems />
        <Capabilities />
        <ProductShowcase />
        <AI />
        <Security />
        <Integrations />
        <SiteLog />
        <Testimonials />
        <Rollout />
        <Pricing />
        <FAQ />
        <CTA />
      </main>
      <Footer />
    </div>
  )
}
