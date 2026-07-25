import { Rocket } from 'lucide-react'
import { LOGIN_URL, DEMO_MAILTO } from '../lib/utils'

const REGIONS = ['UAE', 'KSA', 'Qatar', 'Oman', 'Bahrain', 'Kuwait', 'India']

const COLS = [
  {
    heading: 'Platform',
    links: [
      { label: 'How it works', href: '#how' },
      { label: 'Capabilities', href: '#capabilities' },
      { label: 'AI Insights', href: '#ai' },
      { label: 'Security', href: '#security' },
    ],
  },
  {
    heading: 'Company',
    links: [
      { label: 'Pricing', href: '#pricing' },
      { label: 'Rollout', href: '#rollout' },
      { label: 'FAQ', href: '#faq' },
      { label: 'Book a demo', href: DEMO_MAILTO },
    ],
  },
  {
    heading: 'Account',
    links: [
      { label: 'Log in', href: LOGIN_URL },
      { label: 'Request pricing', href: 'mailto:sales@rocketattendance.com?subject=Pricing%20enquiry' },
    ],
  },
]

export function Footer() {
  const year = new Date().getFullYear()
  return (
    <footer className="relative border-t border-black/8 pt-16 pb-10">
      <div className="mx-auto max-w-6xl px-5 sm:px-8">
        <div className="grid gap-10 md:grid-cols-[1.5fr_1fr_1fr_1fr]">
          <div>
            <a href="#top" className="flex items-center gap-2.5">
              <span className="grid h-8 w-8 place-items-center rounded-xl bg-[#14141a]">
                <Rocket size={17} className="text-white" />
              </span>
              <span className="text-[17px] font-bold text-fg">
                Rocket<span className="text-fg-muted">Attendance</span>
              </span>
            </a>
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-fg-muted">
              Tamper-proof face attendance, geofencing and AI manpower insights for the
              GCC worksite. Works offline. Live in three weeks.
            </p>
          </div>
          {COLS.map((col) => (
            <div key={col.heading}>
              <h4 className="text-sm font-semibold text-fg">{col.heading}</h4>
              <ul className="mt-4 flex flex-col gap-2.5">
                {col.links.map((l) => (
                  <li key={l.label}>
                    <a href={l.href} className="text-sm text-fg-muted transition-colors hover:text-fg">
                      {l.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 flex flex-col gap-3 border-t border-black/8 pt-6 text-sm text-fg-faint md:flex-row md:items-center md:justify-between">
          <p>RocketAttendance © {year} · UAE-region hosting · HTTPS-only, encrypted at rest</p>
          <p className="flex flex-wrap gap-x-2 gap-y-1">
            {REGIONS.map((r, i) => (
              <span key={r}>
                {r}
                {i < REGIONS.length - 1 && <span className="ml-2 text-black/15">·</span>}
              </span>
            ))}
          </p>
        </div>
      </div>
    </footer>
  )
}
