import { Marquee } from '../components/ui/Marquee'

// Placeholder brand marks — swap for real client logos when available.
const LOGOS = [
  'ARABTEC', 'AL FUTTAIM', 'BINLADIN', 'HABTOOR', 'DUTCO',
  'KHANSAHEB', 'SSANGYONG', 'NAKHEEL', 'EMAAR', 'ASGC',
]

export function TrustBar() {
  return (
    <div className="relative border-y border-black/5 py-8">
      <div className="mx-auto max-w-6xl px-5 sm:px-8">
        <p className="mb-6 text-center text-xs uppercase tracking-[0.2em] text-fg-faint">
          Trusted across GCC contracting &amp; labour operations
        </p>
        <Marquee duration={38}>
          {LOGOS.map((l) => (
            <span
              key={l}
              className="select-none px-8 font-display text-lg font-bold tracking-wider text-fg-muted/60 transition-colors hover:text-fg"
            >
              {l}
            </span>
          ))}
        </Marquee>
      </div>
    </div>
  )
}
