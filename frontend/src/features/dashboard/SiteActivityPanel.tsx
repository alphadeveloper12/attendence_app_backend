import { useQuery } from '@tanstack/react-query'
import { Radio, AlertTriangle } from 'lucide-react'
import { getSiteActivity, type SiteActivityRow } from '../../lib/api/dashboard'
import { GlassCard } from '../../components/ui/GlassCard'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { Pill } from '../../components/ui/Pill'
import { cn } from '../../lib/utils'

/** Activity band → pill tone + human label. */
const STATUS_META: Record<string, { tone: 'solid' | 'outline' | 'soft'; label: string }> = {
  active: { tone: 'solid', label: 'Active' },
  idle: { tone: 'outline', label: 'Idle' },
  inactive: { tone: 'soft', label: 'Inactive' },
  never_used: { tone: 'soft', label: 'Never used' },
}

function statusMeta(status: string) {
  return STATUS_META[status] ?? { tone: 'soft' as const, label: status.replace(/_/g, ' ') }
}

/** "3h ago" style relative label from hours-since-last-attendance. */
function lastSeen(hours: number | null): string {
  if (hours == null) return 'never'
  if (hours < 1) return `${Math.round(hours * 60)}m ago`
  if (hours < 24) return `${Math.round(hours)}h ago`
  return `${Math.round(hours / 24)}d ago`
}

/**
 * Site Activity Report — a live, per-site health panel: totals across all sites,
 * a callout for sites that have gone quiet, and one card per site showing
 * present/assigned counts, the present rate, last-seen time and a 7-day
 * check-in sparkline. Clicking a card filters the dashboard to that site.
 * Polls every 60s to stay current.
 */
export function SiteActivityPanel({ onSelectSite }: { onSelectSite: (siteId: number) => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ['site-activity'],
    queryFn: () => getSiteActivity(),
    refetchInterval: 60_000,
  })

  const t = data?.totals
  const sites = data?.sites ?? []
  const inactive = data?.inactive_sites ?? []

  return (
    <GlassCard className="p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="grid h-8 w-8 place-items-center rounded-xl bg-black/[0.05] text-fg"><Radio size={16} /></span>
          <div>
            <h2 className="font-semibold text-fg">Site activity report</h2>
            {t && (
              <p className="text-xs text-fg-muted">
                As of {data!.as_of} · {t.present_today}/{t.active_employees} present ({t.overall_present_rate}%)
              </p>
            )}
          </div>
        </div>
        {t && (
          <div className="flex flex-wrap items-center gap-1.5">
            <Pill tone="solid">{t.sites_active} active</Pill>
            <Pill tone="outline">{t.sites_idle} idle</Pill>
            <Pill tone="soft">{t.sites_inactive} inactive</Pill>
            {t.sites_never_used > 0 && <Pill tone="soft">{t.sites_never_used} never used</Pill>}
          </div>
        )}
      </div>

      {/* Callout for quiet sites */}
      {inactive.length > 0 && (
        <div className="mb-4 flex items-start gap-2 rounded-xl bg-black/[0.03] px-3 py-2.5 text-sm text-fg">
          <AlertTriangle size={15} className="mt-0.5 shrink-0 text-fg-muted" />
          <p>
            <span className="font-semibold">{inactive.length}</span> site{inactive.length === 1 ? '' : 's'} not using the
            system: {inactive.map((s) => s.site_name).join(', ')}
          </p>
        </div>
      )}

      {isLoading ? (
        <SkeletonRows rows={4} />
      ) : sites.length === 0 ? (
        <p className="py-6 text-center text-sm text-fg-muted">No sites.</p>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {sites.map((s) => (
            <SiteCard key={s.site_id} site={s} onClick={() => onSelectSite(s.site_id)} />
          ))}
        </div>
      )}
    </GlassCard>
  )
}

/** One site's activity card with a 7-day check-in sparkline. */
function SiteCard({ site, onClick }: { site: SiteActivityRow; onClick: () => void }) {
  const meta = statusMeta(site.activity_status)
  const peak = Math.max(1, ...site.trend_7d)

  return (
    <button
      type="button"
      onClick={onClick}
      className="glass glass-hover flex flex-col gap-3 rounded-2xl p-4 text-left transition-all"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="truncate font-semibold text-fg">{site.site_name}</p>
        <Pill tone={meta.tone}>{meta.label}</Pill>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <Stat label="Present" value={site.present_today} />
        <Stat label="Assigned" value={site.active_employees} />
        <Stat label="Rate" value={`${site.present_rate}%`} />
      </div>

      <div className="flex items-end justify-between gap-2">
        <span className="text-xs text-fg-faint">seen {lastSeen(site.hours_since_last)}</span>
        <div className="flex h-8 items-end gap-0.5" aria-hidden>
          {site.trend_7d.map((v, i) => (
            <span
              key={i}
              className={cn('w-1.5 rounded-sm', i === 6 ? 'bg-[#14141a]' : 'bg-black/20')}
              style={{ height: `${Math.max(10, (v / peak) * 100)}%` }}
            />
          ))}
        </div>
      </div>
    </button>
  )
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div>
      <p className="text-base font-bold leading-none text-fg tabular-nums">{value}</p>
      <p className="mt-1 text-[11px] text-fg-faint">{label}</p>
    </div>
  )
}
