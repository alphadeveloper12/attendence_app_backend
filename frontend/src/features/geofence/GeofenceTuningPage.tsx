import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { TriangleAlert, MapPin } from 'lucide-react'
import { getGeofenceTuning, type GeofenceSite } from '../../lib/api/geofence'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { Skeleton } from '../../components/ui/Skeleton'
import { Select } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'
import { GeofenceMap } from '../../components/ui/GeofenceMap'

const WINDOWS = [7, 14, 30, 60, 90].map((d) => ({ value: String(d), label: `Last ${d} days` }))

/** Severity → pill tone (monochrome). */
const sevTone: Record<string, 'solid' | 'outline' | 'soft'> = {
  high: 'solid',
  med: 'outline',
  low: 'soft',
}

/**
 * Geofence Tuning — surfaces where employees' check-ins are failing the
 * geofence. Per site: a map of the boundary with count-scaled miss clusters,
 * an action summary and cluster suggestions (e.g. "extend boundary NE ~40m").
 */
export default function GeofenceTuningPage() {
  const { data: me } = useMe()
  const [days, setDays] = useState('30')
  const [site, setSite] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['geofence-tuning', days, site],
    queryFn: () => getGeofenceTuning({ days: Number(days), site: site || undefined }),
  })

  const siteOptions = (me?.sites ?? []).map((s) => ({ value: String(s.id), label: s.name }))
  const sites = data?.sites ?? []

  return (
    <div>
      <PageHeader
        title="Geofence Tuning"
        subtitle={
          data
            ? `${sites.length} site${sites.length === 1 ? '' : 's'} with check-in misses · ${data.window_days}-day window`
            : 'Analysing…'
        }
        actions={
          <>
            <Select aria-label="Window" value={days} onChange={setDays} options={WINDOWS} className="w-40" />
            <Select aria-label="Filter by site" value={site} onChange={setSite}
              options={siteOptions} placeholder="All sites" className="w-44" />
          </>
        }
      />

      {isLoading ? (
        <div className="space-y-4">
          {Array.from({ length: 2 }).map((_, i) => (
            <Skeleton key={i} className="h-72 rounded-3xl" />
          ))}
        </div>
      ) : sites.length === 0 ? (
        <GlassCard>
          <EmptyState
            title="No misses to tune"
            message="No sites recorded geofence misses in this window — boundaries look healthy."
          />
        </GlassCard>
      ) : (
        <div className="space-y-4">
          {sites.map((s) => (
            <SiteTuningCard key={s.site_id} site={s} />
          ))}
        </div>
      )}
    </div>
  )
}

/** One site's tuning card: map + summary + cluster suggestions. */
function SiteTuningCard({ site }: { site: GeofenceSite }) {
  const polygon = site.polygon.map(([lat, lng]) => ({ lat, lng }))
  const clusters = site.clusters.map((c) => ({
    lat: c.lat,
    lng: c.lon,
    count: c.count,
    severity: c.severity,
  }))
  const a = site.action_summary

  return (
    <GlassCard className="overflow-hidden">
      <div className="grid gap-0 lg:grid-cols-[1fr_360px]">
        {/* Map */}
        <div className="p-4">
          <GeofenceMap polygon={polygon} clusters={clusters} height={340} />
        </div>

        {/* Detail */}
        <div className="border-t border-black/[0.06] p-5 lg:border-l lg:border-t-0">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div>
              <p className="font-bold text-fg">{site.site_name}</p>
              <p className="text-xs text-fg-faint">{site.total_misses} total misses</p>
            </div>
            {site.has_action_items && (
              <Pill tone="solid">
                <TriangleAlert size={11} /> action
              </Pill>
            )}
          </div>

          {/* Summary counts */}
          <div className="mb-4 grid grid-cols-2 gap-2 text-sm">
            <Stat label="Near edge" value={a.near_edge_misses} />
            <Stat label="Far off" value={a.far_misses} />
            <Stat label="Boundary clusters" value={a.boundary_clusters} />
            <Stat label="Off-site clusters" value={a.off_site_clusters} />
          </div>

          {/* Cluster suggestions */}
          <div className="space-y-2">
            {site.clusters.slice(0, 4).map((c, i) => (
              <div key={i} className="rounded-xl bg-black/[0.03] p-3">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="flex items-center gap-1.5 text-sm font-semibold text-fg">
                    <MapPin size={13} className="text-fg-faint" />
                    {c.count} miss{c.count === 1 ? '' : 'es'}
                    {c.direction ? ` · ${c.direction}` : ''}
                  </span>
                  <Pill tone={sevTone[c.severity] ?? 'soft'}>{c.kind.replace('_', ' ')}</Pill>
                </div>
                <p className="text-xs text-fg-muted">
                  {c.suggestion ?? `~${c.avg_distance_m} m from boundary · ${c.unique_users} people`}
                </p>
              </div>
            ))}
            {site.clusters.length > 4 && (
              <p className="text-center text-xs text-fg-faint">
                +{site.clusters.length - 4} more clusters
              </p>
            )}
          </div>
        </div>
      </div>
    </GlassCard>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg bg-black/[0.03] px-3 py-2">
      <p className="text-lg font-bold leading-none text-fg">{value}</p>
      <p className="mt-1 text-xs text-fg-faint">{label}</p>
    </div>
  )
}
